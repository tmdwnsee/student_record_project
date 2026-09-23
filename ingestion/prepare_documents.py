"""PDF 단계별 추출 파이프라인과 RAG 청킹을 연결합니다."""

from pathlib import Path
import re
import unicodedata

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.pdf_pipeline import PipelineConfig, build_source_documents


def prepare_documents(
    pdf_path: Path, *, page_numbers: tuple[int, ...] | None = None,
    natural_text_order: bool = False,
) -> list[Document]:
    """지정된 페이지를 pdftotext → PyPDF → PyMuPDF → OCR 순서로 추출합니다."""
    pdf_path = pdf_path.resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

    document_type = pdf_path.stem
    documents, summary = build_source_documents(
        pdf_path,
        document_type=document_type,
        # 대학 모집요강의 평가표는 텍스트형입니다. 장식·이미지 페이지 전체에
        # OCR을 실행하지 않아도 되며, 업로드 생기부의 OCR 경로에는 영향이 없습니다.
        config=PipelineConfig(
            enable_ocr=False,
            pymupdf_sort=not natural_text_order,
            prefer_pymupdf=natural_text_order,
        ),
        page_numbers=page_numbers,
    )
    if not documents:
        raise ValueError(f"PDF에서 텍스트를 불러오지 못했습니다: {pdf_path.name}")
    print(
        f"추출 방식: {summary['methods']} / "
        f"검토 필요 페이지: {summary['review_pages']}/{summary['total_pages']}"
    )
    return documents


def normalize_course_key(value: str) -> str:
    """사용자 입력과 PDF 과목명의 공백·기호·로마 숫자 차이를 없앱니다."""
    normalized = unicodedata.normalize("NFKC", value).lower()
    return re.sub(r"[^0-9a-z가-힣]", "", normalized)


def _course_name_from_page(text: str) -> str:
    """교육과정 과목 카드의 '과목명' 바로 다음 값을 읽습니다."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines[:-1]):
        if re.sub(r"\s+", "", line) != "과목명":
            continue
        candidate = lines[index + 1].strip()
        if 1 <= len(candidate) <= 40 and normalize_course_key(candidate):
            return candidate
    return ""


def annotate_curriculum_documents(documents: list[Document]) -> list[Document]:
    """각 과목 카드와 바로 다음 설명 페이지에 동일한 과목 메타데이터를 붙입니다."""
    annotated: list[Document] = []
    previous_course = ""
    previous_page: int | None = None
    for document in sorted(documents, key=lambda item: int(item.metadata.get("page", -1))):
        metadata = dict(document.metadata)
        page = int(metadata.get("page", -1))
        course_name = _course_name_from_page(document.page_content)
        if not course_name and previous_course and previous_page is not None and page == previous_page + 1:
            course_name = previous_course
        if course_name:
            metadata["course_name"] = course_name
            metadata["course_key"] = normalize_course_key(course_name)
        annotated.append(Document(page_content=document.page_content, metadata=metadata))
        # 다음 한 페이지만 이어지는 설명으로 간주합니다.
        previous_course = _course_name_from_page(document.page_content)
        previous_page = page
    return annotated


def prepare_curriculum_documents(pdf_path: Path) -> list[Document]:
    """교육과정 전체를 텍스트로 읽고 과목 단위 검색용 메타데이터를 추가합니다."""
    return annotate_curriculum_documents(
        prepare_documents(pdf_path, natural_text_order=True)
    )


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("생성된 chunk가 없습니다.")
    return chunks
