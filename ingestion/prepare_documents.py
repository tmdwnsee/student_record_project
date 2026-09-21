"""PDF 단계별 추출 파이프라인과 RAG 청킹을 연결합니다."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.pdf_pipeline import PipelineConfig, build_source_documents


def _document_type(pdf_path: Path) -> str:
    if pdf_path.name == "student_record_rule.pdf":
        return "school_record_guide_2026"
    return pdf_path.stem


def prepare_documents(
    pdf_path: Path, *, page_numbers: tuple[int, ...] | None = None,
    natural_text_order: bool = False,
) -> list[Document]:
    """지정된 페이지를 pdftotext → PyMuPDF → OCR 순서로 추출합니다."""
    pdf_path = pdf_path.resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

    document_type = _document_type(pdf_path)
    documents, summary = build_source_documents(
        pdf_path,
        document_type=document_type,
        # 대학 모집요강의 평가표는 텍스트형입니다. 장식·이미지 페이지 전체에
        # OCR을 실행하지 않아도 되며, 업로드 생기부의 OCR 경로에는 영향이 없습니다.
        config=PipelineConfig(
            enable_ocr=document_type == "school_record_guide_2026",
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
