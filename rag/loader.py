"""PDF를 페이지별로 로드하고 원본 metadata와 텍스트를 확인합니다."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = PROJECT_ROOT / "data"


def split_documents(documents: list[Document]) -> list[Document]:
    """페이지별 문서를 최대 700자 chunk로 분할합니다."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
    )
    # 문자열만 분할하지 않고 Document를 전달해야 원본 metadata가 유지됩니다.
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        raise ValueError("[4단계 chunking] 분할할 텍스트가 없습니다.")
    return chunks


def validate_chunk_metadata(
    documents: list[Document], chunks: list[Document]
) -> None:
    """모든 chunk의 metadata가 출처 페이지의 원본과 같은지 확인합니다."""
    original_metadata = {
        (document.metadata["source"], document.metadata["page"]): document.metadata
        for document in documents
    }
    for index, chunk in enumerate(chunks):
        page_key = (chunk.metadata.get("source"), chunk.metadata.get("page"))
        if chunk.metadata != original_metadata.get(page_key):
            raise ValueError(f"[4단계 metadata 검사] chunk {index}의 metadata가 원본과 다릅니다.")


def load_pdf(pdf_path: Path) -> list[Document]:
    """페이지별 Document를 반환하고 source/page 누락을 검사합니다."""
    pdf_path = pdf_path.resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"[PDF 로딩] 파일을 찾을 수 없습니다: {pdf_path}")
    try:
        documents = PyPDFLoader(str(pdf_path), mode="page").load()
    except Exception as error:
        raise RuntimeError(f"[PDF 로딩] {pdf_path.name}: {error}") from error
    if not documents:
        raise ValueError(f"[PDF 로딩] 페이지가 없습니다: {pdf_path.name}")
    for document in documents:
        if not document.metadata.get("source") or not isinstance(
            document.metadata.get("page"), int
        ):
            raise ValueError(f"[metadata 검사] source/page 오류: {document.metadata}")
    # loader가 제공한 page, page_label 등 metadata를 그대로 보존합니다.
    return documents


def print_document_metadata(documents: list[Document], count: int = 3) -> None:
    """처음 몇 페이지의 실제 metadata를 출력합니다."""
    for document in documents[:count]:
        print(document.metadata)


def inspect_pdf_page(
    documents: list[Document], page_number: int, preview_chars: int = 2000
) -> None:
    """목록 위치가 아닌 metadata['page'] 값으로 페이지를 찾아 출력합니다."""
    for document in documents:
        if document.metadata.get("page") == page_number:
            print("\n" + "=" * 60)
            print("METADATA:", document.metadata)
            print("PAGE (0부터 시작):", document.metadata["page"])
            print("PDF 파일 내 순서 (1부터 시작):", document.metadata["page"] + 1)
            print("PAGE LABEL:", document.metadata.get("page_label", "없음"))
            print("TEXT:")
            print(document.page_content[:preview_chars])
            if not document.page_content.strip():
                print("[텍스트 확인] 추출된 텍스트가 없습니다. 원본 페이지를 확인하세요.")
            return
    raise ValueError(f"[페이지 확인] metadata page={page_number}인 문서가 없습니다.")
