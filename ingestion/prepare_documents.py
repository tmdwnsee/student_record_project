"""PDF 단계별 추출 파이프라인과 RAG 청킹을 연결합니다."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.pdf_pipeline import build_source_documents, targeted_pages_for_document


def _document_type(pdf_path: Path) -> str:
    if pdf_path.name == "college_table.pdf":
        return "skku_2027"
    if pdf_path.name == "student_record_rule.pdf":
        return "school_record_guide_2026"
    return pdf_path.stem


def prepare_documents(pdf_path: Path) -> list[Document]:
    """pdftotext → PyMuPDF → OCR 순서로 PDF 전체를 페이지별 추출합니다."""
    pdf_path = pdf_path.resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

    document_type = _document_type(pdf_path)
    documents, summary = build_source_documents(
        pdf_path,
        document_type=document_type,
        expected_terms_by_page=targeted_pages_for_document(document_type),
    )
    if not documents:
        raise ValueError(f"PDF에서 텍스트를 불러오지 못했습니다: {pdf_path.name}")
    print(
        f"추출 방식: {summary['methods']} / "
        f"검토 필요 페이지: {summary['review_pages']}/{summary['total_pages']}"
    )
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("생성된 chunk가 없습니다.")
    return chunks
