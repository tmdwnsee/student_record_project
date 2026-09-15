# pdf 로딩, 청킹

# ingestion/loader.py

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(pdf_path: Path) -> list[Document]:

    pdf_path = pdf_path.resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF 파일을 찾을 수 없습니다: {pdf_path}"
        )

    documents = PyPDFLoader(
        str(pdf_path),
        mode="page",
    ).load()

    if not documents:
        raise ValueError(
            f"PDF에서 텍스트를 불러오지 못했습니다: {pdf_path.name}"
        )

    return documents


def split_documents(
    documents: list[Document]
) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "생성된 chunk가 없습니다."
        )

    return chunks