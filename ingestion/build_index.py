# 임베딩 
# pdf 변경 시에만 python -m ingestion.build_index 실행

# ingestion/build_index.py

import hashlib
import json
from ingestion.loader import load_pdf, split_documents
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import (
    check_api_key,
    EMBEDDING_MODEL,
    COLLEGE_PDF,
    GUIDELINE_PDF,
    COLLEGE_VECTORSTORE,
    GUIDELINE_VECTORSTORE,
    COLLEGE_COLLECTION,
    GUIDELINE_COLLECTION,
)

from ingestion.loader import (
    load_pdf,
    split_documents,
)


def sync_vectorstore(
    chunks: list[Document],
    persist_directory,
    collection_name,
    embedding,
):
    """
    현재 PDF chunk를 Vector Store와 동기화.
    동일한 chunk는 재임베딩하지 않음.
    """

    persist_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    store = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        embedding_function=embedding,
        client_settings=Settings(
            anonymized_telemetry=False
        ),
    )

    chunk_by_id = {}

    for chunk in chunks:

        payload = json.dumps(
            [
                EMBEDDING_MODEL,
                chunk.metadata,
                chunk.page_content,
            ],
            ensure_ascii=False,
            sort_keys=True,
        )

        chunk_id = hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()

        chunk_by_id[chunk_id] = chunk

    existing_ids = set(
        store.get(include=[])["ids"]
    )

    current_ids = set(chunk_by_id)

    new_ids = sorted(
        current_ids - existing_ids
    )

    stale_ids = sorted(
        existing_ids - current_ids
    )

    # 새 문서만 임베딩
    for start in range(
        0,
        len(new_ids),
        64,
    ):
        batch_ids = new_ids[start:start + 64]

        store.add_documents(
            documents=[
                chunk_by_id[chunk_id]
                for chunk_id in batch_ids
            ],
            ids=batch_ids,
        )

    # 사라진 chunk 삭제
    if stale_ids:
        store.delete(ids=stale_ids)

    print(
        f"{collection_name}: "
        f"전체 {len(current_ids)}개 / "
        f"신규 {len(new_ids)}개 / "
        f"삭제 {len(stale_ids)}개"
    )


def build_index(
    pdf_path,
    persist_directory,
    collection_name,
    embedding,
):
    print(f"\nPDF 로딩: {pdf_path.name}")

    documents = load_pdf(pdf_path)

    print(
        f"페이지 수: {len(documents)}"
    )

    chunks = split_documents(documents)

    print(
        f"chunk 수: {len(chunks)}"
    )

    sync_vectorstore(
        chunks=chunks,
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding=embedding,
    )


def main():

    check_api_key()

    embedding = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    print("Vector Store 갱신 시작")

    build_index(
        pdf_path=COLLEGE_PDF,
        persist_directory=COLLEGE_VECTORSTORE,
        collection_name=COLLEGE_COLLECTION,
        embedding=embedding,
    )

    build_index(
        pdf_path=GUIDELINE_PDF,
        persist_directory=GUIDELINE_VECTORSTORE,
        collection_name=GUIDELINE_COLLECTION,
        embedding=embedding,
    )

    print("\nVector Store 갱신 완료")


if __name__ == "__main__":
    main()