# 임베딩 
# pdf 변경 시에만 python -m ingestion.build_index 실행

# ingestion/build_index.py

import hashlib
import json
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import (
    EMBEDDING_MODEL,
    COLLEGE_GUIDES,
    COLLEGE_VECTORSTORES,
    COLLEGE_COLLECTIONS,
    COLLEGE_GUIDE_PAGES,
    COLLEGE_NATURAL_TEXT_ORDER,
    CURRICULUM_PDF,
    CURRICULUM_VECTORSTORE,
    CURRICULUM_COLLECTION,
)

from ingestion.prepare_documents import (
    prepare_documents,
    prepare_curriculum_documents,
    split_documents,
)
from storage.embeddings import get_embeddings


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

    return store


def build_index(
    pdf_path,
    persist_directory,
    collection_name,
    embedding,
    *,
    page_numbers=None,
    natural_text_order=False,
):
    print(f"\nPDF 로딩: {pdf_path.name}")

    documents = prepare_documents(
        pdf_path,
        page_numbers=page_numbers,
        natural_text_order=natural_text_order,
    )

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

    embedding = get_embeddings()

    print("Vector Store 갱신 시작")

    # 현재 앱은 대학 맞춤 활동 가이드만 제공하므로 모집요강 인덱스만 만듭니다.
    # 작성요령 검증 기능에서 사용하던 GUIDELINE_* 설정은 이전 데이터와의
    # 호환성을 위해 남겨 두되, 기본 인덱싱 경로에서는 제외합니다.
    for university, pdf_path in COLLEGE_GUIDES.items():
        print(f"\n대학: {university}")
        build_index(
            pdf_path,
            COLLEGE_VECTORSTORES[university],
            COLLEGE_COLLECTIONS[university],
            embedding,
            page_numbers=COLLEGE_GUIDE_PAGES[university],
            natural_text_order=university in COLLEGE_NATURAL_TEXT_ORDER,
        )

    print("\n교육과정")
    curriculum_documents = prepare_curriculum_documents(CURRICULUM_PDF)
    print(f"페이지 수: {len(curriculum_documents)}")
    curriculum_chunks = split_documents(curriculum_documents)
    print(f"chunk 수: {len(curriculum_chunks)}")
    sync_vectorstore(
        chunks=curriculum_chunks,
        persist_directory=CURRICULUM_VECTORSTORE,
        collection_name=CURRICULUM_COLLECTION,
        embedding=embedding,
    )

    print("\nVector Store 갱신 완료")


if __name__ == "__main__":
    main()
