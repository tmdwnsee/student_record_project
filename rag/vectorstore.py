"""두 PDF의 chunk를 서로 다른 Chroma 저장소에 저장합니다."""

import hashlib
import json
import os
from pathlib import Path

from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from rag.loader import (
    DATA_DIRECTORY, PROJECT_ROOT, load_pdf, split_documents, validate_chunk_metadata,
)

EMBEDDING_MODEL = "text-embedding-3-small"


def check_api_key() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError("[환경 설정] .env에 OPENAI_API_KEY를 설정하세요.")


def sync_vectorstore(
    chunks: list[Document], persist_directory: Path, collection_name: str,
    embedding: OpenAIEmbeddings,
) -> Chroma:
    """새 chunk만 임베딩하고, 성공 후 이전 PDF의 낡은 chunk를 제거합니다."""
    if not chunks:
        raise ValueError("[5단계 저장] 저장할 chunk가 없습니다.")
    store = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        embedding_function=embedding,
        client_settings=Settings(anonymized_telemetry=False),
    )
    # 내용과 metadata가 같으면 재실행해도 동일한 ID를 사용합니다.
    chunk_by_id = {}
    for chunk in chunks:
        payload = json.dumps(
            [EMBEDDING_MODEL, chunk.metadata, chunk.page_content],
            ensure_ascii=False, sort_keys=True,
        )
        chunk_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        chunk_by_id[chunk_id] = chunk

    existing_ids = set(store.get(include=[])["ids"])
    new_ids = sorted(set(chunk_by_id) - existing_ids)
    for start in range(0, len(new_ids), 64):
        batch_ids = new_ids[start:start + 64]
        store.add_documents([chunk_by_id[key] for key in batch_ids], ids=batch_ids)
    stale_ids = sorted(existing_ids - set(chunk_by_id))
    if stale_ids:
        store.delete(ids=stale_ids)
    print(f"[5단계] {collection_name}: 총 {len(chunk_by_id)}개, 신규 {len(new_ids)}개")
    return store


def build_vectorstores() -> tuple[Chroma, Chroma]:
    """PDF 로딩 → chunking → 두 개의 영구 저장소 준비."""
    check_api_key()
    embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL, request_timeout=60, max_retries=1)
    stores = []
    for filename, directory, collection in (
        ("college_table.pdf", "college", "college_collection"),
        ("student_record_rule.pdf", "guideline", "guideline_collection"),
    ):
        documents = load_pdf(DATA_DIRECTORY / filename)
        chunks = split_documents(documents)
        validate_chunk_metadata(documents, chunks)
        try:
            store = sync_vectorstore(
                chunks, PROJECT_ROOT / "vectorstores" / directory, collection, embedding,
            )
        except Exception as error:
            # API 오류 원문에는 키가 포함될 수 있어 사용자 화면에 노출하지 않습니다.
            raise RuntimeError(
                f"[5단계 저장] {filename} 처리 실패 ({type(error).__name__}). "
                "API 키, 잔액, 네트워크 및 저장 폴더를 확인하세요."
            ) from error
        stores.append(store)
    return stores[0], stores[1]
