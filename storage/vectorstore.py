# 이미 만들어놓은 vector store와 연결
# 여기서는 새로 생성하지 않음
# 생성은 ingestion/build_index.py에서 수행

# storage/vectorstore.py

from functools import lru_cache

from chromadb.config import Settings
from langchain_chroma import Chroma

from config import (
    COLLEGE_VECTORSTORE,
    COLLEGE_VECTORSTORES,
    COLLEGE_COLLECTIONS,
    GUIDELINE_VECTORSTORE,
    COLLEGE_COLLECTION,
    GUIDELINE_COLLECTION,
    CURRICULUM_VECTORSTORE,
    CURRICULUM_COLLECTION,
)
from storage.embeddings import get_embeddings


def validate_vectorstore(directory):
    database_file = directory / "chroma.sqlite3"

    if not database_file.exists():
        raise FileNotFoundError(
            f"""
Vector Store를 찾을 수 없습니다.

먼저 아래 명령어를 실행하세요.

python -m ingestion.build_index

경로:
{directory}
"""
        )


def _open_store(directory, collection, embedding):
    validate_vectorstore(directory)
    return Chroma(
        collection_name=collection,
        persist_directory=str(directory),
        embedding_function=embedding,
        client_settings=Settings(anonymized_telemetry=False),
    )


@lru_cache(maxsize=None)
def load_college_vectorstore(university: str = "성균관대학교"):
    if university not in COLLEGE_VECTORSTORES:
        raise ValueError(f"등록되지 않은 대학입니다: {university}")
    return _open_store(
        COLLEGE_VECTORSTORES[university],
        COLLEGE_COLLECTIONS[university],
        get_embeddings(),
    )


@lru_cache(maxsize=1)
def load_vectorstores():
    embedding = get_embeddings()
    return (
        _open_store(COLLEGE_VECTORSTORE, COLLEGE_COLLECTION, embedding),
        _open_store(GUIDELINE_VECTORSTORE, GUIDELINE_COLLECTION, embedding),
    )


@lru_cache(maxsize=1)
def load_guideline_vectorstore():
    embedding = get_embeddings()

    return _open_store(
        GUIDELINE_VECTORSTORE,
        GUIDELINE_COLLECTION,
        embedding,
    )


@lru_cache(maxsize=1)
def load_curriculum_vectorstore():
    return _open_store(
        CURRICULUM_VECTORSTORE,
        CURRICULUM_COLLECTION,
        get_embeddings(),
    )
