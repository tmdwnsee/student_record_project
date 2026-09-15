# 이미 만들어놓은 vector store와 연결
# 여기서는 새로 생성하지 않음
# 생성은 ingestion/build_index.py에서 수행

# storage/vectorstore.py

from functools import lru_cache

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import (
    check_api_key,
    EMBEDDING_MODEL,
    COLLEGE_VECTORSTORE,
    GUIDELINE_VECTORSTORE,
    COLLEGE_COLLECTION,
    GUIDELINE_COLLECTION,
)


def validate_vectorstore(
    directory
):
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


@lru_cache(maxsize=1)
def load_vectorstores():

    check_api_key()

    validate_vectorstore(
        COLLEGE_VECTORSTORE
    )

    validate_vectorstore(
        GUIDELINE_VECTORSTORE
    )

    embedding = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    college_store = Chroma(
        collection_name=COLLEGE_COLLECTION,
        persist_directory=str(
            COLLEGE_VECTORSTORE
        ),
        embedding_function=embedding,
        client_settings=Settings(
            anonymized_telemetry=False
        ),
    )

    guideline_store = Chroma(
        collection_name=GUIDELINE_COLLECTION,
        persist_directory=str(
            GUIDELINE_VECTORSTORE
        ),
        embedding_function=embedding,
        client_settings=Settings(
            anonymized_telemetry=False
        ),
    )

    return (
        college_store,
        guideline_store,
    )