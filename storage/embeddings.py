"""검색 인덱스 생성과 조회에서 동일한 로컬 임베딩 모델을 제공합니다."""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        # 8GB GPU는 OCR 뒤 Ollama의 9B 모델이 사용합니다. 임베딩 모델까지
        # 상주시킬 경우 Qwen 일부가 CPU로 밀려 전체 생성이 더 느려집니다.
        model_kwargs={"device": "cpu"},
    )
