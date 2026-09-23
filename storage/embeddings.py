"""검색 인덱스 생성과 조회에서 동일한 로컬 임베딩 모델을 제공합니다."""

import threading
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL


_LOAD_LOCK = threading.Lock()
_WARMUP_LOCK = threading.Lock()
_WARMUP_STARTED = False


@lru_cache(maxsize=1)
def _load_embeddings() -> HuggingFaceEmbeddings:
    options = {
        "model_name": EMBEDDING_MODEL,
        # 8GB GPU는 OCR 뒤 Ollama의 9B 모델이 사용합니다. 임베딩 모델까지
        # 상주시킬 경우 Qwen 일부가 CPU로 밀려 전체 생성이 더 느려집니다.
        "model_kwargs": {"device": "cpu", "local_files_only": True},
    }
    try:
        # 이미 내려받은 모델은 네트워크 확인 없이 즉시 엽니다.
        return HuggingFaceEmbeddings(**options)
    except OSError:
        # 새 환경에서 모델이 전혀 없을 때만 최초 다운로드를 허용합니다.
        options["model_kwargs"] = {"device": "cpu"}
        return HuggingFaceEmbeddings(**options)


def get_embeddings() -> HuggingFaceEmbeddings:
    """동시 요청에서도 큰 임베딩 모델을 한 번만 로드합니다."""
    with _LOAD_LOCK:
        return _load_embeddings()


def warm_up_embeddings() -> None:
    """사용자가 폼을 작성하는 동안 임베딩 모델을 백그라운드에서 준비합니다."""
    global _WARMUP_STARTED
    with _WARMUP_LOCK:
        if _WARMUP_STARTED:
            return
        _WARMUP_STARTED = True
    threading.Thread(target=get_embeddings, name="embedding-warmup", daemon=True).start()
