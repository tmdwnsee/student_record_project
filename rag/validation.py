"""LLM 수정안과 RAG 인용이 입력 및 검색 원문과 일치하는지 검증합니다."""

import re
from pathlib import Path

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    source: str = Field(description="검색 결과에 표시된 PDF 파일명")
    page: int | None = Field(default=None, description="검색 결과의 metadata page. 인쇄 쪽수와 다름")
    content: str = Field(description="검색된 본문에서 그대로 복사한 연속된 근거 문구")


def validate_evidence(evidence: list[Evidence], documents: list[Document]) -> None:
    """인용의 파일명·페이지·본문을 실제 검색 문서와 대조합니다."""
    for item in evidence:
        valid = any(
            item.source == Path(doc.metadata["source"]).name
            and item.page == doc.metadata["page"]
            and bool(item.content.strip())
            and " ".join(item.content.split()) in " ".join(doc.page_content.split())
            for doc in documents
        )
        if not valid:
            raise ValueError("[근거 검증] 검색 원문과 일치하지 않는 인용입니다. 다시 분석하세요.")


def select_evidence(evidence_ids: list[int], documents: list[Document]) -> list[Evidence]:
    """유효한 검색 결과 번호만 실제 metadata 및 원문으로 변환합니다."""
    evidence = []
    for index in dict.fromkeys(evidence_ids):
        if not 1 <= index <= len(documents):
            raise ValueError("[근거 검증] 검색 결과에 없는 근거 번호입니다. 다시 분석하세요.")
        document = documents[index - 1]
        evidence.append(Evidence(
            source=Path(document.metadata["source"]).name,
            page=document.metadata["page"],
            content=document.page_content,
        ))
    return evidence


def looks_like_record_sentence(text: str, original: str = "") -> bool:
    """수정안에 지시문·근거 설명·과도한 새 내용이 섞였는지 확인합니다."""
    compact = " ".join(text.split())
    forbidden = (
        "작성할 것", "서술할 것", "포함할 것", "강조할 것", "입력한다",
        "수정 이유", "평가 기준", "평가기준", "반영 비율", "모집요강",
        "작성요령", "metadata page", ".pdf",
    )
    if not compact or any(marker in compact for marker in forbidden):
        return False
    if not original:
        return True

    original_length = len("".join(original.split()))
    revised_length = len("".join(text.split()))
    if revised_length > max(original_length + 25, int(original_length * 1.5)):
        return False

    original_tokens = re.findall(r"[가-힣]{2,}|[A-Za-z]{2,}", original.lower())
    revised_tokens = re.findall(r"[가-힣]{2,}|[A-Za-z]{2,}", text.lower())
    unsupported = [
        token for token in revised_tokens
        if not any(
            token in original_token
            or original_token in token
            or (len(token) >= 3 and len(original_token) >= 3 and token[:2] == original_token[:2])
            for original_token in original_tokens
        )
    ]
    return len(set(unsupported)) <= 2


def conservative_rewrite(student_draft: str) -> str:
    """수정 후보가 검증에 실패했을 때 원문 사실을 보존한 최소 정리본입니다."""
    text = " ".join(student_draft.split()).strip()
    text = re.sub(r"[.!?]+$", "", text)
    text = text.replace("프로젝트를 진행하며", "프로젝트에서")
    text = text.replace("Python으로", "Python을 활용해")
    return text if text.endswith(("함", "임", "됨", "킴", "음")) else f"{text}함"


# 기존 외부 코드가 사용하던 비공개 이름도 당분간 호환합니다.
_looks_like_record_sentence = looks_like_record_sentence
_conservative_rewrite = conservative_rewrite
