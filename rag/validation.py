"""LLM 수정안과 RAG 인용이 입력 및 검색 원문과 일치하는지 검증합니다."""

import re
from pathlib import Path

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    source: str = Field(description="검색 결과에 표시된 PDF 파일명")
    page: int | None = Field(default=None, description="검색 결과의 metadata page. 인쇄 쪽수와 다름")
    content: str = Field(description="검색된 본문에서 그대로 복사한 연속된 근거 문구")
    retrieval_score: float | None = None
    retrieval_reasons: list[str] = Field(default_factory=list)
    guideline_scopes: list[str] = Field(
        default_factory=list
    )
    retrieval_keywords: list[str] = Field(default_factory=list)
    retrieval_keyword_groups: dict[str, list[str]] = Field(default_factory=dict)
    draft_matches: list[str] = Field(default_factory=list)
    draft_matches_by_category: dict[str, list[str]] = Field(default_factory=dict)


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
        evidence.append(
            Evidence(
                source=Path(
                    document.metadata["source"]
                ).name,

                page=document.metadata["page"],

                content=document.page_content,

                retrieval_score=document.metadata.get(
                    "retrieval_score"
                ),

                retrieval_reasons=document.metadata.get(
                    "retrieval_reasons",
                    []
                ),

                guideline_scopes=document.metadata.get(
                    "guideline_scopes",
                    []
                ),

                retrieval_keywords=document.metadata.get(
                    "retrieval_keywords",
                    []
                ),

                retrieval_keyword_groups=document.metadata.get(
                    "retrieval_keyword_groups",
                    {}
                ),

                draft_matches=document.metadata.get(
                    "draft_matches",
                    []
                ),

                draft_matches_by_category=document.metadata.get(
                    "draft_matches_by_category",
                    {}
                ),
            )
        )
    return evidence


def looks_like_record_sentence(
    text: str,
) -> bool:

    compact = " ".join(
        text.split()
    )

    forbidden = (
        "작성할 것",
        "서술할 것",
        "포함할 것",
        "강조할 것",
        "입력한다",
        "수정 이유",
        "평가 기준",
        "평가기준",
        "반영 비율",
        "모집요강",
        "작성요령",
        "metadata page",
        ".pdf",
    )

    return (
        bool(compact)
        and not any(
            marker in compact
            for marker in forbidden
        )
    )

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
