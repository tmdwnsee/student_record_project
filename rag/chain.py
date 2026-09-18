"""RAG 검색부터 검증까지 정해진 실행 순서만 관리합니다."""

import re

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from config import MODEL, OLLAMA_BASE_URL
from rag.future import generate_future_guide
from rag.criteria import CollegeCriterion, extract_college_criteria
from rag.retriever import retrieve_college_context, retrieve_guideline_context
from rag.reviewer import (
    DraftDiagnosis,
    ReviewResult,
    build_revision_reason,
    diagnose_draft,
    rewrite_student_draft,
)
from rag.validation import (
    Evidence,
    conservative_rewrite,
    looks_like_record_sentence,
    select_evidence,
    validate_evidence,
)
from storage.vectorstore import (
    load_guideline_vectorstore,
    load_vectorstores,
)

VALID_RECORD_SECTIONS = (
    "세특",
    "자율자치활동",
    "동아리활동",
    "진로활동",
    "행특",
)

SAMPLE_DRAFT = "데이터 분석 프로젝트를 진행하며 Python을 활용해 데이터를 분석하고 문제 해결 능력을 향상하였다."


class RecordGroundedExamples(BaseModel):
    examples: list[str] = Field(
        description=(
            "생기부 원문에 실제로 확인되는 활동 사건과 과정을 바탕으로 작성한 3개의 예시 문장. "
            "원문에 없는 활동, 결과, 역할, 수치, 감정은 추가하지 않음"
        )
    )

# 이전 import 경로를 사용하던 코드와 테스트의 호환성을 유지합니다.
_conservative_rewrite = conservative_rewrite
_looks_like_record_sentence = looks_like_record_sentence


def generate_review(
    student_draft: str,
    record_section: str,
    guideline_results: list[Document],
) -> ReviewResult:
    """추출 → 진단 → 수정 → 근거 선택 → 검증 순서로 검토 결과를 만듭니다."""
    if not student_draft.strip():
        raise ValueError("생기부 초안을 입력하세요.")

    model = ChatOllama(
        model=MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        reasoning=False,
    )

    try:
        diagnosis = diagnose_draft(
            model=model,
            student_draft=student_draft,
            record_section=record_section,
            guideline_results=guideline_results,
        )
        revised_text = rewrite_student_draft(
            model=model,
            student_draft=student_draft,
            record_section=record_section,
            diagnosis=diagnosis,
            guideline_results=guideline_results,
        )
    except Exception as error:
        raise RuntimeError(
            f"[초안 진단/수정안 생성 실패] "
            f"{type(error).__name__}: {error}"
        ) from error

    # 재정렬된 작성요령 5개를 Streamlit 근거 영역에 모두 표시합니다.
    guideline_evidence_ids = list(range(1, len(guideline_results) + 1))
    guideline_evidence = select_evidence(guideline_evidence_ids, guideline_results)

    if diagnosis.missing_details:
        caution = (
            "확인이 필요한 항목:\n- "
            + "\n- ".join(
                diagnosis.missing_details
            )
        )
    else:
        caution = (
            "초안에서 추가로 확인할 "
            "구체적 정보가 없습니다."
        )

    caution += (
        "\n\n초안에서 확인되지 않은 내용은 "
        "수정안에 추가하지 않았습니다."
    )


    revision_reason = build_revision_reason(
        student_draft,
        revised_text,
        record_section,
        diagnosis,
    )


    references = list(
        dict.fromkeys(
            f"{item.source} "
            f"(metadata page: {item.page})"
            for item in guideline_evidence
        )
    )

    if references:
        revision_reason += (
            "\n\n사용한 작성요령 근거: "
            + ", ".join(references)
        )

    result = ReviewResult(
        original_text=student_draft,
        revised_text=revised_text,
        revision_reason=revision_reason,
        guideline_evidence=guideline_evidence,
        caution=caution,
        record_section=record_section,
    )
    validate_evidence(result.guideline_evidence, guideline_results)
    return result



def review_draft(
    student_draft: str,
    record_section: str,
) -> ReviewResult:

    if not student_draft.strip():
        raise ValueError(
            "생기부 초안을 입력하세요."
        )

    if (
        record_section
        not in VALID_RECORD_SECTIONS
    ):
        raise ValueError(
            "올바른 생기부 항목을 선택하세요."
        )

    guideline_store = (
        load_guideline_vectorstore()
    )

    try:
        guideline_results = (
            retrieve_guideline_context(
                guideline_store,
                student_draft,
                record_section,
            )
        )

    except Exception as error:
        raise RuntimeError(
            "[작성요령 RAG 검색 실패] "
            f"{type(error).__name__}: {error}"
        ) from error

    return generate_review(
        student_draft=student_draft,
        record_section=record_section,
        guideline_results=guideline_results,
    )


__all__ = [
    "CollegeCriterion",
    "DraftDiagnosis",
    "Evidence",
    "ReviewResult",
    "SAMPLE_DRAFT",
    "extract_college_criteria",
    "generate_future_guide",
    "generate_review",
    "review_draft",
    "select_evidence",
    "validate_evidence",
]
