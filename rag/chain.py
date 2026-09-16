"""RAG 검색부터 검증까지 정해진 실행 순서만 관리합니다."""

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from config import MODEL, OLLAMA_BASE_URL
from rag.attachment import prepare_record_context
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
from storage.vectorstore import load_vectorstores

SAMPLE_DRAFT = "데이터 분석 프로젝트를 진행하며 Python을 활용해 데이터를 분석하고 문제 해결 능력을 향상하였다."

# 이전 import 경로를 사용하던 코드와 테스트의 호환성을 유지합니다.
_conservative_rewrite = conservative_rewrite
_looks_like_record_sentence = looks_like_record_sentence


def generate_review(
    student_draft: str,
    guideline_results: list[Document],
    college_results: list[Document],
    previous_record: str = "",
    university: str = "",
    department: str = "",
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
    college_criteria, criterion_evidence_ids = extract_college_criteria(
        college_results,
        student_draft,
        university,
    )

    try:
        diagnosis = diagnose_draft(
            model,
            student_draft,
            guideline_results,
            college_criteria,
            previous_record,
        )
        revised_text = rewrite_student_draft(
            model,
            student_draft,
            diagnosis,
            college_criteria,
            previous_record,
        )
    except Exception as error:
        raise RuntimeError(
            f"[초안 진단/수정안 생성] Ollama 응답 생성 실패 ({type(error).__name__}). "
            f"Ollama가 실행 중인지와 {MODEL} 모델이 설치되어 있는지 확인하세요."
        ) from error

    guideline_evidence_ids = list(range(1, min(3, len(guideline_results)) + 1))
    guideline_evidence = select_evidence(guideline_evidence_ids, guideline_results)
    college_evidence = select_evidence(criterion_evidence_ids, college_results)

    if diagnosis.missing_details:
        caution = "확인이 필요한 항목:\n- " + "\n- ".join(diagnosis.missing_details)
    else:
        caution = "초안에서 추가로 확인할 구체적 정보가 없습니다."
    caution += "\n\n초안에서 확인되지 않은 내용은 수정안에 추가하지 않았습니다."
    if not college_criteria:
        caution += " 모집요강 검색 결과에서는 평가영역과 반영비율을 확인하지 못했습니다."

    revision_reason = build_revision_reason(
        student_draft,
        revised_text,
        college_criteria,
        diagnosis,
    )
    references = list(dict.fromkeys(
        f"{item.source} (metadata page: {item.page})"
        for item in guideline_evidence + college_evidence
    ))
    if references:
        revision_reason += "\n\n사용한 근거: " + ", ".join(references)

    result = ReviewResult(
        original_text=student_draft,
        revised_text=revised_text,
        revision_reason=revision_reason,
        guideline_evidence=guideline_evidence,
        college_evidence=college_evidence,
        college_criteria=college_criteria,
        caution=caution,
        record_context=previous_record,
        university=university,
        department=department,
    )
    validate_evidence(result.guideline_evidence, guideline_results)
    validate_evidence(result.college_evidence, college_results)
    return result


def review_draft(
    student_draft: str,
    previous_record: str = "",
    university: str = "성균관대학교",
    department: str = "",
) -> ReviewResult:
    """Streamlit에서 호출하는 전체 검토 진입점입니다."""
    if not student_draft.strip():
        raise ValueError("생기부 초안을 입력하세요.")

    college_store, guideline_store = load_vectorstores()
    try:
        college_results = retrieve_college_context(
            college_store,
            student_draft,
            university,
            department,
        )
        guideline_results = retrieve_guideline_context(
            guideline_store,
            student_draft,
        )
    except Exception as error:
        raise RuntimeError(f"RAG 검색 실패 ({type(error).__name__})") from error

    if previous_record:
        previous_record = prepare_record_context(previous_record, student_draft)

    return generate_review(
        student_draft,
        guideline_results,
        college_results,
        previous_record,
        university,
        department,
    )


__all__ = [
    "CollegeCriterion",
    "DraftDiagnosis",
    "Evidence",
    "ReviewResult",
    "SAMPLE_DRAFT",
    "extract_college_criteria",
    "generate_review",
    "review_draft",
    "select_evidence",
    "validate_evidence",
]
