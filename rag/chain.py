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
            guideline_results,
            college_criteria,
            previous_record,
        )
    except Exception as error:
        raise RuntimeError(
            f"[초안 진단/수정안 생성 실패] "
            f"{type(error).__name__}: {error}"
        ) from error

    # 재정렬된 작성요령 5개를 Streamlit 근거 영역에 모두 표시합니다.
    guideline_evidence_ids = list(range(1, len(guideline_results) + 1))
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


def generate_future_guide(
    student_draft: str,
    university: str = "성균관대학교",
    department: str = "",
    revised_text: str | None = None,
) -> dict:
    """검증된 수정안을 기반으로, 대학의 평가 항목과 반영 비율에 맞춘 미래 활동 가이드를 제시합니다."""
    source_text = (revised_text or student_draft).strip()
    if not source_text:
        raise ValueError("생기부 초안을 입력하세요.")

    college_store, _ = load_vectorstores()
    try:
        college_results = retrieve_college_context(
            college_store,
            source_text,
            university,
            department,
        )
    except Exception as error:
        raise RuntimeError(f"RAG 검색 실패 ({type(error).__name__})") from error

    college_criteria, _ = extract_college_criteria(
        college_results,
        source_text,
        university,
    )

    if not college_criteria:
        return {
            "original_text": source_text,
            "university": university,
            "department": department,
            "college_criteria": [],
            "future_activities": [{
                "criterion": "종합 평가",
                "weight": "-",
                "example_activity": "다음 학기에는 현재 수정안에서 가장 잘 드러나는 활동을 중심으로, 역할·과정·결과를 더욱 구체적으로 기록하는 방향으로 보완 계획을 구성해보세요.",
                "rationale": "모집요강에서 평가 항목을 확인할 수 없어, 현재 수정안에서 이미 강하게 드러난 활동을 중심으로 강화하는 것이 가장 안전하고 타당합니다.",
            }],
            "summary": "모집요강 기준을 확인할 수 없는 경우, 현재 수정안에서 이미 잘 드러난 활동을 중심으로 강화하는 방향이 가장 안전합니다.",
        }

    recommendations: list[dict] = []
    for criterion in college_criteria:
        missing = ", ".join(criterion.missing_aspects[:2]) if criterion.missing_aspects else "자기주도적 역할 수행과 협업 과정"
        direction = criterion.revision_direction or "실제 활동 과정과 결과"
        example_activity = (
            f"다음 학기에는 {criterion.area}({criterion.weight}) 평가 항목을 강화하기 위해, "
            f"{direction}을 중심으로 {missing}을 실제로 수행하고, "
            f"주 1회 이상 정기적인 활동 기록, 역할 수행 과정, 그리고 결과를 문장으로 남기는 방식으로 보완하겠습니다."
        )
        recommendations.append({
            "criterion": criterion.area,
            "weight": criterion.weight,
            "example_activity": example_activity,
            "rationale": (
                f"{criterion.area} 항목은 {criterion.weight} 반영 비율을 가지므로, "
                f"{missing}을 실제 활동과 연결해 기록하면 대학이 요구하는 역량을 더 선명하게 확인할 수 있습니다."
            ),
        })

    return {
        "original_text": source_text,
        "university": university,
        "department": department,
        "college_criteria": college_criteria,
        "future_activities": recommendations,
        "summary": (
            "이 가이드는 검증된 수정안을 기준으로, 대학별 평가 항목과 반영 비율을 근거로 다음 학기 보완 활동을 설계합니다. "
            "새로운 활동을 임의로 창작하기보다, 현재 수정안에 이미 반영된 경험을 대학이 중요하게 보는 역량과 연결해 구체화하는 방향으로 제안합니다."
        ),
    }


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
    "generate_future_guide",
    "generate_review",
    "review_draft",
    "select_evidence",
    "validate_evidence",
]
