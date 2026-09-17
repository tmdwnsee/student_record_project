"""RAG 검색부터 검증까지 정해진 실행 순서만 관리합니다."""

import re

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

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


def generate_future_guide(
    student_draft: str,
    university: str = "성균관대학교",
    department: str = "",
    revised_text: str | None = None,
    section_type: str = "세부능력특기사항",
    previous_record: str = "",
    subject: str = "",
) -> dict:
    """검증된 수정안을 기반으로, 대학의 평가 항목과 반영 비율에 맞춘 미래 활동 가이드를 제시합니다."""
    source_text = (revised_text or student_draft).strip()
    if not source_text:
        raise ValueError("생기부 초안을 입력하세요.")

    section_label = section_type or "세부능력특기사항"
    subject_label = subject.strip() if section_label == "세부능력특기사항" else ""
    subject_context = f" {subject_label} 과목" if subject_label else ""
    record_context = ""
    if previous_record.strip():
        record_context = prepare_record_context(
            previous_record,
            f"{section_label}{subject_context}에서 {source_text}",
        )
    section_examples = {
        "세부능력특기사항": [
            "예시: 수업 시간에 배운 개념을 활용해 과제 자료를 비교·분석하고, 문제 해결 과정을 정리하며 발표를 준비하는 활동처럼 구성하면 좋습니다.",
            "예시: 과학 수업에서 실험 결과를 정리하며 원인과 근거를 비교하고, 그 과정과 의문점을 기록해 탐구 역량을 드러내는 활동으로 보완하면 좋습니다.",
            "예시: 교과 과제와 연계해 관련 자료를 수집하고 적용한 뒤, 결과를 성찰하며 개선점을 정리하는 과정을 기록하면 학업 역량을 더 분명하게 보여줄 수 있습니다."
        ],
        "창체·동아리 활동": [
            "예시: 동아리 활동에서 팀원 역할을 나누고 자료를 정리하며 회의 내용을 기록하는 과정을 보여주면 협업과 참여도를 자연스럽게 드러낼 수 있습니다.",
            "예시: 행사 준비 과정에서 발표 자료를 구성하고 역할을 분담하며 아이디어를 공유하는 활동을 기록하면 소통 능력과 책임감을 확인할 수 있습니다.",
            "예시: 동아리 활동을 마친 뒤 자신의 역할, 활동 과정, 문제 해결 과정을 정리해 기록하는 방식으로 활동의 의미를 구체적으로 나타낼 수 있습니다."
        ],
        "자율활동": [
            "예시: 자율활동 시간에 학습 내용을 정리하고, 이번 주 목표를 설정한 뒤 수행 과정을 점검하는 기록을 남기면 자기주도성이 드러납니다.",
            "예시: 자기계발 주제를 정해 관련 자료를 수집하고 정리한 뒤, 학습한 내용을 토대로 성장 과정을 기록하는 활동으로 구성하면 좋습니다.",
            "예시: 자율활동에서 목표를 세우고 실천 과정과 개선점을 성찰하며 기록하는 활동은 자기관리와 성실성을 보여주는 데 효과적입니다."
        ],
        "진로·봉사활동": [
            "예시: 진로 탐색 활동에서 관심 분야를 조사한 뒤 관련 경험을 찾고, 활동의 의미를 정리하는 과정을 기록하면 진로 역량이 드러납니다.",
            "예시: 봉사 활동에서 맡은 역할을 수행하고 협력 과정과 배운 점을 정리해 기록하면 공동체 의식과 책임감을 자연스럽게 보여줄 수 있습니다.",
            "예시: 진로나 봉사 활동을 통해 목표를 설정하고 활동에 참여한 과정, 느낀 점, 성장 내용을 연결해 기록하면 의미 있는 활동으로 보입니다."
        ],
    }

    college_store, _ = load_vectorstores()
    try:
        college_results = retrieve_college_context(
            college_store,
            f"{subject_label} {source_text}".strip(),
            university,
            department,
        )
    except Exception as error:
        raise RuntimeError(f"RAG 검색 실패 ({type(error).__name__})") from error

    college_criteria, _ = extract_college_criteria(
        college_results,
        f"{subject_label} {source_text}".strip(),
        university,
    )

    def build_revision_excerpt() -> str:
        if not college_criteria:
            return (
                "다음 학기 보완을 위해서는 ‘현재 자기평가보고서에서 드러난 활동을 바탕으로, 학업과 탐구 활동을 지속적으로 수행하며 과정과 결과를 정리하는 방식으로 보완하면 좋습니다.’와 같이 구성하면 좋습니다. "
                "‘수업에서 학습한 내용을 실제 문제에 적용하고, 관심 분야를 중심으로 자료를 조사하며 분석하는 과정을 이어가며 학업 역량과 탐구 역량을 더욱 분명하게 드러낼 수 있도록 정리하면 좋습니다.’"
            )

        parts: list[str] = []
        for criterion in college_criteria:
            missing = ", ".join(criterion.missing_aspects[:2]) if criterion.missing_aspects else "자기주도적 역할 수행과 협업 과정"
            direction = criterion.revision_direction or "실제 활동 과정과 결과"
            if "탐구" in criterion.area or "탐구" in direction or "탐구" in missing:
                if "범위" in missing or "심화" in missing or "확장" in missing:
                    parts.append(
                        "‘다음 학기에는 관심 분야를 더 넓은 관점에서 확장해 관련 자료를 비교하고 심화 조사하는 활동을 구성하면 좋습니다. "
                        "주제별로 자료를 정리하고 핵심 내용을 분석하며, 그 과정과 결과를 기록해 탐구 역량이 실제로 드러나는 활동으로 보완하면 좋습니다.’"
                    )
                elif "관심" in missing or "호기심" in missing or "주제" in missing:
                    parts.append(
                        "‘다음 학기에는 관심 있는 분야를 중심으로 주제를 선정하고 관련 자료를 조사하며 탐구를 지속하는 활동을 구성하면 좋습니다. "
                        "자료를 비교하고 분석하는 과정을 정리하고, 핵심 질문을 설정해 문제를 해결하는 과정을 기록하며 탐구 역량을 강화하는 방식으로 보완하면 좋습니다.’"
                    )
                else:
                    parts.append(
                        "‘다음 학기에는 관련 주제를 선정해 자료를 수집하고 비교·분석하는 탐구 활동을 구성하면 좋습니다. "
                        "이 과정에서 문제를 정의하고 해결 방법을 찾는 과정을 기록하며, 탐구 역량이 실제로 나타나는 활동으로 보완하면 좋습니다.’"
                    )
            elif "학업" in criterion.area:
                parts.append(
                    "‘다음 학기에는 수업 시간에 학습한 개념을 실제 문제 상황에 적용하는 과정을 구성하면 좋습니다. "
                    "개념을 정리하고 문제를 해결하는 활동을 이어가며, 학습 목표와 결과를 연결해 학업 역량이 구체적으로 드러나는 방식으로 보완하면 좋습니다.’"
                )
            elif "잠재" in criterion.area or "리더" in missing or "협업" in missing:
                parts.append(
                    "‘다음 학기에는 팀 프로젝트와 공동 활동에서 역할을 맡아 계획을 수립하고 협업을 수행하는 활동을 구성하면 좋습니다. "
                    "문제 해결 과정과 결과를 기록하며 자기주도성과 협업 능력을 함께 강화하고, 성장 과정을 체계적으로 정리하는 방식으로 보완하면 좋습니다.’"
                )
            else:
                parts.append(
                    f"‘다음 학기에는 {criterion.area}({criterion.weight}) 평가 항목에서 {direction}을 더욱 분명하게 드러낼 수 있도록, "
                    f"활동의 과정·역할·결과를 정리하며 본인의 성장과 의미를 구체적으로 표현하는 방식으로 보완하면 좋습니다.’"
                )

        if not parts:
            return source_text

        intro = (
            "다음 학기 보완을 위해서는 현재 자기평가보고서에서 드러난 활동을 바탕으로, 각 역량이 실제로 성장하는 과정을 정리하는 방식으로 구성하면 좋습니다. "
        )
        return intro + " ".join(parts[:2])

    revised_draft = build_revision_excerpt()

    def build_record_style_examples(criterion: object) -> list[str]:
        missing = ", ".join(criterion.missing_aspects[:2]) if criterion.missing_aspects else "자기주도적 역할 수행과 협업 과정"
        direction = criterion.revision_direction or "실제 활동 과정과 결과"
        missing_text = " ".join(criterion.missing_aspects[:2]).lower() if criterion.missing_aspects else ""

        if record_context:
            prompt = f"""
당신은 학교생활기록부 활동 가이드 작성자다.

학생의 생기부 원문에서 현재 자기평가보고서 초안과 같은 활동 구분의 내용을 참고해,
대학 평가 항목별 가이드 예시 3개를 작성하라.

[활동 구분]
{section_label}{subject_context}

[자기평가보고서 초안]
{source_text}

[생기부 원문에서 선별한 관련 내용]
{record_context}

[대학 평가 항목]
{criterion.area} ({criterion.weight})
평가 질문: {criterion.evaluation_question}
보완 요소: {missing}
강조 방향: {direction}

작성 규칙:
1. 반드시 생기부 원문에 실제로 확인되는 활동, 과목, 역할, 자료, 과정만 사용하라.
2. 원문에 없는 새로운 활동, 성과, 수치, 역할, 결과를 만들지 말라.
3. 각 문장은 "예시:"로 시작하고, 실제 학생의 활동 에피소드처럼
   "수업에서 무엇을 했고, 어떤 문제나 장면을 만났으며, 어떻게 정리·해결했는지"가 드러나게 하라.
4. "역량을 강화하기 위해", "활동이 적합합니다", "구성하면 좋습니다" 같은
   일반적인 필요성 설명으로 시작하지 말라.
5. 생기부 원문에 확인되는 과거 활동을 중심으로 작성하되, 마지막에는
   해당 평가 항목이 드러나도록 기록을 보완하는 가이드 문장으로 마무리하라.
6. 출력은 설명 없이 examples 필드에 문장 3개만 넣어라.
"""
            try:
                model = ChatOllama(
                    model=MODEL,
                    base_url=OLLAMA_BASE_URL,
                    temperature=0,
                    reasoning=False,
                ).with_structured_output(RecordGroundedExamples)
                generated = model.invoke(prompt)
                examples = [item.strip() for item in generated.examples if item.strip()]
                if len(examples) >= 3:
                    return examples[:3]
            except Exception:
                pass

            record_sentences = [
                part.strip()
                for part in re.split(r"(?<=[.!?다])\s+|\n+", record_context)
                if part.strip()
            ]
            if record_sentences:
                selected_sentences = (record_sentences * 3)[:3]
                return [
                    (
                        f"예시: 생기부 원문에 기록된 '{sentence}' 활동을 중심으로, "
                        f"{criterion.area} 평가 항목과 연결되는 실제 수행 과정과 판단 근거가 드러나도록 정리해 제시하면 좋습니다."
                    )
                    for sentence in selected_sentences
                ]

        if "탐구" in criterion.area or "탐구" in direction or "탐구" in missing_text:
            if "범위" in missing_text or "심화" in missing_text or "확장" in missing_text:
                return [
                    f"예시: {criterion.area} 역량을 강화하기 위해 과제와 연계해 관련 자료를 비교하고 심화 조사한 뒤, 핵심 내용을 정리하는 탐구 활동을 구성할 수 있습니다.",
                    f"예시: {criterion.area} 역량을 강화하기 위해 주제별 자료를 정리하고 분석하면서 문제 해결 과정을 기록하는 활동을 이어가면 탐구 과정이 더욱 드러납니다.",
                    f"예시: {criterion.area} 역량을 강화하기 위해 연구 주제와 관련된 자료를 수집하고 해석하는 과정을 기록하며 탐구 결과를 정리하는 활동이 적합합니다.",
                ]
            if "관심" in missing_text or "호기심" in missing_text or "주제" in missing_text:
                return [
                    f"예시: {criterion.area} 역량을 강화하기 위해 관심 있는 분야를 중심으로 주제를 정하고 관련 자료를 조사하며 분석하는 과정을 기록하는 활동이 적합합니다.",
                    f"예시: {criterion.area} 역량을 강화하기 위해 핵심 질문을 설정하고 자료를 비교·검토하며 탐구 흐름을 정리하는 과정을 보여주면 역량이 구체적으로 드러납니다.",
                    f"예시: {criterion.area} 역량을 강화하기 위해 흥미를 느낀 주제를 중심으로 탐구 활동을 수행하고, 과정과 의미를 정리하는 방식으로 기록하면 좋습니다.",
                ]
            return [
                f"예시: {criterion.area} 역량을 강화하기 위해 자료 수집과 비교·분석, 정리 발표를 반복하는 탐구 활동을 정기적으로 구성하는 방식이 효과적입니다.",
                f"예시: {criterion.area} 역량을 강화하기 위해 어떤 문제를 제기하고 어떤 근거로 해석했는지를 기록하며 탐구 과정을 구체화하는 활동이 적합합니다.",
                f"예시: {criterion.area} 역량을 강화하기 위해 탐구 결과를 바탕으로 배운 점과 적용 가능성을 정리하여 활동의 의미를 드러내는 방식이 좋습니다.",
            ]

        if "학업" in criterion.area:
            return [
                "예시: 수학 수업에서 함수 개념을 배우고 난 뒤 오답노트를 정리하며 같은 유형의 문제를 다시 풀어보는 과정을 기록하면 학업 역량이 더 선명하게 드러납니다. 개념을 이해한 뒤 적용해 보는 흐름이 자연스럽게 보이고, 학습 과정과 성장을 함께 보여줄 수 있습니다.",
                "예시: 과학 수업 시간에 실험 결과를 정리하면서 오류 원인을 분석하고, 그 내용을 다시 설명하는 과정을 남기면 학업 역량을 구체적으로 보여줄 수 있습니다. 단순히 결과만 적는 것이 아니라 배운 내용을 어떻게 적용했는지가 드러나면서 학업 성실성이 강조됩니다.",
                "예시: 국어 수업에서 발표 자료를 준비하며 핵심 내용을 정리하고 관련 자료를 비교해 보완하는 과정을 기록하면 학업 역량이 살아있는 형태로 나타납니다. 수업 내용을 자기 것으로 만드는 과정이 드러나기 때문에 대학에서 학업 태도와 이해도를 확인하기에 적합합니다.",
            ]

        if "잠재" in criterion.area or "리더" in missing_text or "협업" in missing_text:
            return [
                f"예시: {criterion.area} 역량을 강화하기 위해 팀 프로젝트에서 역할을 맡아 계획을 수립하고 소통하며 협업 과정을 기록하는 활동이 적합합니다.",
                f"예시: {criterion.area} 역량을 강화하기 위해 공동 과제에서 본인의 역할과 의사결정 과정을 정리하며 책임감과 협업 능력을 드러내는 방식이 좋습니다.",
                f"예시: {criterion.area} 역량을 강화하기 위해 협업 과정에서 나타난 성장과 문제 해결 방식을 기록하며 잠재역량을 보완하는 활동을 구성하면 좋습니다.",
            ]

        return [
            f"예시: {criterion.area} 역량을 강화하기 위해 {direction}을 중심으로 {missing}을 실제로 수행하는 활동을 구성하는 방식이 적합합니다.",
            f"예시: {criterion.area} 역량을 강화하기 위해 활동 과정과 역할, 결과를 정리하며 성장의 의미를 구체적으로 표현하는 활동을 구성하면 좋습니다.",
            f"예시: {criterion.area} 역량을 강화하기 위해 정기적인 활동 기록과 성찰 내용을 연결해 자신만의 역량 표현을 정리하는 방식으로 구성하면 좋습니다.",
        ]

    if not college_criteria:
        fallback_examples = section_examples.get(section_label, section_examples["세부능력특기사항"])
        return {
            "original_text": source_text,
            "university": university,
            "department": department,
            "section_type": section_label,
            "subject": subject_label,
            "record_context": record_context,
            "college_criteria": [],
            "future_activities": [{
                "criterion": "종합 평가",
                "weight": "-",
                "example_activity": fallback_examples[0],
                "example_activities": fallback_examples,
                "rationale": "모집요강에서 평가 항목을 확인할 수 없어, 현재 수정안에서 이미 강하게 드러난 활동을 중심으로 강화하는 것이 가장 안전하고 타당합니다.",
            }],
            "summary": "모집요강 기준을 확인할 수 없는 경우, 현재 수정안에서 이미 잘 드러난 활동을 중심으로 강화하는 방향이 가장 안전합니다.",
            "section_examples": fallback_examples,
        }

    recommendations: list[dict] = []
    for criterion in college_criteria:
        example_activities = build_record_style_examples(criterion)
        example_activity = " ".join(example_activities[:2])
        recommendations.append({
            "criterion": criterion.area,
            "weight": criterion.weight,
            "example_activity": example_activity,
            "example_activities": example_activities,
            "rationale": (
                f"{criterion.area} 항목은 {criterion.weight} 반영 비율을 가지므로, "
                f"{criterion.missing_aspects[0] if criterion.missing_aspects else '실제 활동과 성장 과정'}을 기록하면 대학이 요구하는 역량을 더 선명하게 확인할 수 있습니다."
            ),
        })

    return {
        "original_text": source_text,
        "university": university,
        "department": department,
        "section_type": section_label,
        "subject": subject_label,
        "record_context": record_context,
        "college_criteria": college_criteria,
        "future_activities": recommendations,
        "summary": (
            "이 가이드는 검증된 수정안을 기준으로, 대학별 평가 항목과 반영 비율을 근거로 다음 학기 보완 활동을 설계합니다. "
            "새로운 활동을 임의로 창작하기보다, 현재 수정안에 이미 반영된 경험을 대학이 중요하게 보는 역량과 연결해 구체화하는 방향으로 제안합니다."
        ),
        "section_examples": section_examples.get(section_label, section_examples["세부능력특기사항"]),
    }


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
