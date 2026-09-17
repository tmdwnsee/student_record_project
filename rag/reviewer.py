"""LLM을 이용해 초안을 진단하고 생기부 기록 문장으로 수정합니다."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from rag.validation import Evidence, conservative_rewrite, looks_like_record_sentence


class ReviewResult(BaseModel):
    original_text: str
    revised_text: str
    revision_reason: str

    guideline_evidence: list[Evidence] = Field(
        default_factory=list
    )

    caution: str = ""

    record_section: str = ""


class DraftRewrite(BaseModel):
    revised_text: str = Field(
        description="학생 초안의 사실만 유지해 고친 하나의 생기부 기록 문단. 설명, 지시, 근거, 제목은 포함하지 않음"
    )


class DraftDiagnosis(BaseModel):
    preserved_facts: list[str] = Field(
        default_factory=list,
        description="학생 초안에 직접 존재하는 사실"
    )

    writing_issues: list[str] = Field(
        default_factory=list,
        description="작성요령을 기준으로 발견한 문제점"
    )

    missing_details: list[str] = Field(
        default_factory=list,
        description="수정에 도움이 되지만 현재 초안에서 확인할 수 없는 정보"
    )

    rewrite_directions: list[str] = Field(
        default_factory=list,
        description=(
            "학생 초안에 존재하는 사실만 사용하면서 "
            "학교생활기록부 작성요령을 반영해 "
            "어떻게 수정할지에 대한 구체적인 방향"
        )
    )
    deletion_targets: list[str] = Field(
        default_factory=list,
        description=(
            "작성요령에 따라 삭제하거나 "
            "최종 수정안에서 제외해야 하는 표현"
        ),
    )

class RewriteVerification(BaseModel):
    faithful: bool = Field(
        description="수정안이 학생 초안에 없는 새로운 사실을 추가하지 않았으면 true"
    )

    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="학생 초안에서 확인할 수 없는데 수정안에 새로 추가된 사실"
    )

    corrected_text: str = Field(
        default="",
        description=(
            "faithful이 false일 경우 새 사실을 제거하여 다시 작성한 생기부 문장. "
            "true이면 빈 문자열."
        ),
    )

REWRITE_PROMPT = """
학생이 작성한 초안을
교육부 학교생활기록부 작성요령에 맞는 문장으로 수정한다.

[작성 항목]
{record_section}

반드시 다음 원칙을 지킨다.

1. 학생 초안에 존재하는 사실만 사용한다.

2. 검색된 작성요령의 공통 규정과
   {record_section} 관련 규정을 적용한다.

3. 작성요령에 어긋나는 내용은 삭제할 수 있다.

4. 학생의 주관적 표현은 가능한 경우
   초안에 실제 존재하는 활동 중심 표현으로 수정한다.

5. 초안에 없는 활동, 역할, 성과, 수치,
   책 제목, 작품명 등을 추가하지 않는다.

6. missing_details에 있는 정보는 만들어 넣지 않는다.

7. 대학 모집요강, 대학 평가기준,
   희망 대학, 희망 학과는 사용하지 않는다.

8. 기존 생기부도 사용하지 않는다.

9. 최종 결과에는 설명을 넣지 않고
   실제 수정된 문장만 작성한다.


[학생 초안]

{student_draft}


[작성요령]

{guideline_context}


[진단 결과]

{diagnosis}
"""


DIAGNOSIS_PROMPT = """
당신은 교육부 학교생활기록부 기재요령을 기준으로
학생이 작성한 초안을 검토하는 전문가다.

[작성하려는 생기부 항목]
{record_section}

판단에는 오직 제공된 학교생활기록부 작성요령만 사용한다.

다음 원칙을 따른다.

1. 학생 초안에서 실제 확인되는 사실을 preserved_facts에 기록한다.

2. 제공된 작성요령과 비교하여
   부적절하거나 수정이 필요한 표현을 writing_issues에 기록한다.

3. 작성요령상 기재해서는 안 되거나
   선택 항목에 적합하지 않은 내용은
   deletion_targets에 기록한다.

4. 초안에 없는 활동, 역할, 성과, 수치,
   작품명, 책 제목 등을 만들어내지 않는다.

5. 추가 정보가 필요하지만 초안에서 확인할 수 없는 것은
   missing_details에 기록한다.

6. 실제 수정안에서 어떻게 고칠지
   rewrite_directions에 구체적으로 기록한다.

7. 대학 모집요강이나 대학 평가기준은 사용하지 않는다.

8. 기존 생기부 내용도 사용하지 않는다.


[학생 초안]

{student_draft}


[검색된 학교생활기록부 작성요령]

{guideline_context}
"""


REWRITE_VERIFY_PROMPT = """
다음 학생 초안과 수정안을 비교하라.

목적은 수정안이 학생 초안에 없는 사실을
새롭게 만들어냈는지 검사하는 것이다.

중요:

문장을 자연스럽게 바꾸거나 표현을 바꾼 것은
새로운 사실로 판단하지 않는다.

다음은 허용된다.

- 문장 순서 변경
- 학생부 문체로 변경
- 동의어 사용
- 주관적 표현을 객관적 활동 표현으로 정리
- 초안에 이미 있는 내용을 더 명확하게 표현

다음은 허용되지 않는다.

- 하지 않은 활동 추가
- 새로운 역할 추가
- 새로운 연구방법 추가
- 새로운 성과 추가
- 새로운 수치 추가
- 새로운 발표·수상·결과 추가
- 초안에 없는 구체적인 작품명이나 활동명 추가

수정안에 새로운 사실이 없다면 faithful=true로 한다.

새로운 사실이 있다면 faithful=false로 하고
unsupported_claims에 해당 내용을 기록한다.

faithful=false인 경우 corrected_text에는
문제가 되는 새로운 사실만 제거하고
학생 초안에 존재하는 사실만 사용해
학생부 문체로 수정문을 다시 작성한다.


[학생 초안]

{student_draft}


[LLM 수정안]

{revised_text}
"""


def format_guideline_context(
    documents: list[Document],
) -> str:

    return "\n\n".join(
        f"[{Path(document.metadata['source']).name} "
        f"· metadata page {document.metadata.get('page')}]\n"
        f"{document.page_content[:1_500]}"
        for document in documents[:5]
    ) or "검색된 작성요령 없음"


def diagnose_draft(
    model: ChatOllama,
    student_draft: str,
    record_section: str,
    guideline_results: list[Document],
) -> DraftDiagnosis:

    diagnosis_model = model.with_structured_output(
        DraftDiagnosis,
        method="json_schema",
    )

    diagnosis = diagnosis_model.invoke(
        DIAGNOSIS_PROMPT.format(
            student_draft=student_draft,
            record_section=record_section,
            guideline_context=format_guideline_context(
                guideline_results
            ),
        )
    )

    if not isinstance(
        diagnosis,
        DraftDiagnosis,
    ):
        raise ValueError(
            "구조화된 초안 진단을 생성하지 못했습니다."
        )

    return diagnosis


def format_diagnosis(
    diagnosis: DraftDiagnosis,
) -> str:

    preserved = "\n".join(
        f"- {item}"
        for item in diagnosis.preserved_facts
    ) or "- 없음"

    issues = "\n".join(
        f"- {item}"
        for item in diagnosis.writing_issues
    ) or "- 없음"

    missing = "\n".join(
        f"- {item}"
        for item in diagnosis.missing_details
    ) or "- 없음"

    directions = "\n".join(
        f"- {item}"
        for item in diagnosis.rewrite_directions
    ) or "- 별도의 수정 방향 없음"

    return f"""
[초안에서 확인된 사실]
{preserved}

[작성상 문제]
{issues}

[추가 확인이 필요한 정보]
{missing}

[수정 방향]
{directions}
""".strip()


def rewrite_student_draft(
    model: ChatOllama,
    student_draft: str,
    record_section: str,
    diagnosis: DraftDiagnosis,
    guideline_results: list[Document],
) -> str:
    """
    학교생활기록부 작성요령과 진단 결과를 이용해 수정안을 생성하고,
    별도 LLM 검증으로 원문에 없는 사실이 추가됐는지 확인한다.
    """

    # ---------------------------------
    # 1. 실제 수정안 생성
    # ---------------------------------

    rewrite_model = model.with_structured_output(
        DraftRewrite,
        method="json_schema",
    )

    rewritten = rewrite_model.invoke(
        REWRITE_PROMPT.format(
            student_draft=student_draft,
            record_section=record_section,
            guideline_context=format_guideline_context(
                guideline_results
            ),
            diagnosis=format_diagnosis(
                diagnosis
            ),
        )
    )

    if not isinstance(
        rewritten,
        DraftRewrite,
    ):
        raise ValueError(
            "구조화된 수정안을 생성하지 못했습니다."
        )

    revised_text = " ".join(
        rewritten.revised_text.split()
    ).strip()

    # ---------------------------------
    # 2. 학생부 문장 형식 검사
    # ---------------------------------

    # 여기서는 original을 넘기지 않는다.
    #
    # 기존 코드는:
    #
    # looks_like_record_sentence(
    #     revised_text,
    #     student_draft
    # )
    #
    # 였기 때문에 정상적인 의역까지
    # 새로운 내용으로 오판했다.
    if not looks_like_record_sentence(
        revised_text
    ):
        raise ValueError(
            "생성된 수정안이 학생부 문장 형식을 "
            "충족하지 못했습니다."
        )

    # ---------------------------------
    # 3. 새로운 사실 추가 여부 검사
    # ---------------------------------

    verifier = model.with_structured_output(
        RewriteVerification,
        method="json_schema",
    )

    verification = verifier.invoke(
        REWRITE_VERIFY_PROMPT.format(
            student_draft=student_draft,
            revised_text=revised_text,
        )
    )

    if not isinstance(
        verification,
        RewriteVerification,
    ):
        raise ValueError(
            "수정안 사실성 검증에 실패했습니다."
        )

    # ---------------------------------
    # 4. 문제 없으면 원래 수정안 사용
    # ---------------------------------

    if verification.faithful:
        return revised_text

    # ---------------------------------
    # 5. 새로운 사실이 있으면
    #    검증 모델이 고친 문장 사용
    # ---------------------------------

    corrected = " ".join(
        verification.corrected_text.split()
    ).strip()

    if (
        corrected
        and looks_like_record_sentence(
            corrected
        )
    ):
        return corrected

    raise ValueError(
        "수정안에 초안에서 확인되지 않은 "
        "새로운 사실이 포함되어 수정안을 확정하지 못했습니다."
    )


def build_revision_reason(
    original: str,
    revised: str,
    record_section: str,
    diagnosis: DraftDiagnosis,
) -> str:
    """작성요령 기준으로 수정 이유를 정리합니다."""

    lines = [
        f"- **선택 항목**: {record_section}",
        (
            "- **사실 보존**: "
            "초안에서 확인되지 않은 활동, 역할, 과정, "
            "성과는 추가하지 않았습니다."
        ),
    ]

    if revised != original:
        lines.append(
            "- **문장 정리**: "
            "초안의 사실을 유지하면서 선택한 "
            "생기부 항목에 맞는 문장으로 수정했습니다."
        )

    if diagnosis.writing_issues:
        lines.append(
            "- **수정한 표현**: "
            + "; ".join(
                diagnosis.writing_issues[:3]
            )
        )

    if diagnosis.deletion_targets:
        lines.append(
            "- **삭제·제외한 내용**: "
            + "; ".join(
                diagnosis.deletion_targets[:3]
            )
        )

    if diagnosis.rewrite_directions:
        lines.append(
            "- **작성요령 반영 방향**: "
            + "; ".join(
                diagnosis.rewrite_directions[:3]
            )
        )

    return "\n".join(lines)
