import hashlib
import streamlit as st

from config import COLLEGE_GUIDES
from rag.attachment import extract_context
from rag.chain import review_draft


st.set_page_config(
    page_title="자기평가보고서 초안 검토",
    page_icon="📝",
)

st.title(
    "자기평가보고서 초안 검토"
)

st.caption(
    "학교생활기록부 작성요령과 "
    "대학 모집요강을 기반으로 "
    "학생의 자기평가보고서 초안을 검토합니다."
)


university = st.selectbox("희망 대학교", options=list(COLLEGE_GUIDES))

department = st.text_input(
    "희망 학과",
    placeholder="예: 소프트웨어학과",
)

student_draft = st.text_area(
    "자기평가보고서 초안",
    height=180,
    placeholder=(
        "검토할 활동 기록을 "
        "입력하세요."
    ),
)

uploaded_record = st.file_uploader(
    "기존 생기부 첨부 (선택)",
    type=["pdf", "txt"],
    help="기존 활동을 이해하는 참고 자료로만 사용합니다. 텍스트 PDF 또는 UTF-8 TXT, 10MB 이하.",
)
upload_bytes = uploaded_record.getvalue() if uploaded_record else b""
context_key = (
    university,
    department,
    student_draft,
    hashlib.sha256(upload_bytes).hexdigest() if uploaded_record else "",
)


if st.button(
    "분석하기",
    type="primary",
):
    st.session_state.pop("review_result", None)
    st.session_state.pop("review_context_key", None)

    if not department.strip() or not student_draft.strip():

        st.warning(
            "희망 학과와 자기평가보고서 초안을 모두 입력하세요."
        )

    else:

        try:
            previous_record = (
                extract_context(uploaded_record.name, upload_bytes)
                if uploaded_record else ""
            )

            with st.spinner(
                "첨부 생기부 전체와 관련 근거를 살펴보고 분석 중입니다..."
            ):

                result = review_draft(
                    student_draft,
                    previous_record,
                    university,
                    department.strip(),
                )

            st.session_state[
                "review_result"
            ] = result
            st.session_state["review_context_key"] = context_key

        except Exception as error:

            st.error(
                f"분석 실패: {error}"
            )


result = st.session_state.get(
    "review_result"
)


if result and st.session_state.get("review_context_key") == context_key:

    st.divider()

    st.subheader("원문")

    st.write(
        result.original_text
    )


    st.subheader("수정안")

    st.write(
        result.revised_text
    )


    st.subheader("수정 이유")

    st.write(
        result.revision_reason
    )


    st.subheader(
        "학교생활기록부 작성요령 근거"
    )

    for evidence in (
        result.guideline_evidence
    ):

        st.caption(
            f"{evidence.source} "
            f"· page {evidence.page}"
        )

        st.write(
            evidence.content
        )


    st.subheader(
        "대학 모집요강 근거"
    )

    if result.college_criteria:
        st.table([
            {
                "평가영역": criterion.area,
                "반영비율/배점": criterion.weight,
                "초안 진단 및 수정 방향": criterion.recommendation,
            }
            for criterion in result.college_criteria
        ])
    else:
        st.write("모집요강에서 학생부 평가영역과 반영비율을 확인하지 못했습니다.")


    st.subheader(
        "주의사항"
    )

    st.write(
        result.caution
        or "추가 주의사항 없음"
    )

    if result.record_context:
        with st.expander("검토에 참고한 기존 생기부 구간"):
            st.write(result.record_context)
