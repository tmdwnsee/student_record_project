# app.py

import streamlit as st

from rag.chain import review_draft


st.set_page_config(
    page_title="생기부 초안 검토",
    page_icon="📝",
)

st.title(
    "생기부 초안 검토"
)

st.caption(
    "학교생활기록부 작성요령과 "
    "대학 모집요강을 기반으로 "
    "학생의 초안을 검토합니다."
)


student_draft = st.text_area(
    "생기부 초안",
    height=180,
    placeholder=(
        "검토할 활동 기록을 "
        "입력하세요."
    ),
)


if st.button(
    "분석하기",
    type="primary",
):

    if not student_draft.strip():

        st.warning(
            "생기부 초안을 입력하세요."
        )

    else:

        try:

            with st.spinner(
                "관련 근거를 검색하고 "
                "분석 중입니다..."
            ):

                result = review_draft(
                    student_draft
                )

            st.session_state[
                "review_result"
            ] = result

        except Exception as error:

            st.error(
                f"분석 실패: {error}"
            )


result = st.session_state.get(
    "review_result"
)


if result:

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

    for evidence in (
        result.college_evidence
    ):

        st.caption(
            f"{evidence.source} "
            f"· page {evidence.page}"
        )

        st.write(
            evidence.content
        )


    st.subheader(
        "주의사항"
    )

    st.write(
        result.caution
        or "추가 주의사항 없음"
    )