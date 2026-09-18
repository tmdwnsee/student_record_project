"""대학 맞춤 다음 학기 활동 가이드 화면."""
import hashlib
import streamlit as st
from config import COLLEGE_GUIDES
from rag.attachment import extract_context
from rag.chain import generate_future_guide
from rag.future import ACTIVITY_SECTIONS, next_semester

st.set_page_config(page_title="대학 맞춤 미래 가이드", page_icon="🎓", layout="wide")
st.markdown("""<style>
.stMarkdown, .stAlert {line-height:1.8}
[data-testid="stTable"] td {white-space:normal;vertical-align:top}
[data-testid="stTable"] th {white-space:nowrap}
</style>""", unsafe_allow_html=True)
st.title("🎓 대학 맞춤 미래 활동 가이드")
st.caption("자기평가보고서 초안과 기존 생기부를 바탕으로, 희망 대학의 평가 기준에 맞는 다음 학기 보완 활동을 설계합니다.")

with st.container(border=True):
    a, b = st.columns(2)
    with a:
        university = st.selectbox("희망 대학교", list(COLLEGE_GUIDES), key="guide_university")
    with b:
        department = st.text_input("희망 학과", placeholder="예: 미디어커뮤니케이션학과", key="guide_department")
    a, b = st.columns(2)
    with a:
        grade = st.selectbox("현재 학년", [1, 2, 3], format_func=lambda value: f"{value}학년", key="guide_grade")
    with b:
        semester = st.selectbox("현재 학기", [1, 2], format_func=lambda value: f"{value}학기", key="guide_semester")
    section = st.selectbox("초안의 활동 구분", ACTIVITY_SECTIONS, key="guide_section_type")
    subject = ""
    if section == "세부능력특기사항":
        subject = st.text_input("반영을 희망하는 과목", placeholder="예: 국어, 확률과 통계, 생명과학Ⅰ", key="guide_subject")
    try:
        target = next_semester(grade, semester)
        st.caption(f"설계 대상: {target} · {section}" + (f" · {subject}" if subject else ""))
    except ValueError as error:
        target = ""
        st.warning(str(error))
    draft = st.text_area("자기평가보고서 초안", height=180, key="guide_draft_input", placeholder="현재까지 수행한 활동, 과정, 역할과 배운 점을 입력하세요.")
    uploaded = st.file_uploader("기존 생기부 원문", type=["pdf", "txt"], key="guide_uploaded_record", help="PDF 또는 UTF-8 TXT, 최대 10MB")
    st.caption("등록된 모집요강 지원 대학: " + ", ".join(COLLEGE_GUIDES))

content = uploaded.getvalue() if uploaded else b""
context_key = (university, department.strip(), grade, semester, section, subject.strip(), draft,
               uploaded.name if uploaded else "", hashlib.sha256(content).hexdigest())
requested = st.button("대학 맞춤 활동 가이드 생성", type="primary", key="guide_button", disabled=not target)
if requested and st.session_state.get("guide_result") and st.session_state.get("guide_context_key") == context_key:
    st.caption("같은 입력으로 생성한 가이드를 불러왔습니다.")
elif requested:
    st.session_state.pop("guide_result", None)
    st.session_state.pop("guide_context_key", None)
    if not department.strip() or not draft.strip() or not uploaded or (section == "세부능력특기사항" and not subject.strip()):
        st.warning("희망 학과, 초안, 생기부 원문을 입력하고 세특인 경우 과목도 입력하세요.")
    else:
        try:
            with st.spinner("대학 평가 기준과 관련 생기부를 확인하고 다음 학기 활동을 설계하고 있습니다..."):
                result = generate_future_guide(draft, university, department.strip(), current_grade=grade,
                    current_semester=semester, section_type=section, previous_record=extract_context(uploaded.name, content), subject=subject)
            st.session_state["guide_result"] = result
            st.session_state["guide_context_key"] = context_key
        except Exception as error:
            st.error(f"가이드 생성 실패: {error}")

result = st.session_state.get("guide_result")
if result and st.session_state.get("guide_context_key") == context_key:
    st.divider()
    st.subheader("1. 활동 가이드")
    st.info(result["summary"])
    st.caption("반영 비율이 큰 평가항목부터 표시합니다. 반영 비율은 합격 확률이나 활동 시간 배분을 뜻하지 않습니다.")
    for item in result["future_activities"]:
        with st.container(border=True):
            st.markdown(f"### {item['criterion']} · {item['weight']}")
            st.markdown(f"**{item['title']}**")
            st.write(item["rationale"])
            a, b = st.columns(2)
            with a:
                st.markdown("**다음 학기 보완 목표**")
                st.write(item["goal"])
            with b:
                st.markdown("**희망 학과와 연결하는 방향**")
                st.write(item["department_connection"])
            st.table([{"시기": period, "구체적인 실행 방법": step["action"], "결과물·기록": step["output"]}
                      for period, step in zip(["학기 초 · 준비", "학기 중 · 실행", "학기 말 · 정리·성찰"], item["steps"])])
            st.markdown("**완료·성장 확인 기준**")
            st.write(item["success_check"])

    st.subheader("2. 근거")
    st.markdown("#### 대학 평가 항목과 반영 비율")
    for criterion in result["college_criteria"]:
        with st.container(border=True):
            st.markdown(f"**{criterion.area} · {criterion.weight}**")
            if criterion.subcriteria:
                st.write("평가요소: " + " · ".join(criterion.subcriteria))
            if criterion.evaluation_question:
                st.write(criterion.evaluation_question)
            for point in criterion.evaluation_points:
                st.write(f"• {point}")
    for evidence in result["college_evidence"]:
        page = f"PDF {evidence.page + 1}쪽" if evidence.page is not None else "페이지 미확인"
        with st.expander(f"모집요강 원문 · {evidence.source} · {page}"):
            st.write(evidence.content)
    st.markdown("#### 활동 연결에 참고한 기존 생기부")
    with st.expander("관련 생기부 구간 보기"):
        st.write(result["record_context"])
    st.caption("대학 기준은 등록된 모집요강에서 확인한 내용이며, 학과 연결과 미래 활동은 AI의 제안입니다. 적용 학년도와 전형은 원문을 확인하세요.")
