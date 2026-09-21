"""대학 맞춤 다음 학기 활동 가이드 화면."""
import hashlib
import html
import streamlit as st
from config import COLLEGE_GUIDES
from rag.attachment import extract_context
from rag.future import ACTIVITY_SECTIONS, generate_future_guide, next_semester

GUIDE_PIPELINE_VERSION = "grade-scope-and-record-context-v8"

st.set_page_config(page_title="대학 맞춤 미래 가이드", page_icon="🎓", layout="wide")
st.markdown("""<style>
.stMarkdown, .stAlert {line-height:1.8}
[data-testid="stTable"] td {white-space:normal;vertical-align:top}
[data-testid="stTable"] th {white-space:nowrap}
.record-item {padding:1.1rem 0 1.8rem;margin-bottom:.7rem;border-bottom:1px solid rgba(128,128,128,.22)}
.record-item:last-child {border-bottom:0;margin-bottom:0}
.record-heading {display:flex;align-items:center;gap:.7rem;margin-bottom:1.5rem;font-size:1.25rem;font-weight:700}
.record-number {display:inline-flex;align-items:center;justify-content:center;min-width:2rem;height:2rem;padding:0 .45rem;border-radius:999px;background:#e8eef8;color:#34527a;font-size:.82rem;font-weight:700}
.record-group {margin:0 0 1.45rem 2.7rem}
.record-group:last-child {margin-bottom:0}
.record-label {display:flex;align-items:center;gap:.55rem;margin-bottom:.55rem;font-size:.92rem;font-weight:700}
.record-dot {width:.5rem;height:.5rem;border-radius:999px;background:#5578a8;flex:none}
.record-dot.reason {background:#5f8a72}
.record-body {margin:0;line-height:1.9;font-size:1rem;word-break:keep-all;overflow-wrap:anywhere}
.record-note {margin:.5rem 0 0;color:#6b7280;font-size:.82rem;line-height:1.6}
</style>""", unsafe_allow_html=True)
st.title("🎓 대학 맞춤 미래 활동 가이드")
st.caption("자기평가보고서와 대학 평가 기준을 바탕으로 다음 학기 보완 활동을 설계합니다. 1학년 2학기부터는 기존 생기부 경험도 함께 반영합니다.")

with st.container(border=True):
    a, b = st.columns(2)
    with a:
        university = st.selectbox("희망 대학교", list(COLLEGE_GUIDES), key="guide_university")
    with b:
        department = st.text_input("희망 학과", placeholder="예: 미디어커뮤니케이션학과", key="guide_department")
    a, b = st.columns(2)
    with a:
        grade = st.selectbox("현재 학년", [1, 2], format_func=lambda value: f"{value}학년", key="guide_grade")
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
    draft = st.text_area("자기평가보고서", height=180, key="guide_draft_input", placeholder="현재까지 수행한 활동, 과정, 역할과 배운 점을 입력하세요.")
    uses_previous_record = not (grade == 1 and semester == 1)
    uploaded = None
    if uses_previous_record:
        uploaded = st.file_uploader("기존 생기부 원문", type=["pdf", "txt"], key="guide_uploaded_record", help="PDF 또는 UTF-8 TXT, 최대 10MB")
    else:
        st.caption("1학년 1학기는 기존 생기부 없이 자기평가보고서만 참고합니다.")
    st.caption("등록된 모집요강 지원 대학: " + ", ".join(COLLEGE_GUIDES))

content = uploaded.getvalue() if uploaded else b""
context_key = (GUIDE_PIPELINE_VERSION, university, department.strip(), grade, semester, section, subject.strip(), draft,
               uploaded.name if uploaded else "", hashlib.sha256(content).hexdigest())
requested = st.button("대학 맞춤 활동 가이드 생성", type="primary", key="guide_button", disabled=not target)
if requested and st.session_state.get("guide_result") and st.session_state.get("guide_context_key") == context_key:
    st.caption("같은 입력으로 생성한 가이드를 불러왔습니다.")
elif requested:
    st.session_state.pop("guide_result", None)
    st.session_state.pop("guide_context_key", None)
    missing_record = uses_previous_record and not uploaded
    if not department.strip() or not draft.strip() or missing_record or (section == "세부능력특기사항" and not subject.strip()):
        if uses_previous_record:
            st.warning("희망 학과, 자기평가보고서, 기존 생기부 원문을 입력하고 세특인 경우 과목도 입력하세요.")
        else:
            st.warning("희망 학과와 자기평가보고서를 입력하고 세특인 경우 과목도 입력하세요.")
    else:
        try:
            with st.spinner("대학 평가 기준과 입력 경험을 확인하고 다음 학기 활동을 설계하고 있습니다..."):
                previous_record = extract_context(uploaded.name, content) if uses_previous_record else ""
                result = generate_future_guide(draft, university, department.strip(), current_grade=grade,
                    current_semester=semester, section_type=section, previous_record=previous_record, subject=subject)
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
    if result.get("uses_previous_record"):
        st.markdown("3. 참고한 기존 생기부 경험")
        with st.expander("참고한 경험과 선정 이유 보기"):
            if result.get("record_matches"):
                st.caption("학생이 이미 수행한 주제·탐구 방법·역할 중 다음 학기 활동으로 이어 갈 수 있는 경험을 선별했습니다.")
                for index, match in enumerate(result["record_matches"], 1):
                    display_original = " ".join(match["original"].split())
                    experience_title = match.get("experience_title", "").strip()
                    if not experience_title:
                        experience_title = display_original[:40] + ("…" if len(display_original) > 40 else "")
                    repair_note = (
                        '<div class="record-note">PDF 표에서 본문 사이에 끼어든 과목명을 제거해 문장을 복원했습니다.</div>'
                        if match.get("text_was_repaired") else ""
                    )
                    st.markdown(
                        f"""
                        <section class="record-item">
                          <div class="record-heading">
                            <span class="record-number">{index:02d}</span>
                            <span>{html.escape(experience_title)}</span>
                          </div>
                          <div class="record-group">
                            <div class="record-label"><span class="record-dot"></span>기존 생기부 원문</div>
                            <p class="record-body">{html.escape(display_original)}</p>
                            {repair_note}
                          </div>
                          <div class="record-group">
                            <div class="record-label"><span class="record-dot reason"></span>이 경험을 관련 있다고 판단한 이유</div>
                            <p class="record-body">{html.escape(match["connection_reason"])}</p>
                          </div>
                        </section>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.write(result["record_context"])
    st.caption("대학 기준은 등록된 모집요강에서 확인한 내용이며, 학과 연결과 미래 활동은 AI의 제안입니다. 적용 학년도와 전형은 원문을 확인하세요.")
