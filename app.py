import hashlib
import html
import re
import streamlit as st

from config import COLLEGE_GUIDES
from rag.attachment import extract_context
from rag.chain import generate_future_guide, review_draft


DISPLAY_MIN_GUIDELINE_SCORE = 0.28
CATEGORY_LABELS = {
    "특정 명칭 규정": "특정 기관·대학·상호명 규정",
    "시험·수상·자격 규정": "시험·수상·자격 기재 규정",
    "논문·지식재산 규정": "논문·지식재산 기재 규정",
    "개인·가족정보 규정": "개인·가족정보 기재 규정",
    "관찰·사실성": "관찰 가능한 사실 중심",
    "구체성·개별성": "구체적 활동·개별성",
}


def evidence_excerpt(content: str, keywords: list[str]) -> str:
    """검색 핵심어가 들어간 문단을 우선 표시하고 원문이 없으면 전체를 표시합니다."""
    if not keywords:
        return content
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]
    selected = [part for part in paragraphs if any(keyword in part for keyword in keywords)]
    return "\n\n".join(selected) if selected else content


def concise_evidence_text(content: str, keywords: list[str]) -> str:
    """글자 수로 자르지 않고 핵심어가 포함된 완결 문장만 선택합니다."""
    excerpt = evidence_excerpt(content, keywords)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", excerpt) if part.strip()]
    sentences = []
    for paragraph in paragraphs:
        normalized = re.sub(r"\s+", " ", paragraph)
        sentences.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip())
    relevant = [sentence for sentence in sentences if any(keyword in sentence for keyword in keywords)]
    return "\n\n".join((relevant or sentences)[:2])


def guideline_display_groups(evidence_items) -> list[tuple[str, list[dict]]]:
    """원문 한 문장 아래에 표시할 관련 규정 카드를 묶습니다."""
    grouped: dict[str, list[dict]] = {}
    seen = set()
    for evidence in evidence_items:
        score = evidence.retrieval_score or 0.0
        if score < DISPLAY_MIN_GUIDELINE_SCORE or not evidence.draft_matches:
            continue
        category_matches = evidence.draft_matches_by_category
        if not category_matches:
            continue
        for category, expressions in category_matches.items():
            if category not in CATEGORY_LABELS:
                continue
            category_keywords = evidence.retrieval_keyword_groups.get(category, [])
            for expression in expressions:
                summary = concise_evidence_text(evidence.content, category_keywords)
                key = (expression, category, evidence.source, evidence.page, summary)
                if key in seen:
                    continue
                seen.add(key)
                grouped.setdefault(expression, []).append({
                    "category": CATEGORY_LABELS[category],
                    "reason": category,
                    "source": evidence.source,
                    "page": evidence.page,
                    "excerpt": summary,
                    "score": score,
                })
    return list(grouped.items())


def render_guideline_card(items: list[dict]) -> None:
    categories = list(dict.fromkeys(item["category"] for item in items))
    sources = list(dict.fromkeys(
        f"{item['source']} (metadata page {item['page']})" for item in items
    ))
    summaries = list(dict.fromkeys(item["excerpt"] for item in items))
    badges = "".join(f'<span class="guideline-badge">{html.escape(category)}</span>' for category in categories)
    summary_html = "<br><br>".join(html.escape(summary) for summary in summaries[:2])
    st.markdown(
        f"""
        <div class="guideline-card">
          <div class="guideline-badges">{badges}</div>
          <div class="guideline-label">관련 기재요령 요약</div>
          <div class="guideline-quote">{summary_html}</div>
          <div class="guideline-source">출처: {html.escape(' / '.join(sources))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_college_criterion(criterion) -> None:
    """대학 평가기준을 질문·확인 항목·초안 보완 방향으로 나누어 표시합니다."""
    subcriteria = "".join(
        f'<span class="college-chip">{html.escape(item)}</span>'
        for item in criterion.subcriteria
    ) or '<span class="college-empty">확인된 하위 평가요소 없음</span>'
    question = html.escape(criterion.evaluation_question or "모집요강에서 별도 질문을 확인하지 못했습니다.")
    points = "".join(
        f"<li>{html.escape(item)}</li>" for item in criterion.evaluation_points
    ) or "<li>세부 확인 항목을 확인하지 못했습니다.</li>"
    evidence = "".join(
        f"<li>{html.escape(item)}</li>" for item in criterion.draft_evidence
    ) or "<li>현재 초안에서 직접 확인되는 관련 내용이 충분하지 않습니다.</li>"
    missing = "".join(
        f"<li>{html.escape(item)}</li>" for item in criterion.missing_aspects
    ) or "<li>현재 기준에서 별도로 표시할 부족 항목이 없습니다.</li>"
    direction = html.escape(criterion.revision_direction or "실제 활동 과정과 결과")
    st.markdown(
        f"""
        <div class="college-card">
          <div class="college-card-header">
            <span class="college-area">{html.escape(criterion.area)}</span>
            <span class="college-weight">{html.escape(criterion.weight)}</span>
          </div>
          <div class="college-section-label">하위 평가요소</div>
          <div class="college-chips">{subcriteria}</div>
          <div class="college-focus">
            <div class="college-section-label">대학의 핵심 평가 질문</div>
            <div class="college-question">{question}</div>
            <div class="college-section-label">중점 확인 항목</div>
            <ul>{points}</ul>
          </div>
          <div class="college-diagnosis-grid">
            <div class="college-diagnosis shown">
              <div class="college-section-label">초안에서 드러난 점</div>
              <ul>{evidence}</ul>
            </div>
            <div class="college-diagnosis missing">
              <div class="college-section-label">보완할 점</div>
              <ul>{missing}</ul>
            </div>
          </div>
          <div class="college-direction"><strong>강조할 방향</strong><br>{direction}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="자기평가보고서 초안 검토",
    page_icon="📝",
)

st.markdown("""
<style>
    .stMarkdown, .stAlert { line-height: 1.75; }
    [data-testid="stTable"] td { white-space: normal; vertical-align: top; }
    [data-testid="stTable"] th { white-space: nowrap; }
    .expression-title { color: #5f6b7a; font-weight: 700; margin: 1.4rem 0 .55rem; }
    .expression-box { background: #ffe2e2; color: #ef4444; font-size: 1.08rem;
        font-weight: 700; padding: 1rem 1.1rem; border-radius: .75rem; margin-bottom: .9rem; }
    .guideline-card { background: #f6f7f9; padding: 1.15rem; border-radius: .8rem; margin: .75rem 0 1rem; }
    .guideline-badges { display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: .75rem; }
    .guideline-badge { display: inline-block; color: white; background: #e8790c; font-weight: 700;
        padding: .4rem .62rem; border-radius: .45rem; font-size: .9rem; }
    .guideline-label { color: #526071; font-weight: 700; margin: .7rem 0 .3rem; }
    .guideline-text { color: #313843; line-height: 1.7; }
    .guideline-quote { background: #e8f0ff; border-left: 4px solid #3b82f6; padding: .85rem 1rem;
        border-radius: .45rem; color: #303846; line-height: 1.75; white-space: pre-wrap; }
    .guideline-source { color: #718096; font-size: .86rem; margin-top: .7rem; }
    .college-card { border: 1px solid #dfe5ec; border-radius: .9rem; padding: 1.2rem;
        margin: .8rem 0 1rem; background: white; }
    .college-card-header { display: flex; align-items: center; gap: .55rem; margin-bottom: .9rem; }
    .college-area { color: #1f2937; font-size: 1.15rem; font-weight: 800; }
    .college-weight { background: #e8f0ff; color: #2563eb; font-weight: 800;
        padding: .25rem .55rem; border-radius: 999px; }
    .college-section-label { color: #526071; font-weight: 750; margin: .55rem 0 .3rem; }
    .college-chips { display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: .8rem; }
    .college-chip { background: #e7f7f2; color: #087f5b; font-weight: 700;
        padding: .3rem .55rem; border-radius: .45rem; }
    .college-empty { color: #718096; }
    .college-focus { background: #f7f9fc; border-radius: .65rem; padding: .75rem 1rem; }
    .college-question { color: #263244; font-weight: 650; margin-bottom: .45rem; }
    .college-focus ul, .college-diagnosis ul { margin: .25rem 0 .1rem; padding-left: 1.25rem; }
    .college-diagnosis-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: .7rem; margin-top: .8rem; }
    .college-diagnosis { border-radius: .65rem; padding: .7rem .9rem; }
    .college-diagnosis.shown { background: #effaf5; }
    .college-diagnosis.missing { background: #fff7ed; }
    .college-direction { margin-top: .8rem; padding: .75rem .9rem; background: #eef4ff;
        border-left: 4px solid #3b82f6; border-radius: .45rem; line-height: 1.7; }
    @media (max-width: 720px) { .college-diagnosis-grid { grid-template-columns: 1fr; } }
</style>
""", unsafe_allow_html=True)

st.title(
    "📝 생기부 작성요령 검증 및 대학 맞춤 미래 활동 설계"
)

st.caption(
    "기능 1: 자기평가보고서 초안이 작성요령에 어긋나지 않는지 검증하고, 수정 방향과 근거를 제시합니다.\n"
    "기능 2: 희망 대학의 평가 항목과 반영 비율을 바탕으로, 다음 학기 보완 활동을 구체적으로 설계합니다."
)

review_tab, guide_tab = st.tabs([
    "1. 작성요령 검증",
    "2. 대학 맞춤 미래 가이드",
])

with review_tab:
    st.caption("자기평가보고서 초안을 입력하면, 작성요령과 실제 기록의 충돌 여부를 검토하고 정교한 수정안을 제시합니다.")
    review_university = st.selectbox("희망 대학교", options=list(COLLEGE_GUIDES), key="review_university")
    review_department = st.text_input("희망 학과", placeholder="예: 소프트웨어학과", key="review_department")
    review_text = st.text_area(
        "자기평가보고서 초안",
        height=180,
        placeholder="검토할 활동 내용을 입력하세요.",
        key="review_draft_input",
    )
    review_uploaded_record = st.file_uploader(
        "기존 생기부 첨부 (선택)",
        type=["pdf", "txt"],
        help="기존 활동을 이해하는 참고 자료로만 사용합니다. 텍스트 PDF 또는 UTF-8 TXT, 10MB 이하.",
        key="review_uploaded_record",
    )
    review_upload_bytes = review_uploaded_record.getvalue() if review_uploaded_record else b""
    review_context_key = (
        review_university,
        review_department,
        review_text,
        hashlib.sha256(review_upload_bytes).hexdigest() if review_uploaded_record else "",
    )

    if st.button("작성요령 검증 및 수정안 생성", key="review_button", type="primary"):
        st.session_state.pop("review_result", None)
        st.session_state.pop("review_context_key", None)

        if not review_department.strip() or not review_text.strip():
            st.warning("희망 학과와 자기평가보고서 초안을 모두 입력하세요.")
        else:
            try:
                previous_record = (
                    extract_context(review_uploaded_record.name, review_upload_bytes)
                    if review_uploaded_record else ""
                )
                with st.spinner("작성요령과 대학 기준을 기준으로 초안을 검증하는 중입니다..."):
                    result = review_draft(
                        review_text,
                        previous_record,
                        review_university,
                        review_department.strip(),
                    )
                st.session_state["review_result"] = result
                st.session_state["review_context_key"] = review_context_key
            except Exception as error:
                st.error(f"검증 실패: {error}")

    review_result = st.session_state.get("review_result")
    if review_result and st.session_state.get("review_context_key") == review_context_key:
        st.divider()
        st.subheader("1. 입력 원문")
        with st.container(border=True):
            st.markdown(review_result.original_text)

        st.subheader("2. 수정안")
        st.caption("입력한 사실만 사용해 생기부 기록 문장으로 다듬은 결과입니다.")
        with st.container(border=True):
            st.markdown(f"**{review_result.revised_text}**")

        st.subheader("3. 수정 이유")
        with st.container(border=True):
            st.markdown(review_result.revision_reason)

        st.subheader("4. 학교생활기록부 작성요령 근거")
        st.caption("문제가 발견된 원문을 한 문장씩 나누고, 관련성이 높은 작성요령만 연결했습니다.")
        guideline_groups = guideline_display_groups(review_result.guideline_evidence)
        if guideline_groups:
            for expression, cards in guideline_groups:
                st.markdown('<div class="expression-title">발견된 표현</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="expression-box">“{html.escape(expression)}”</div>', unsafe_allow_html=True)
                render_guideline_card(cards)
        else:
            st.info("표시 기준을 충족하는 작성요령 근거가 없습니다.")

        st.subheader("5. 대학 모집요강 근거")
        if review_result.college_criteria:
            st.caption("평가영역별로 대학이 확인하는 내용과 현재 초안의 보완 방향을 나누어 표시합니다.")
            for criterion in review_result.college_criteria:
                render_college_criterion(criterion)
        else:
            st.write("모집요강에서 학생부 평가영역과 반영비율을 확인하지 못했습니다.")

        st.subheader("6. 추가 확인사항")
        if review_result.caution:
            st.warning(review_result.caution)
        else:
            st.success("추가로 확인할 사항이 없습니다.")

        if review_result.record_context:
            with st.expander("검토에 참고한 기존 생기부 구간"):
                st.write(review_result.record_context)

with guide_tab:
    st.caption("검증된 수정안을 기준으로, 모집요강에서 강조하는 평가 항목과 반영 비율을 근거로 다음 학기 보완 활동을 설계합니다.")
    guide_university = st.selectbox("희망 대학교", options=list(COLLEGE_GUIDES), key="guide_university")
    guide_department = st.text_input("희망 학과", placeholder="예: 소프트웨어학과", key="guide_department")
    guide_default_text = ""
    review_result = st.session_state.get("review_result")
    if review_result:
        guide_default_text = review_result.revised_text
    guide_draft = st.text_area(
        "검증 결과 반영 초안",
        height=180,
        value=guide_default_text,
        placeholder="검증에서 나온 수정안을 붙여넣거나, 대학 맞춤 가이드를 적용할 초안을 입력하세요.",
        key="guide_draft_input",
    )

    if st.button("수정안 기반 미래 가이드 생성", key="guide_button", type="secondary"):
        if not guide_department.strip() or not guide_draft.strip():
            st.warning("희망 학과와 수정안 반영 초안을 모두 입력하세요.")
        else:
            try:
                with st.spinner("검증된 수정안을 바탕으로 대학별 평가 항목과 반영 비율에 맞는 보완 계획을 구성하고 있습니다..."):
                    guide_result = generate_future_guide(
                        guide_draft,
                        guide_university,
                        guide_department.strip(),
                        revised_text=guide_draft,
                    )

                st.subheader("1. 기준이 되는 수정안")
                st.write(guide_result["original_text"])

                st.subheader("2. 다음 학기 보완 활동 예시")
                st.info(guide_result["summary"])
                for item in guide_result["future_activities"]:
                    st.markdown(f"### {item['criterion']} ({item['weight']})")
                    st.write(item["rationale"])
                    st.code(item["example_activity"], language="text")

                st.subheader("3. 대학 평가 항목 근거")
                if guide_result["college_criteria"]:
                    for criterion in guide_result["college_criteria"]:
                        render_college_criterion(criterion)
                else:
                    st.write("모집요강에서 평가 영역과 반영 비율을 확인하지 못했습니다.")
            except Exception as error:
                st.error(f"가이드 생성 실패: {error}")
