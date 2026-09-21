"""대학 모집요강 원문에서 평가영역과 반영비율을 추출합니다."""

import re
import unicodedata
from pathlib import Path

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class CollegeCriterion(BaseModel):
    area: str
    weight: str
    subcriteria: list[str] = Field(default_factory=list)
    evaluation_question: str = ""
    evaluation_points: list[str] = Field(default_factory=list)
    draft_evidence: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    revision_direction: str = ""
    recommendation: str
    source: str
    page: int | None = None
    evidence_pages: list[int] = Field(default_factory=list)
    printed_pages: list[int] = Field(default_factory=list)


def _compact(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"퍼센트|percent", "%", normalized)
    return re.sub(r"\s+", "", normalized)


def _explicit_weighted_areas(source: str) -> list[tuple[str, str, int]]:
    """괄호 유무와 관계없이 평가항목 옆에 명시된 비율을 찾습니다."""
    area = (
        r"[가-힣A-Za-z· ]{1,30}?"
        r"(?:역량|적합성|가능성|사회성|공동체의식|성실성|주도성)"
    )
    patterns = (
        re.compile(rf"(?:^|\s)({area})\s*\(\s*(\d+(?:\.\d+)?)\s*(?:%|퍼센트)\s*\)"),
        re.compile(rf"^\s*({area})\s+(\d+(?:\.\d+)?)\s*(?:%|퍼센트)"),
    )
    results = []
    offset = 0
    for line in unicodedata.normalize("NFKC", source).splitlines():
        for pattern in patterns:
            for match in pattern.finditer(line):
                name = " ".join(match.group(1).split())
                name = re.sub(r"^.*(?:서류평가|종합평가)\s+", "", name)
                repeated_areas = re.findall(r"\S+(?:역량|적합성|가능성|사회성|공동체의식|성실성|주도성)", name)
                if len(repeated_areas) > 1:
                    name = repeated_areas[0]
                if name not in {"계", "서류평가", "면접평가"}:
                    results.append((name, f"{match.group(2)}%", offset + match.start(1)))
        offset += len(_compact(line))
    return list(dict.fromkeys(results))


def _extract_subcriteria(block: str) -> list[str]:
    return list(dict.fromkeys(
        match.group(1).strip()
        for match in re.finditer(r"([가-힣A-Za-z·]{2,20})\s*\(\s*\d+(?:\.\d+)?\s*점\s*\)", block)
    ))


def _extract_points(block: str) -> list[str]:
    return list(dict.fromkeys(
        " ".join(point.split()).strip(" ,-·")
        for point in re.findall(r"-\s*([^\n]+)", block)
        if point.strip(" ,-·")
    ))


def _table_explanations(source: str) -> list[tuple[str, list[str]]]:
    """PDF 표의 세 번째 열을 질문과 하위 확인 항목 묶음으로 분리합니다."""
    question_matches = list(re.finditer(
        r"[^\n]*?(?:보여주는가|있는가)", source,
    ))
    explanations = []
    for index, match in enumerate(question_matches):
        question = " ".join(match.group(0).split()).strip()
        question = re.sub(r"^.*?\)\s*", "", question)
        start = match.end()
        end = question_matches[index + 1].start() if index + 1 < len(question_matches) else len(source)
        next_section = re.search(r"\n\s*\d+\.\s", source[start:end])
        if next_section:
            end = start + next_section.start()
        points = []
        for point in _extract_points(source[start:end]):
            point = re.sub(r"[가-힣A-Za-z·]{2,20}\s*\(\s*\d+(?:\.\d+)?\s*점\s*\).*$", "", point).strip()
            if point:
                points.append(point)
        explanations.append((question, points))
    return explanations


def _object_particle(word: str) -> str:
    if not word:
        return "을"
    code = ord(word[-1]) - ord("가")
    return "을" if 0 <= code <= 11171 and code % 28 else "를"


AREA_SUBCRITERIA = {
    "학업역량": (
        "학업수월성", "학업충실성", "기초학업역량", "학습의 주도성",
        "학업성취도", "학업태도 및 탐구력",
    ),
    "탐구역량": ("탐구확장성", "탐구주도성"),
    "잠재역량": ("미래성장성", "공동체의식"),
    "전공적합성": ("전공수학역량", "전공관심도 및 진로탐색노력"),
    "인성 및 사회성": ("역할의 주도성", "협업소통능력"),
    "진로역량": ("전공(계열) 관련 교과 이수 노력 및 성취도", "진로 탐색 활동과 경험"),
    "자기주도역량": ("자기주도 교과 이수 노력 및 성취도", "자기주도 진로 탐색 활동과 경험"),
    "공동체역량": ("협업과 소통능력, 리더십", "나눔과 배려, 성실성과 규칙준수"),
    "성장가능성": ("학업에 대한 열의", "다양한 경험을 위한 노력"),
}

HIERARCHICAL_SUBCRITERIA = {
    "학업역량": ("학업성취도", "학업태도 및 탐구력"),
    "진로역량": ("전공(계열) 관련 교과 이수 노력 및 성취도", "진로 탐색 활동과 경험"),
    "자기주도역량": ("자기주도 교과 이수 노력 및 성취도", "자기주도 진로 탐색 활동과 경험"),
    "공동체역량": ("협업과 소통능력, 리더십", "나눔과 배려, 성실성과 규칙준수"),
}


def _flexible_term_pattern(term: str) -> str:
    """PDF 줄바꿈과 공백 차이를 허용하는 문구 패턴입니다."""
    return r"\s*".join(re.escape(part) for part in term.split())


def _hierarchical_weighted_candidate(
    pages: list[dict], department: str,
) -> tuple[dict, str, list[tuple[str, str]], dict[str, int | None]] | None:
    """상위 역량 비율이 하위 평가항목 비율의 합으로 제시된 표를 처리합니다."""
    second_area = (
        "자기주도역량"
        if re.search(r"자율\s*전공|자유\s*전공", department)
        else "진로역량"
    )
    selected_areas = ("학업역량", second_area, "공동체역량")
    weighted_areas: list[tuple[str, str]] = []
    area_pages: dict[str, int | None] = {}
    relevant_pages: list[dict] = []

    for area in selected_areas:
        subcriteria = HIERARCHICAL_SUBCRITERIA[area]
        weights: list[float] = []
        matched_page = None
        for page in pages:
            text = unicodedata.normalize("NFKC", page["text"])
            page_weights = []
            for subcriterion in subcriteria:
                match = re.search(
                    rf"{_flexible_term_pattern(subcriterion)}\s*\(\s*(\d+(?:\.\d+)?)\s*%\s*\)",
                    text,
                )
                if match:
                    page_weights.append(float(match.group(1)))
            if page_weights:
                weights.extend(page_weights)
                relevant_pages.append(page)
                matched_page = page["page"] if matched_page is None else matched_page
        if len(weights) != len(subcriteria):
            return None
        total = sum(weights)
        weighted_areas.append((area, f"{total:g}%"))
        area_pages[area] = matched_page

    if not 99 <= sum(float(weight.rstrip("%")) for _, weight in weighted_areas) <= 101:
        return None
    first_page = min(
        relevant_pages,
        key=lambda page: page["page"] if page["page"] is not None else 10**9,
    )
    combined = "\n".join(
        page["text"] for page in sorted(
            {id(page): page for page in relevant_pages}.values(),
            key=lambda page: page["page"] if page["page"] is not None else 10**9,
        )
    )
    return first_page, combined, weighted_areas, area_pages


def _student_comprehensive_section(text: str) -> str:
    """한 페이지에 교과전형 표가 함께 있으면 학생부종합 표만 남깁니다."""
    marker = re.search(r"■\s*학생부종합(?:\s|$)", text)
    if not marker:
        return text
    end_marker = re.search(r"■\s*학생부교과", text[marker.end():])
    end = marker.end() + end_marker.start() if end_marker else len(text)
    return text[marker.start():end]


def _group_pages(documents: list[Document]) -> list[dict]:
    groups: dict[tuple[str, int | None], list[tuple[int, Document]]] = {}
    for index, document in enumerate(documents, start=1):
        key = (str(document.metadata.get("source", "")), document.metadata.get("page"))
        groups.setdefault(key, []).append((index, document))
    pages = []
    for (source, page), items in groups.items():
        items.sort(key=lambda item: (item[1].metadata.get("start_index", 10**12), item[0]))
        indexed_items = [
            (item[1].metadata.get("start_index"), item[1].page_content)
            for item in items
        ]
        if indexed_items and all(isinstance(start, int) for start, _ in indexed_items):
            text_length = max(start + len(content) for start, content in indexed_items)
            characters = [" "] * text_length
            for start, content in indexed_items:
                for offset, character in enumerate(content):
                    characters[start + offset] = character
            page_text = "".join(characters)
        else:
            page_text = "\n".join(document.page_content for _, document in items)
        pages.append({
            "source": source,
            "page": page,
            "printed_page": items[0][1].metadata.get("printed_page"),
            "ids": [index for index, _ in items],
            "text": page_text,
        })
    return pages


def _page_score(text: str, areas: list[tuple[str, str, int]]) -> int:
    score = len(areas) * 3
    score += 5 if re.search(r"서류종합평가|학생부종합전형\s*서류평가", text) else 0
    score += 3 if re.search(r"평가기준|평가\s*영역\s*및\s*반영", text) else 0
    score += 2 if "세부평가항목" in text else 0
    score += 8 if re.search(r"Do\s*Dream|불교추천인재|서류종합평가", text) else 0
    score += 15 if re.search(r"■\s*학생부종합(?:\s|$)", text) else 0
    score -= 8 if "DONGGUK UNIVERSITY WISE" in text else 0
    score -= 7 if "면접평가" in text[:200] else 0
    return score


def _subcriteria_for_area(area: str, text: str, scored: list[str], position: int) -> list[str]:
    if len(scored) >= (position + 1) * 2:
        return scored[position * 2:position * 2 + 2]
    return [
        term for term in AREA_SUBCRITERIA.get(area, ())
        if re.search(_flexible_term_pattern(term), text)
    ]


def _hierarchical_evaluation_points(
    area: str, pages: list[dict], criterion_page: int | None,
) -> tuple[list[str], list[int], int | None]:
    """상위 역량별 표 영역을 자른 뒤 줄바꿈된 공식 질문을 물음표까지 복원합니다."""
    page = next(
        (candidate for candidate in pages if candidate["page"] == criterion_page),
        None,
    )
    if not page:
        return [], [], None
    text = unicodedata.normalize("NFKC", page["text"])
    start_match = re.search(_flexible_term_pattern(area), text)
    if not start_match:
        return [], [], page["page"]
    stop_terms = {
        "학업역량": ("진로역량", "자기주도역량", "공동체역량"),
        "진로역량": ("자기주도역량", "공동체역량"),
        "자기주도역량": ("공동체역량",),
        "공동체역량": ("학생부종합전형 면접평가",),
    }.get(area, ())
    end_positions = []
    for term in stop_terms:
        match = re.search(_flexible_term_pattern(term), text[start_match.end():])
        if match:
            end_positions.append(start_match.end() + match.start())
    end = min(end_positions) if end_positions else len(text)
    block = text[start_match.end():end]
    points = []
    bullets = list(re.finditer(r"(?:^|\n)\s*-\s*", block))
    for index, bullet in enumerate(bullets):
        next_bullet = bullets[index + 1].start() if index + 1 < len(bullets) else len(block)
        segment = block[bullet.end():next_bullet]
        question_end = segment.rfind("?")
        if question_end < 0:
            continue
        point = segment[:question_end + 1]
        trailing = segment[question_end + 1:].lstrip()
        if trailing.startswith("(") and ")" in trailing:
            point += trailing[:trailing.find(")") + 1]
        point = re.sub(r"[\x00-\x1f\x7f]", " ", point)
        point = " ".join(point.split()).strip(" ,-·")
        if point:
            points.append(point)
    return list(dict.fromkeys(points)), page["ids"], page["page"]


def _area_evaluation_points(
    area: str,
    area_names: list[str],
    pages: list[dict],
    criterion_page: int | None,
) -> tuple[list[str], list[int], int | None]:
    area_point_hints = {
        "학업역량": r"기초수학|학업",
        "탐구역량": r"탐구|호기심|배움",
        "잠재역량": r"자기주도|리더십|소통|성실",
        "전공적합성": r"전공|진로",
        "인성 및 사회성": r"역할|책임|공동체|협력|소통",
        "진로역량": r"전공|진로|관심 분야",
        "자기주도역량": r"자기주도|관심 분야",
        "공동체역량": r"협업|소통|리더십|나눔|배려|성실|규칙",
    }
    best_points: list[str] = []
    best_ids: list[int] = []
    best_page: int | None = None
    best_score = -10**9
    nearby_pages = [
        candidate for candidate in pages
        if criterion_page is None or candidate["page"] is None or abs(candidate["page"] - criterion_page) <= 1
    ]
    preferred_pages = [
        candidate for candidate in nearby_pages
        if "서류종합평가" in candidate["text"] and "평가항목" in candidate["text"]
    ]
    for page in preferred_pages or nearby_pages:
        text = unicodedata.normalize("NFKC", page["text"])
        expected_subcriteria = AREA_SUBCRITERIA.get(area, ())
        all_subcriteria = tuple(dict.fromkeys(
            term for terms in AREA_SUBCRITERIA.values() for term in terms
        ))
        subcriterion_positions = []
        for term in all_subcriteria:
            matches = list(re.finditer(r"\s*".join(map(re.escape, term.split())), text))
            if not matches and " " in term:
                matches = list(re.finditer(re.escape(term.split()[0]), text))
            subcriterion_positions.extend((match.start(), term) for match in matches)
        points = []
        if subcriterion_positions and expected_subcriteria:
            bullets = list(re.finditer(r"[■•]\s*", text))
            if not bullets:
                bullets = list(re.finditer(r"(?:^|\n)\s*-\s*", text))
            for bullet_index, bullet in enumerate(bullets):
                next_bullet = bullets[bullet_index + 1].start() if bullet_index + 1 < len(bullets) else len(text)
                preceding_terms = [
                    (term_position, term)
                    for term_position, term in subcriterion_positions
                    if term_position < bullet.start()
                ]
                following_terms = [
                    (term_position, term)
                    for term_position, term in subcriterion_positions
                    if bullet.end() <= term_position < next_bullet
                ]
                nearest_preceding = (
                    max(preceding_terms, key=lambda item: item[0])
                    if preceding_terms else None
                )
                if nearest_preceding and nearest_preceding[1] in expected_subcriteria:
                    matched_term = nearest_preceding[1]
                elif following_terms:
                    matched_term = min(following_terms, key=lambda item: item[0])[1]
                else:
                    matched_term = min(
                        subcriterion_positions,
                        key=lambda item: abs(item[0] - bullet.start()),
                    )[1]
                point_end = min(
                    [next_bullet] + [position for position, _ in following_terms]
                )
                point_text = " ".join(text[bullet.end():point_end].split()).strip(" ,-·")
                belongs_to_another_area = (
                    area == "학업역량"
                    and bool(re.search(r"전공|진로|공동체|협력|소통|책임|역할", point_text))
                )
                if (
                    point_text
                    and
                    not belongs_to_another_area
                    and (
                        matched_term in expected_subcriteria
                        or re.search(area_point_hints.get(area, r"$^"), point_text)
                    )
                ):
                    points.append(point_text)
            blocks = []
        else:
            blocks = []
            for match in re.finditer(re.escape(area), text):
                following = text[match.end():]
                boundaries = [following.find(other) for other in area_names if other != area and following.find(other) >= 0]
                blocks.append(following[:min(boundaries)] if boundaries else following[:1_500])

        for block in blocks:
            point_matches = re.findall(r"[■•]\s*([^\n]+)", block)
            if not point_matches:
                point_matches = re.findall(r"(?:^|\n)\s*-\s*([^\n]+)", block)
            for point in point_matches:
                cleaned = " ".join(point.split()).strip(" ,-·")
                cleaned = re.sub(r"\s*(?:제출서류|전형별|세부사항).*$", "", cleaned).strip()
                if cleaned and not re.search(r"^(?:학생부종합|학생부교과|평가절차)$", cleaned):
                    points.append(cleaned)
        cleaned_points = []
        for point in points:
            cleaned = " ".join(point.split()).strip(" ,-·")
            cleaned = re.sub(r"\s*(?:제출서류|전형별|세부사항).*$", "", cleaned).strip()
            if cleaned and not re.search(r"학생부종합|학생부교과|평가절차", cleaned):
                cleaned_points.append(cleaned)
        points = list(dict.fromkeys(cleaned_points))[:12]
        score = len(points)
        score += 4 * sum(term in text for term in expected_subcriteria)
        score += 10 if "서류종합평가" in text else 0
        score -= 8 if "합격자 사정원칙" in text else 0
        score -= 5 if "■ 학생부교과" in text and "서류종합평가" not in text else 0
        if score > best_score:
            best_score = score
            best_points = points
            best_ids = page["ids"]
            best_page = page["page"]
    return best_points, best_ids, best_page


def _draft_assessment(area: str, student_draft: str) -> tuple[list[str], list[str], str]:
    """초안에 실제로 보이는 신호와 추가로 구체화할 항목을 구분합니다."""
    indicators = {
        "학업역량": {
            "학업 활동과 성취 과정": r"학습|수업|과목|성취|피드백|수정|이해|읽|작성",
            "분석·비교 과정": r"분석|비교|정리|해석|판단",
        },
        "탐구역량": {
            "관심 분야의 탐구 과정": r"탐구|조사|자료|실험|보고서|호기심|궁금",
            "탐구 범위의 확장": r"확장|심화|추가|다른\s|관점|범위",
        },
        "잠재역량": {
            "자기주도적 참여": r"주도|스스로|계획|끝까지|참여",
            "협업·소통 경험": r"협력|협업|소통|토론|모둠|친구|도와|리더",
        },
        "전공적합성": {
            "전공 관련 학업·탐구 경험": r"전공|진로|탐구|조사|분석|보고서|자료",
            "진로 탐색과 관심의 구체화": r"관심|계기|진로|학과|확장|심화",
        },
        "인성 및 사회성": {
            "주도적인 역할 수행": r"주도|역할|책임|계획|이끌|끝까지",
            "협업·소통 경험": r"협력|협업|소통|토론|모둠|친구|도와|공동체",
        },
        "진로역량": {
            "전공 관련 교과 이수와 성취": r"전공|진로|과목|교과|이수|성취",
            "진로 탐색 활동과 경험": r"진로|관심|탐색|활동|경험",
        },
        "자기주도역량": {
            "자기주도적인 교과 선택과 이수": r"자기주도|스스로|선택|교과|과목|이수",
            "관심 분야 탐색 활동": r"관심|탐색|활동|경험|진로",
        },
        "공동체역량": {
            "협업·소통과 리더십": r"협력|협업|소통|토론|모둠|리더|이끌",
            "나눔·배려와 책임 있는 태도": r"나눔|배려|성실|책임|규칙|공동체|도와",
        },
    }
    area_indicators = indicators.get(area, {})
    shown = [label for label, pattern in area_indicators.items() if re.search(pattern, student_draft)]
    missing = [label for label in area_indicators if label not in shown]
    direction = {
        "학업역량": "학업 활동에서 수행한 과정, 판단 근거와 확인 가능한 성취",
        "탐구역량": "호기심이 생긴 계기, 조사·분석 방법과 탐구가 확장된 과정",
        "잠재역량": "스스로 맡은 역할, 협업 방식과 활동 전후의 변화",
        "전공적합성": "전공 관심이 생긴 계기, 관련 교과의 학습 과정과 진로 탐색 노력",
        "인성 및 사회성": "맡은 역할, 책임을 다한 과정과 구성원과 협력·소통한 방식",
        "진로역량": "전공 관련 교과의 선택·이수 과정, 성취와 진로 탐색 경험",
        "자기주도역량": "스스로 선택한 교과의 이수 과정과 관심 분야를 탐색한 경험",
        "공동체역량": "협업·소통 과정에서 맡은 역할과 나눔·배려·책임을 실천한 행동",
    }.get(area, "실제 활동 과정과 결과")
    return shown, missing, direction


def extract_college_criteria(
    documents: list[Document],
    student_draft: str,
    university: str,
    department: str = "",
) -> tuple[list[CollegeCriterion], list[int]]:
    """모델 추론 없이 모집요강에 퍼센트로 명시된 평가영역을 추출합니다."""
    pages = _group_pages(documents)
    candidates = []
    for page in pages:
        section = _student_comprehensive_section(unicodedata.normalize("NFKC", page["text"]))
        areas = _explicit_weighted_areas(section)
        unique_areas = list(dict.fromkeys((area, weight) for area, weight, _ in areas))
        total = sum(float(weight.rstrip("%")) for _, weight in unique_areas)
        if len(unique_areas) >= 2 and 99 <= total <= 101:
            candidates.append((_page_score(section, areas), page, section, unique_areas, {}))
    hierarchical = _hierarchical_weighted_candidate(pages, department)
    if hierarchical:
        hierarchical_page, hierarchical_section, hierarchical_areas, area_pages = hierarchical
        candidates.append((
            _page_score(hierarchical_section, []) + 20,
            hierarchical_page,
            hierarchical_section,
            hierarchical_areas,
            area_pages,
        ))
    if not candidates:
        return [], []

    _, page, section, weighted_areas, area_pages = max(candidates, key=lambda item: item[0])
    scored_subcriteria = _extract_subcriteria(section)
    explanations = _table_explanations(section)
    area_names = [area for area, _ in weighted_areas]
    all_text = "\n".join(item["text"] for item in pages)
    criteria: list[CollegeCriterion] = []
    evidence_ids = list(page["ids"])

    for position, (area, weight) in enumerate(weighted_areas):
        criterion_page = area_pages.get(area, page["page"])
        subcriteria = _subcriteria_for_area(area, all_text, scored_subcriteria, position)
        question, table_points = (
            explanations[position]
            if not area_pages and position < len(explanations)
            else ("", [])
        )
        if area_pages:
            detail_points, detail_ids, detail_page = _hierarchical_evaluation_points(
                area, pages, criterion_page
            )
        else:
            detail_points, detail_ids, detail_page = _area_evaluation_points(
                area, area_names, pages, criterion_page
            )
        points = table_points or detail_points
        evidence_pages = [criterion_page] if criterion_page is not None else []
        if detail_points and not table_points:
            evidence_ids.extend(detail_ids)
            if detail_page is not None:
                evidence_pages.append(detail_page)
        printed_pages = sorted({
            candidate["printed_page"]
            for candidate in pages
            if candidate["page"] in evidence_pages
            and candidate.get("printed_page") is not None
        })
        shown, missing, direction = _draft_assessment(area, student_draft)
        focus = ", ".join(subcriteria) or area
        official_focus = question or (", ".join(points) if points else focus)
        diagnosis = (
            f"현재 초안에는 {', '.join(shown)}이 드러납니다."
            if shown else "현재 초안에서는 이를 뒷받침할 구체적인 과정이 충분히 드러나지 않습니다."
        )
        if missing:
            diagnosis += f" 특히 {', '.join(missing)}이 부족합니다."
        criteria.append(CollegeCriterion(
            area=area,
            weight=weight,
            subcriteria=subcriteria,
            evaluation_question=question,
            evaluation_points=points,
            draft_evidence=shown,
            missing_aspects=missing,
            revision_direction=direction,
            recommendation=(
                f"{university}는 {area}{_object_particle(area)} {weight} 반영하며 "
                f"{focus}{_object_particle(focus)} 통해 '{official_focus}'를 중점적으로 확인합니다. "
                f"{diagnosis} 실제로 수행한 {direction}{_object_particle(direction)} 구체적으로 강조하는 것이 필요합니다."
            ),
            source=Path(page["source"]).name,
            page=criterion_page,
            evidence_pages=sorted(set(evidence_pages)),
            printed_pages=printed_pages,
        ))

    return criteria, list(dict.fromkeys(evidence_ids))
