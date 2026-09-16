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


def _compact(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"퍼센트|percent", "%", normalized)
    return re.sub(r"\s+", "", normalized)


def _explicit_weighted_areas(source: str) -> list[tuple[str, str, int]]:
    """`학업역량(40%)`처럼 원문에 명시된 비율만 찾습니다."""
    pattern = re.compile(
        r"(?:^|\s)([가-힣A-Za-z·]{1,20}역량)\s*\(\s*(\d+(?:\.\d+)?)\s*(?:%|퍼센트)\s*\)"
    )
    results = []
    offset = 0
    for line in unicodedata.normalize("NFKC", source).splitlines():
        for match in pattern.finditer(line):
            results.append((match.group(1), f"{match.group(2)}%", offset + match.start(1)))
        offset += len(_compact(line))
    return results


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
    }
    area_indicators = indicators.get(area, {})
    shown = [label for label, pattern in area_indicators.items() if re.search(pattern, student_draft)]
    missing = [label for label in area_indicators if label not in shown]
    direction = {
        "학업역량": "학업 활동에서 수행한 과정, 판단 근거와 확인 가능한 성취",
        "탐구역량": "호기심이 생긴 계기, 조사·분석 방법과 탐구가 확장된 과정",
        "잠재역량": "스스로 맡은 역할, 협업 방식과 활동 전후의 변화",
    }.get(area, "실제 활동 과정과 결과")
    return shown, missing, direction


def extract_college_criteria(
    documents: list[Document],
    student_draft: str,
    university: str,
) -> tuple[list[CollegeCriterion], list[int]]:
    """모델 추론 없이 모집요강에 퍼센트로 명시된 평가영역을 추출합니다."""
    criteria: list[CollegeCriterion] = []
    evidence_ids: list[int] = []
    seen: set[tuple[str, str]] = set()
    for evidence_id, document in enumerate(documents, start=1):
        weighted_areas = _explicit_weighted_areas(document.page_content)
        normalized = unicodedata.normalize("NFKC", document.page_content)
        all_subcriteria = _extract_subcriteria(normalized)
        explanations = _table_explanations(normalized)
        for position, (area, weight, _) in enumerate(weighted_areas):
            key = (area, weight)
            if key in seen:
                continue
            seen.add(key)
            subcriteria = all_subcriteria[position * 2:position * 2 + 2]
            question, points = explanations[position] if position < len(explanations) else ("", [])
            shown, missing, direction = _draft_assessment(area, student_draft)
            focus = ", ".join(subcriteria) or area
            official_focus = question or (", ".join(points) if points else focus)
            if shown:
                diagnosis = f"현재 초안에는 {', '.join(shown)}이 드러납니다."
            else:
                diagnosis = "현재 초안에서는 이를 뒷받침할 구체적인 과정이 충분히 드러나지 않습니다."
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
                source=Path(document.metadata["source"]).name,
                page=document.metadata.get("page"),
            ))
            evidence_ids.append(evidence_id)

    return criteria, list(dict.fromkeys(evidence_ids))
