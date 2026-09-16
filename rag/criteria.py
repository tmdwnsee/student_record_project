"""대학 모집요강 원문에서 평가영역과 반영비율을 추출합니다."""

import re
import unicodedata
from pathlib import Path

from langchain_core.documents import Document
from pydantic import BaseModel


class CollegeCriterion(BaseModel):
    area: str
    weight: str
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


def extract_college_criteria(
    documents: list[Document],
    student_draft: str,
    university: str,
) -> tuple[list[CollegeCriterion], list[int]]:
    """모델 추론 없이 모집요강에 퍼센트로 명시된 평가영역을 추출합니다."""
    directions = {
        "학업역량": "학업 활동의 구체적인 과정, 성취 수준, 학업 태도",
        "탐구역량": "탐구 질문, 분석 과정, 사용한 방법, 발견한 결과",
        "잠재역량": "자기주도적으로 맡은 역할, 협업 방식, 활동의 발전 과정",
    }
    criteria: list[CollegeCriterion] = []
    evidence_ids: list[int] = []
    seen: set[tuple[str, str]] = set()
    draft_excerpt = " ".join(student_draft.split())[:80]

    for evidence_id, document in enumerate(documents, start=1):
        for area, weight, _ in _explicit_weighted_areas(document.page_content):
            key = (area, weight)
            if key in seen:
                continue
            seen.add(key)
            direction = directions.get(area, "평가영역과 관련된 실제 활동 과정과 결과")
            criteria.append(CollegeCriterion(
                area=area,
                weight=weight,
                recommendation=(
                    f"현재 초안 '{draft_excerpt}'만으로는 {area}의 구체적인 근거가 충분한지 판단하기 어렵습니다. "
                    f"{university}는 {area}을 {weight} 반영하므로, {direction} 중 실제로 수행한 내용만 확인해 보완하세요."
                ),
                source=Path(document.metadata["source"]).name,
                page=document.metadata.get("page"),
            ))
            evidence_ids.append(evidence_id)

    return criteria, list(dict.fromkeys(evidence_ids))
