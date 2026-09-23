"""업로드한 기존 생기부의 전체 텍스트를 읽고 관련 활동을 고릅니다."""

import re
import tempfile
from difflib import SequenceMatcher
from functools import lru_cache
from math import sqrt
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from ingestion.pdf_pipeline import (
    PipelineConfig,
    locate_student_record_pages,
    process_pdf,
    release_ocr_engines,
)
from config import MODEL, OLLAMA_BASE_URL
from storage.embeddings import get_embeddings

MAX_FILE_BYTES = 10 * 1024 * 1024
CHUNK_SIZE = 5_000
CHUNK_OVERLAP = 200
MAX_SELECTED_CHUNKS = 3

SECTION_PATTERNS = {
    "세부능력특기사항": r"(?m)^[ \t]*세[ \t]*부[ \t]*능[ \t]*력[ \t]*(?:및[ \t]*)?특[ \t]*기[ \t]*사[ \t]*항[ \t]*$",
    "동아리활동": r"(?m)^[ \t]*동[ \t]*아[ \t]*리[ \t]*활[ \t]*동[ \t]*$",
    "자율자치활동": r"(?m)^[ \t]*자[ \t]*율(?:[ \t]*[·ㆍ･/]?[ \t]*자[ \t]*치)?[ \t]*활[ \t]*동[ \t]*$",
    "진로활동": r"(?m)^[ \t]*진[ \t]*로[ \t]*활[ \t]*동[ \t]*$",
}
SUBJECT_HEADING_PATTERN = re.compile(
    r"(?m)^[ \t]*(?=[^\n]{1,25}[ \t]*$)"
    r"(?=[^\n]*(?:국어|수학|영어|사회|과학|물리|화학|생명|지구|역사|윤리|지리|경제|정치|법|정보|"
    r"기술|가정|체육|음악|미술|한문|외국어|프로그래밍|인공지능|문학|독서|언어|확률|미적분|기하))"
    r"[가-힣A-Za-z0-9ⅠⅡⅢ·ㆍ\- ]+[ \t]*$",
    re.I,
)


class CandidateVerdict(BaseModel):
    candidate_number: int = Field(description="검증한 후보 번호")
    experience_title: str = Field(default="", max_length=40, description="원문에서 실제 수행한 활동을 나타내는 짧은 명사형 제목. 주제·행동·결과물로 후보들을 구분하며 '참고한 기존 경험' 같은 공통 제목이나 미래 활동은 쓰지 않음")
    relevance: int = Field(description="새 초안과의 관련성: 0 없음, 1 약함, 2 관련, 3 직접 관련")
    text_integrity: int = Field(description="원문 무결성: 0 깨짐·의미 훼손, 1 경미한 서식 흔적이나 의미는 완전함, 2 완전한 문장")
    integrity_issue: str = Field(default="", description="원문이 깨졌거나 의심스러울 때 그 이유")
    corrected_original: str = Field(
        default="",
        description="OCR 철자·조사 오류만 문맥에 맞게 고친 전체 원문. 사실·수치·활동을 추가하거나 요약하지 않음",
    )
    connection_reason: str = Field(
        description="이 기존 경험이 자기평가보고서 초안·희망 학과·선택 활동 구분과 관련 있다고 판단한 이유"
    )


class RecordSelection(BaseModel):
    verdicts: list[CandidateVerdict] = Field(description="후보별 관련성 검증 결과")


class OcrCorrection(BaseModel):
    candidate_number: int = Field(description="보정한 후보 번호")
    corrected_original: str = Field(
        description="사실·수치·문장 순서를 유지하고 OCR 철자·조사·단어 내부 공백만 고친 전체 원문"
    )


class OcrCorrectionBatch(BaseModel):
    corrections: list[OcrCorrection] = Field(description="요청된 모든 후보의 OCR 보정 결과")


RECORD_CANDIDATE_COUNT = 12
RECORD_MIN_SIMILARITY = 0.40
RECORD_RRF_K = 60
RECORD_MMR_LAMBDA = 0.78
_ALLOWED_SINGLE_HANGUL = {
    "이", "그", "저", "한", "두", "세", "네", "첫", "각", "및", "더", "새", "전", "후", "내",
    "중", "때", "줄", "수", "또", "데", "서", "게", "지", "할", "말", "풀", "길", "생",
}
_FOREIGN_SCRIPT_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_FOREIGN_TERM_REPLACEMENTS = {"朗誦": "낭송"}


def sanitize_generated_korean(text: str) -> str:
    """사용자 표시용 생성문에서 원문에 없던 외국 문자를 제거합니다."""
    cleaned = text
    for foreign, korean in _FOREIGN_TERM_REPLACEMENTS.items():
        cleaned = cleaned.replace(foreign, korean)
    cleaned = _FOREIGN_SCRIPT_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\s+([,.:;!?])", r"\1", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def extract_context(filename: str, content: bytes, record_section: str = "") -> str:
    if not content or len(content) > MAX_FILE_BYTES:
        raise ValueError("첨부 파일은 비어 있지 않은 10MB 이하 파일이어야 합니다.")
    suffix = filename.lower().rsplit(".", 1)[-1]
    try:
        if suffix == "pdf":
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "uploaded_record.pdf"
                path.write_bytes(content)
                selected_pages = locate_student_record_pages(path, record_section)
                pages = process_pdf(
                    path,
                    document_type="student_record",
                    # EasyOCR의 기본 입력 크기(긴 변 2,560px)보다 큰 렌더링은 다시
                    # 축소되므로 300 DPI는 시간만 늘고 한글 인식 결과는 거의 같습니다.
                    config=PipelineConfig(dpi=220),
                    page_numbers=selected_pages,
                )
                text = clean_record_ocr_text("\n".join(page.text for page in pages))
                release_ocr_engines()
        elif suffix == "txt":
            text = content.decode("utf-8-sig")
        else:
            raise ValueError("PDF 또는 UTF-8 TXT 파일을 첨부하세요.")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("첨부 파일에서 텍스트를 읽지 못했습니다. PDF 또는 UTF-8 TXT 파일을 확인하세요.") from error
    text = text.strip()
    if not text:
        raise ValueError("첨부 파일에서 텍스트를 추출하지 못했습니다.")
    return text


_RECORD_TABLE_LABELS = {
    "학년", "영역", "시간", "특기사항", "희망분야", "구분", "학과",
    "반", "번호", "이름", "담임성명", "학생정보", "주소",
}


def clean_record_ocr_text(text: str) -> str:
    """스캔 생기부의 표 셀·전자문서 머리말을 제거하고 활동 본문은 보존합니다."""
    cleaned: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        compact = re.sub(r"\s+", "", line)
        if not line:
            if cleaned and cleaned[-1]:
                cleaned.append("")
            continue
        if compact in _RECORD_TABLE_LABELS:
            continue
        if re.fullmatch(r"\d{1,3}(?:/\d{1,3})?", compact):
            continue
        if re.fullmatch(r"\d{4}년\d{1,2}월\d{1,2}일", compact):
            continue
        if re.fullmatch(r"[가-힣A-Za-z]+(?:중|고등?)학교", compact):
            continue
        if any(marker in compact for marker in (
            "문서확인번호:", "발급일로부터90일까지", "문서하단의바코드로도진위확인",
            "인터넷발급문서진위확인메뉴", "정부24(gov.kr)",
        )):
            continue
        # 스캔 얼룩이 한 글자로 인식된 행은 활동 문장으로 사용하지 않습니다.
        if len(compact) == 1 and not compact.isdigit():
            continue
        cleaned.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


def split_record(text: str) -> list[str]:
    """문서 전체를 순서대로 분할합니다. 마지막 구간도 빠뜨리지 않습니다."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    ).split_text(text)


def split_record_units(chunk: str) -> list[str]:
    """PDF 줄바꿈에 문장이 잘리지 않도록 문장부호 기준으로 원문을 나눕니다."""
    units = []
    start = 0
    for boundary in re.finditer(
        # 숫자 바로 뒤의 마침표는 날짜·학년도·소수점일 수 있으므로 경계로
        # 사용하지 않습니다. 생기부 문장은 대개 '함/됨/보임'으로도 끝납니다.
        r"(?:(?<!\d)[.!?]|함|됨|보임|있음|없음|였음|었음|았음|했음|되었음|읽음|높음|낮음)(?=\s|$)",
        chunk,
    ):
        unit = chunk[start:boundary.end()].strip()
        if unit:
            units.append(unit)
        start = boundary.end()
    tail = chunk[start:].strip()
    if tail:
        units.append(tail)
    return units or ([chunk.strip()] if chunk.strip() else [])


def _section_ranges(text: str, requested_section: str) -> list[tuple[int, int]]:
    """생기부에 활동 항목 표제가 있으면 해당 항목의 본문 범위만 반환합니다."""
    headings = []
    for section, pattern in SECTION_PATTERNS.items():
        headings.extend((match.start(), match.end(), section) for match in re.finditer(pattern, text, re.I))
    headings.sort()
    ranges = []
    for index, (start, heading_end, section) in enumerate(headings):
        if section != requested_section:
            continue
        end = headings[index + 1][0] if index + 1 < len(headings) else len(text)
        ranges.append((heading_end, end))
    return ranges


def _subject_ranges(text: str, subject: str, *, window: int = 5_000) -> list[tuple[int, int]]:
    """세특 표에서 선택 과목명이 나온 위치 주변으로 후보 범위를 제한합니다."""
    compact_subject = re.sub(r"\s+", "", subject)
    if not compact_subject:
        return []
    pattern = r"\s*".join(re.escape(char) for char in compact_subject)
    ranges = []
    for match in re.finditer(pattern, text, re.I):
        end = min(len(text), match.end() + window)
        next_heading = SUBJECT_HEADING_PATTERN.search(text, match.end())
        if next_heading and next_heading.start() < end:
            end = next_heading.start()
        ranges.append((max(0, match.start() - 200), end))
    return ranges


def _in_ranges(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start < range_end and end > range_start for range_start, range_end in ranges)


def _unit_offsets(text: str, units: list[str]) -> list[tuple[int, int]]:
    """원문 단위의 시작·끝 위치를 원본 문자열에서 찾습니다."""
    offsets = []
    cursor = 0
    for unit in units:
        position = text.find(unit, cursor)
        if position < 0:
            return []
        offsets.append((position, position + len(unit)))
        cursor = position + len(unit)
    return offsets


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


@lru_cache(maxsize=8)
def _embed_record_units_cached(texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
    """같은 생기부를 다시 분석할 때 문장 임베딩을 재사용합니다."""
    vectors = get_embeddings().embed_documents(list(texts))
    return tuple(tuple(vector) for vector in vectors)


def _obvious_text_corruption(text: str) -> str:
    """LLM 검증 전에 명백한 PDF/OCR 혼입을 보수적으로 차단합니다."""
    if "�" in text or any(ord(char) < 32 and char not in "\n\t\r" for char in text):
        return "대체문자 또는 제어문자 포함"
    isolated = [
        token for token in re.findall(r"(?<![가-힣\r\n])([가-힣])(?![가-힣\r\n])", text)
        if token not in _ALLOWED_SINGLE_HANGUL
    ]
    if len(isolated) >= 2:
        return "문장 중간에 표 셀이나 머리글로 보이는 한 글자 조각이 반복됨"
    return ""


def _grounded_experience_title(title: str, original: str) -> bool:
    """과거 경험 제목의 핵심 단어가 실제 생기부 후보에 있는지 확인합니다."""
    if _FOREIGN_SCRIPT_PATTERN.search(title) or re.search(r"(?:후보|근거)\s*\d+", title):
        return False
    stopwords = {"기존", "경험", "활동", "통한", "관련", "대한", "기반", "함양", "강화"}
    tokens = [
        token for token in re.findall(r"[가-힣A-Za-z]{2,}", title)
        if token not in stopwords
    ]
    if not tokens:
        return False
    compact_original = re.sub(r"\s+", "", original).lower()
    grounded = sum(token.lower() in compact_original for token in tokens)
    return grounded / len(tokens) >= 0.5


def _fallback_experience_title(original: str, record_section: str) -> str:
    cleaned = re.sub(r"\s+", " ", original).strip()
    cleaned = re.sub(r"^(?:진로활동|자율활동|자율자치활동|동아리활동)\s*", "", cleaned)
    excerpt = cleaned[:32].rstrip(" ,.")
    return f"기존 {record_section or '생기부'} 경험 · {excerpt}" + ("…" if len(cleaned) > 32 else "")


def _safe_corrected_ocr(original: str, corrected: str) -> str:
    """수치와 전체 형태를 보존한 OCR 보정문만 표시용으로 허용합니다."""
    corrected = re.sub(r"[ \t]+\n", "\n", corrected).strip()
    # 보정 모델이 글자는 바로잡으면서 숫자·조사 사이에 새 공백을
    # 만들 때가 있다. 의미를 바꾸지 않는 명확한 경우만 후처리한다.
    corrected = re.sub(r"(?<=\d)\s+(?=학기\b)", "", corrected)
    corrected = re.sub(r"\)\s+(?=(?:으로|에서|까지|부터|에)\b)", ")", corrected)
    corrected = re.sub(
        r"(?<![은는이가을를에도와과로])\s+시키(?=는|며|고|기|도록|려고|었다|었음|다)",
        "시키",
        corrected,
    )
    if not corrected:
        return original
    original_compact = re.sub(r"\s+", "", original)
    corrected_compact = re.sub(r"\s+", "", corrected)
    if not original_compact or not 0.70 <= len(corrected_compact) / len(original_compact) <= 1.30:
        return original
    original_foreign_chars = set(_FOREIGN_SCRIPT_PATTERN.findall(original))
    if any(
        char not in original_foreign_chars
        for char in _FOREIGN_SCRIPT_PATTERN.findall(corrected)
    ):
        return original
    # OCR이 연도나 날짜 중간에 넣은 공백은 보정할 수 있어야 합니다.
    # 공백을 제거한 뒤 숫자와 날짜 구분자를 함께 비교하면 `202 1` ->
    # `2021`은 허용하면서 실제 날짜·수치 변경은 계속 차단할 수 있습니다.
    numeric_pattern = r"\d+(?:[-./:]\d+)*"
    original_numbers = re.findall(numeric_pattern, original_compact)
    corrected_numbers = re.findall(numeric_pattern, corrected_compact)
    if original_numbers != corrected_numbers:
        return original
    if SequenceMatcher(None, original_compact, corrected_compact).ratio() < 0.55:
        return original
    return corrected


def _needs_ocr_correction(text: str) -> bool:
    """전용 OCR 보정을 재시도할 만큼 명백한 인식 흔적이 있는지 확인합니다."""
    return bool(
        re.search(r"(?<=\d)\s+(?=\d)", text)
        or re.search(r"[가-힣]{3,}(?:올|울|름|틀|록|논)(?=\s|[,.:;!?)]|$)", text)
        or re.search(r"[가-힣]{1,3}\s+[가-힣]{2,}(?:올|울|름|틀|록|논)(?=\s|[,.:;!?)]|$)", text)
        or re.search(r"(?:선출|관찰|발견|확인)\s+[팀됨](?=\s|[,.:;!?)]|$)", text)
    )


_OCR_SUFFIX_EXCEPTIONS = {
    "서울", "마을", "거울", "가을", "겨울", "세월", "노을",
    "기록", "목록", "등록", "수록", "초록", "도록",
    "이름", "흐름", "구름", "필름", "얼음",
    "토론", "언론", "물론", "결론", "이론", "논",
}
_OCR_SUFFIX_REPLACEMENTS = {
    "올": "을", "울": "을", "틀": "를", "록": "를", "름": "를", "논": "는",
}


def _ocr_correction_suggestions(text: str) -> list[str]:
    """정상 단어는 보존하면서 흔한 한글 OCR 조사·어미 혼동 후보를 만듭니다."""
    suggestions: list[str] = []
    for token in re.findall(r"[가-힣]{3,}", text):
        if token in _OCR_SUFFIX_EXCEPTIONS:
            continue
        suffix = token[-1]
        replacement = _OCR_SUFFIX_REPLACEMENTS.get(suffix)
        if replacement:
            suggestions.append(f"{token}→{token[:-1]}{replacement}")
    for match in re.finditer(r"(선출|관찰|발견|확인)\s+팀(?=\s|[,.:;!?)]|$)", text):
        suggestions.append(f"{match.group(0)}→{match.group(1)}됨")
    return list(dict.fromkeys(suggestions))


def _retry_ocr_corrections(candidates: list[dict], candidate_numbers: list[int]) -> dict[int, str]:
    """복합 판단에서 누락된 OCR 보정만 작은 전용 요청으로 다시 수행합니다."""
    if not candidate_numbers:
        return {}
    requested = [
        (number, candidates[number - 1]["original"])
        for number in candidate_numbers
        if 1 <= number <= len(candidates)
    ]
    if not requested:
        return {}
    correction_model = ChatOllama(
        model=MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        reasoning=False,
        num_ctx=4_096,
    ).with_structured_output(OcrCorrectionBatch, method="json_schema")
    candidate_blocks = []
    for number, original in requested:
        suspicious_tokens = re.findall(
            r"[가-힣]{2,}(?:올|울|틀|록|름|논)(?=\s|[,.:;!?)]|$)|"
            r"(?:선출|관찰|발견|확인)\s+[팀됨](?=\s|[,.:;!?)]|$)",
            original,
        )
        suggestions = _ocr_correction_suggestions(original)
        token_hint = ""
        if suspicious_tokens:
            token_hint += "\nOCR 혼동 의심 토큰: " + ", ".join(dict.fromkeys(suspicious_tokens))
        if suggestions:
            token_hint += "\n문맥 검토용 교정 후보: " + ", ".join(suggestions)
        candidate_blocks.append(f"후보 {number}\n원문: {original}{token_hint}")
    candidate_text = "\n\n".join(candidate_blocks)
    base_system_prompt = (
        "스캔 학교생활기록부의 OCR 오류만 보정한다. 요청된 모든 후보를 정확히 한 번씩 반환한다. "
        "잘못 인식된 철자와 조사, 단어 내부에 끼어든 공백, 숫자 내부에 끼어든 공백만 고친다. "
        "한글 OCR은 목적격 조사 '을/를'을 '올/울/틀/록/름'으로, 관형형 어미 '는'을 '논'으로, "
        "명사형 종결 '됨'을 '팀'으로 혼동할 수 있다. 앞뒤 문법과 문맥이 분명할 때 이러한 혼동을 바로잡는다. "
        "원문 전체와 문장 순서를 유지하고 날짜의 숫자, 기간, 활동, 역할, 평가 내용은 절대 바꾸거나 추가·삭제·요약하지 않는다. "
        "보정문은 현대 한국어 한글로 작성하며 원문에 없던 한자, 중국어, 일본어 문자를 절대 넣지 않는다. "
        "오류가 없으면 원문을 그대로 반환한다."
    )
    try:
        result = correction_model.invoke([
            ("system", base_system_prompt),
            ("human", candidate_text),
        ])
    except Exception:
        return {}
    if not isinstance(result, OcrCorrectionBatch):
        return {}
    allowed = set(candidate_numbers)
    corrections = {
        item.candidate_number: item.corrected_original
        for item in result.corrections
        if item.candidate_number in allowed
    }
    original_by_number = dict(requested)
    invalid_numbers = [
        number for number, corrected in corrections.items()
        if any(
            char not in set(_FOREIGN_SCRIPT_PATTERN.findall(original_by_number[number]))
            for char in _FOREIGN_SCRIPT_PATTERN.findall(corrected)
        )
    ]
    unresolved_numbers = [
        number
        for number, corrected in corrections.items()
        if any(
            token in corrected
            for token in re.findall(
                r"[가-힣]{2,}(?:올|울|틀|록|름|논)(?=\s|[,.:;!?)]|$)|"
                r"(?:선출|관찰|발견|확인)\s+[팀됨](?=\s|[,.:;!?)]|$)",
                original_by_number[number],
            )
        )
    ]
    invalid_numbers = list(dict.fromkeys(invalid_numbers + unresolved_numbers))
    if invalid_numbers:
        retry_text = "\n\n".join(
            f"후보 {number}\n원문: {original_by_number[number]}\n"
            f"아직 오류가 의심되어 다시 검토할 보정문: {corrections[number]}"
            for number in invalid_numbers
        )
        try:
            second_result = correction_model.invoke([
                ("system", base_system_prompt + " 직전 보정문에 남은 OCR 혼동 의심 토큰은 앞뒤 문법을 확인해 반드시 다시 검토한다. 정상 단어가 아닌 조사·어미 오인은 그대로 반환하지 않는다."),
                ("human", retry_text),
            ])
        except Exception:
            second_result = None
        if isinstance(second_result, OcrCorrectionBatch):
            for item in second_result.corrections:
                if item.candidate_number in invalid_numbers:
                    corrections[item.candidate_number] = item.corrected_original
    return {
        number: corrected for number, corrected in corrections.items()
        if _safe_corrected_ocr(original_by_number[number], corrected) != original_by_number[number]
    }


def _repair_layout_intrusions(text: str, noise_terms: list[str]) -> tuple[str, bool]:
    """PDF 표에서 본문 중간에 삽입된 입력 과목명을 제거해 분리된 단어를 복원합니다."""
    repaired = text
    changed = False
    for term in sorted({term.strip() for term in noise_terms if term.strip()}, key=len, reverse=True):
        # 같은 줄 안에 표 셀이 끼어든 경우만 복원합니다. 줄 하나를 차지하는 정상 과목 표제는 보존합니다.
        pattern = re.compile(rf"(?<=[가-힣A-Za-z0-9])[ \t]+{re.escape(term)}[ \t]+(?=[가-힣A-Za-z0-9])")
        repaired, count = pattern.subn("", repaired)
        changed = changed or count > 0
    return repaired, changed


def _retrieve_record_candidates(
    text: str,
    query: str,
    layout_noise_terms: list[str] | None = None,
    *,
    retrieval_queries: list[str] | None = None,
    record_section: str = "",
    subject: str = "",
) -> list[dict]:
    """다중 질의 Hit/RRF와 MMR로 관련성·다양성을 함께 확보합니다."""
    units = split_record_units(text)
    offsets = _unit_offsets(text, units)
    if not units or not offsets or not query.strip():
        return []

    cleaned_units = []
    repaired_flags = []
    for unit in units:
        cleaned, was_repaired = _repair_layout_intrusions(unit, layout_noise_terms or [])
        cleaned_units.append(cleaned)
        repaired_flags.append(was_repaired)

    section_ranges = _section_ranges(text, record_section) if record_section else []
    subject_ranges = _subject_ranges(text, subject) if record_section == "세부능력특기사항" else []

    if record_section and not section_ranges:
        return []
    allowed_indexes = list(range(len(units)))
    if section_ranges:
        allowed_indexes = [
            index for index, (start, end) in enumerate(offsets)
            if _in_ranges(start, end, section_ranges)
        ]
    if subject_ranges:
        subject_indexes = [
            index for index, (start, end) in enumerate(offsets)
            if _in_ranges(start, end, subject_ranges)
        ]
        # 과목명이 실제로 발견된 경우에는 해당 과목 주변 문장만 사용합니다.
        if subject_indexes:
            allowed_indexes = [index for index in allowed_indexes if index in set(subject_indexes)]
    if not allowed_indexes:
        return []

    embeddings = get_embeddings()
    searches = [item.strip() for item in (retrieval_queries or [query]) if item.strip()]
    # 선택한 활동 구간 밖의 문장은 이후 검색에서 버려지므로 임베딩하지 않습니다.
    allowed_units = tuple(cleaned_units[index] for index in allowed_indexes)
    query_vectors = embeddings.embed_documents(searches)
    unit_vectors = _embed_record_units_cached(allowed_units)
    candidates = []
    for index, vector in zip(allowed_indexes, unit_vectors):
        start, end = offsets[index]
        cleaned = cleaned_units[index]
        was_repaired = repaired_flags[index]
        unit_index = index + 1
        query_scores = [_cosine_similarity(query_vector, vector) for query_vector in query_vectors]
        candidates.append({
            "unit_index": unit_index,
            "original": cleaned,
            "raw_original": text[start:end],
            "text_was_repaired": was_repaired,
            "query_scores": query_scores,
            "similarity_score": max(query_scores),
            "draft_similarity": query_scores[0],
            "_vector": vector,
        })

    if not candidates:
        return []

    # 여러 검색 초점에서 반복해 상위권에 든 횟수(Hit)와 순위(RRF)를 계산합니다.
    per_query = max(2, RECORD_CANDIDATE_COUNT // len(query_vectors))
    for candidate in candidates:
        candidate["query_hit_count"] = 0
        candidate["rrf_score"] = 0.0
    for query_index in range(len(query_vectors)):
        ranked = sorted(
            candidates,
            key=lambda item: (-item["query_scores"][query_index], item["unit_index"]),
        )
        for rank, candidate in enumerate(ranked, 1):
            candidate["rrf_score"] += 1.0 / (RECORD_RRF_K + rank)
            if rank <= per_query:
                candidate["query_hit_count"] += 1

    max_rrf = max(candidate["rrf_score"] for candidate in candidates) or 1.0
    for candidate in candidates:
        candidate["rrf_score"] /= max_rrf
        candidate["retrieval_score"] = (
            candidate["similarity_score"] * 0.85 + candidate["rrf_score"] * 0.15
        )
        candidate["retrieval_query_number"] = (
            max(range(len(candidate["query_scores"])), key=candidate["query_scores"].__getitem__) + 1
        )

    # MMR은 같은 내용의 문장만 반복 선정되는 것을 막습니다.
    pool = list(candidates)
    selected: list[dict] = []
    while pool and len(selected) < RECORD_CANDIDATE_COUNT:
        def mmr(candidate: dict) -> tuple[float, int, float, int]:
            redundancy = max(
                (_cosine_similarity(candidate["_vector"], item["_vector"]) for item in selected),
                default=0.0,
            )
            score = (
                RECORD_MMR_LAMBDA * candidate["retrieval_score"]
                - (1.0 - RECORD_MMR_LAMBDA) * max(0.0, redundancy)
            )
            return (
                score,
                candidate["query_hit_count"],
                candidate["similarity_score"],
                -candidate["unit_index"],
            )

        winner = max(pool, key=mmr)
        winner["mmr_score"] = mmr(winner)[0]
        selected.append(winner)
        pool = [candidate for candidate in pool if candidate is not winner]
    return [
        {
            **{key: value for key, value in candidate.items() if key != "_vector"},
            "rrf_score": round(candidate["rrf_score"], 4),
            "mmr_score": round(candidate["mmr_score"], 4),
        }
        for candidate in selected
    ]


def _format_record_context(matches: list[dict]) -> str:
    if not matches:
        return "기존 생기부를 검토했으나 관련성과 원문 무결성 검증을 모두 통과한 활동을 찾지 못했습니다."
    return "\n\n".join(
        f"[과거 근거 {index}: 기존 생기부]\n선정 이유: {match['connection_reason']}\n"
        f"원문:\n{match.get('display_original', match['original'])}"
        for index, match in enumerate(matches, 1)
    )


def _select_grounded_candidates(candidates: list[dict], limit: int) -> list[dict]:
    """검색 점수와 원문 무결성으로 근거를 확정하고 LLM의 거절값에 의존하지 않습니다."""
    readable = [
        candidate for candidate in candidates
        if len(re.findall(r"[가-힣A-Za-z0-9]", candidate["original"])) >= 8
        and not _obvious_text_corruption(candidate["original"])
    ]
    if not readable:
        return []
    best_similarity = max(candidate["similarity_score"] for candidate in readable)
    best_draft_similarity = max(candidate["draft_similarity"] for candidate in readable)
    similarity_floor = max(RECORD_MIN_SIMILARITY, best_similarity - 0.12)
    draft_floor = max(0.32, best_draft_similarity - 0.15)
    return [
        candidate for candidate in readable
        if candidate["similarity_score"] >= similarity_floor
        and (
            candidate["draft_similarity"] >= draft_floor
            or candidate["query_hit_count"] >= 2
        )
    ][:limit]


def prepare_record_context(
    text: str,
    draft: str,
    max_selected_chunks: int = MAX_SELECTED_CHUNKS,
    *,
    return_matches: bool = False,
    layout_noise_terms: list[str] | None = None,
    retrieval_queries: list[str] | None = None,
    record_section: str = "",
    subject: str = "",
) -> str | tuple[str, list[dict]]:
    """Hit/RRF/MMR로 근거를 고르고 선택 문장의 OCR만 보정합니다."""
    if not text.strip():
        return ("", []) if return_matches else ""
    candidates = _retrieve_record_candidates(
        text,
        draft,
        layout_noise_terms,
        retrieval_queries=retrieval_queries,
        record_section=record_section,
        subject=subject,
    )
    candidates = _select_grounded_candidates(candidates, max_selected_chunks)
    if not candidates:
        context = _format_record_context([])
        return (context, []) if return_matches else context

    # 후보 선별은 코드에서 끝났습니다. LLM에게 관련성을 다시 평가시키지 않고,
    # 사용자에게 표시할 선택 문장의 명백한 OCR 오류만 한 번에 보정합니다.
    candidate_numbers = list(range(1, len(candidates) + 1))
    corrections = _retry_ocr_corrections(candidates, candidate_numbers)
    matches = []
    for candidate_number, candidate in enumerate(candidates, 1):
        display_original = _safe_corrected_ocr(
            candidate["original"], corrections.get(candidate_number, "")
        )
        experience_title = _fallback_experience_title(display_original, record_section)
        matches.append({
            "experience_title": experience_title,
            "original": candidate["original"],
            "display_original": display_original,
            "connection_reason": (
                "기존 생기부에서 확인되는 이 활동의 주제·방법 또는 역할이 "
                "현재 자기평가보고서와 연결되어 참고 경험으로 선정되었습니다."
            ),
            "similarity_score": round(candidate["similarity_score"], 3),
            "query_hit_count": candidate["query_hit_count"],
            "rrf_score": candidate["rrf_score"],
            "mmr_score": candidate["mmr_score"],
            "retrieval_query_number": candidate.get("retrieval_query_number", 1),
            "llm_relevance": None,
            "text_integrity": None,
            "text_was_repaired": candidate["text_was_repaired"],
        })
    context = _format_record_context(matches)
    return (context, matches) if return_matches else context
