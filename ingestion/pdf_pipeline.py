"""RAG용 PDF 단계별 추출 파이프라인.

순서:
1) pdftotext -layout
2) PyPDF text
3) PyMuPDF text / blocks
4) 각 추출 결과의 텍스트 품질 검사
5) 한글 문서는 EasyOCR, 부족하면 RapidOCR
6) Vision hook (선택)

문서별로 페이지 단위 Document를 만든 뒤, 페이지 메타데이터를 보존한 상태로
후속 chunking/retrieval에 넘기는 것을 목표로 합니다.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterable, Optional

import pymupdf
from langchain_core.documents import Document


VISION_HOOK = Callable[[bytes, dict], str]


@dataclass
class QualityReport:
    ok: bool
    score: float
    char_count: int
    korean_ratio: float
    alnum_ratio: float
    noise_ratio: float
    heading_hits: int
    expected_hits: int
    issues: list[str] = field(default_factory=list)


@dataclass
class PageResult:
    page_number: int
    printed_page: Optional[int]
    text: str
    method: str
    quality: QualityReport
    metadata: dict


@dataclass
class PipelineConfig:
    min_chars: int = 80
    dpi: int = 180
    enable_ocr: bool = True
    enable_korean_ocr: bool = True
    pymupdf_sort: bool = True
    prefer_pymupdf: bool = False


# PDF 문서별 검색 품질을 높이기 위한 표준 헤딩 후보
HEADING_PATTERNS = [
    r"학교생활기록부",
    r"학생부종합전형",
    r"서류평가",
    r"평가요소",
    r"평가 영역",
    r"반영비율|반영 비율",
    r"세부능력 및 특기사항",
    r"창의적 체험활동",
    r"기재",
]


def _run_pdftotext(
    path: Path, first_page: int | None = None, last_page: int | None = None,
) -> list[str]:
    """pdftotext -layout 결과를 페이지 단위 문자열로 반환."""
    try:
        command = ["pdftotext", "-layout"]
        if first_page is not None:
            command.extend(["-f", str(first_page)])
        if last_page is not None:
            command.extend(["-l", str(last_page)])
        command.extend([str(path), "-"])
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Poppler가 없거나 추출에 실패하면 페이지별 PyMuPDF 단계부터 진행합니다.
        return []
    return proc.stdout.split("\f")


def _korean_ratio(text: str) -> float:
    letters = [c for c in text if not c.isspace()]
    if not letters:
        return 0.0
    ko = sum("가" <= c <= "힣" for c in letters)
    return ko / len(letters)


def _alnum_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(c.isalnum() for c in chars) / len(chars)


def _noise_ratio(text: str) -> float:
    if not text:
        return 1.0
    # PDF 파싱 실패 시 자주 나타나는 replacement char / 제어문자 / 반복 기호 비율
    replacement = text.count("�")
    controls = sum(ord(c) < 32 and c not in "\n\t\r" for c in text)
    repeated = len(re.findall(r"(?:[_▪•·]{5,}|[=/\\]{5,}|([^\s])\1{4,})", text))
    denom = max(1, len(text))
    return min(1.0, (replacement + controls + repeated * 4) / denom)


def quality_check(
    text: str,
    *,
    expected_terms: Iterable[str] = (),
    min_chars: int = 80,
) -> QualityReport:
    expected_terms = list(expected_terms)
    headings = sum(bool(re.search(p, text, re.I)) for p in HEADING_PATTERNS)
    expected = sum(1 for term in expected_terms if term and term in text)
    kr = _korean_ratio(text)
    alnum = _alnum_ratio(text)
    noise = _noise_ratio(text)

    issues: list[str] = []
    if len(text.strip()) < min_chars:
        issues.append("text_too_short")
    if noise > 0.02:
        issues.append("high_noise")
    if kr < 0.05 and alnum < 0.25:
        issues.append("likely_image_or_unreadable")
    if headings == 0 and len(text.strip()) < 250:
        issues.append("no_semantic_anchor")

    # 검색에 쓰기 위한 휴리스틱 점수
    score = 0.0
    score += min(0.30, len(text.strip()) / 1500 * 0.30)
    score += min(0.25, alnum * 0.25)
    score += min(0.15, kr * 0.15)
    score += min(0.15, headings / 3 * 0.15)
    score += min(0.15, expected / max(1, len(expected_terms)) * 0.15)
    score -= min(0.25, noise * 3.0)
    score = max(0.0, min(1.0, score))

    readable = len(text.strip()) >= min_chars and noise <= 0.02 and (kr >= 0.05 or alnum >= 0.25)
    expected_content_found = not expected_terms or expected > 0
    ok = readable and expected_content_found
    return QualityReport(ok, score, len(text), kr, alnum, noise, headings, expected, issues)


def _clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pymupdf_text(
    doc: pymupdf.Document, page_number: int, *, sort: bool = True,
) -> str:
    return _clean_text(doc[page_number - 1].get_text("text", sort=sort))


def open_pypdf(path: Path):
    """PyPDF로 문서를 열고, 읽을 수 없으면 다음 파서가 처리하도록 None을 반환합니다."""
    try:
        from pypdf import PdfReader

        # BytesIO를 사용하면 Windows에서 PdfReader가 원본 파일 핸들을 오래
        # 잡고 있어 업로드 임시 파일을 지우지 못하는 문제를 피할 수 있습니다.
        return PdfReader(BytesIO(path.read_bytes()))
    except Exception:  # Parser-specific exceptions vary by PDF structure/version.
        return None


def extract_pypdf_text(reader, page_number: int) -> str:
    """PyPDF로 한 페이지의 텍스트를 추출합니다."""
    if reader is None:
        return ""
    try:
        return _clean_text(reader.pages[page_number - 1].extract_text() or "")
    except Exception:  # A malformed page must fall through to the next parser.
        return ""


def extract_pymupdf_blocks(doc: pymupdf.Document, page_number: int) -> str:
    blocks = doc[page_number - 1].get_text("blocks", sort=True)
    parts: list[str] = []
    for block in blocks:
        if len(block) < 5:
            continue
        txt = block[4]
        if not txt or not txt.strip():
            continue
        # 헤더/본문 순서를 최대한 유지
        parts.append(txt.strip())
    return _clean_text("\n".join(parts))


def render_page(doc: pymupdf.Document, page_number: int, dpi: int = 180) -> bytes:
    page = doc[page_number - 1]
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    return pix.tobytes("png")


@lru_cache(maxsize=1)
def _get_ocr_engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "RapidOCR가 설치되어 있지 않습니다. requirements.txt를 설치하세요."
        ) from exc
    return RapidOCR()


def rapidocr_text(image_bytes: bytes) -> tuple[str, float]:
    """이미지에서 OCR 텍스트를 추출합니다. 엔진은 한 번만 생성해 재사용합니다."""
    from PIL import Image
    import numpy as np

    image = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))

    result, _ = _get_ocr_engine()(image)
    if not result:
        return "", 0.0

    texts: list[str] = []
    scores: list[float] = []
    for row in result:
        # [box, text, score] 형태
        if len(row) >= 3:
            texts.append(str(row[1]))
            try:
                scores.append(float(row[2]))
            except (TypeError, ValueError):
                pass
    return _clean_text("\n".join(texts)), (sum(scores) / len(scores) if scores else 0.0)


@lru_cache(maxsize=1)
def _get_korean_ocr_engine():
    try:
        import easyocr
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("EasyOCR가 설치되어 있지 않습니다. requirements.txt를 설치하세요.") from exc
    import torch

    use_gpu = False
    if torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
        # EasyOCR와 페이지 텐서가 사용할 안전 여유를 확보합니다. Ollama가 이미
        # VRAM을 점유한 경우 CPU로 내려가 시스템 전체의 메모리 압박을 피합니다.
        use_gpu = free_bytes >= 3 * 1024**3
    return easyocr.Reader(["ko", "en"], gpu=use_gpu, verbose=False)


def release_ocr_engines() -> None:
    """OCR 뒤 GPU 메모리를 Ollama에 돌려줍니다."""
    _get_korean_ocr_engine.cache_clear()
    _get_ocr_engine.cache_clear()
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def easyocr_korean_text(image_bytes: bytes) -> tuple[str, float]:
    """RapidOCR가 한글을 다른 문자로 오인할 때 한국어 모델로 다시 읽습니다."""
    from PIL import Image
    import numpy as np

    image = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
    result = _get_korean_ocr_engine().readtext(image, detail=1, paragraph=False)
    if not result:
        return "", 0.0
    ordered = _order_korean_record_ocr(result, image)
    texts, scores = [], []
    for row in ordered:
        if len(row) >= 3:
            texts.append(str(row[1]))
            try:
                scores.append(float(row[2]))
            except (TypeError, ValueError):
                pass
    return _clean_text("\n".join(texts)), (sum(scores) / len(scores) if scores else 0.0)


_RECORD_SECTION_ALIASES = {
    "세부능력특기사항": ("세부능력및특기사항", "세부능력특기사항"),
    "자율자치활동": ("자율자치활동", "자율활동"),
    "동아리활동": ("동아리활동",),
    "진로활동": ("진로활동",),
}


def locate_student_record_pages(
    pdf_path: str | Path,
    record_section: str,
    *,
    dpi: int = 110,
) -> tuple[int, ...] | None:
    """작은 왼쪽 열만 OCR해 선택한 생기부 영역이 있는 페이지를 찾습니다."""
    aliases = _RECORD_SECTION_ALIASES.get(record_section)
    if not aliases:
        return None
    document = pymupdf.open(pdf_path)
    headings_by_page: dict[int, set[str]] = {}
    reader = _get_korean_ocr_engine()
    try:
        from PIL import Image
        import numpy as np

        for page_number, page in enumerate(document, 1):
            # 표의 '영역' 열과 활동 제목은 페이지 왼쪽에 있습니다. 본문 전체를
            # 저해상도로 인식하지 않아 페이지 탐색 비용과 잘못된 본문 매칭을 줄입니다.
            clip = pymupdf.Rect(0, 0, page.rect.width * 0.42, page.rect.height)
            scale = dpi / 72.0
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(scale, scale), clip=clip, alpha=False
            )
            image = np.array(Image.open(BytesIO(pixmap.tobytes("png"))).convert("RGB"))
            lines = reader.readtext(
                image,
                detail=0,
                paragraph=False,
                canvas_size=1280,
                mag_ratio=1.0,
            )
            compact_lines = [re.sub(r"\s+", "", str(line)) for line in lines]
            found: set[str] = set()
            for section, section_aliases in _RECORD_SECTION_ALIASES.items():
                if any(
                    alias in line
                    or (
                        2 <= len(line) <= len(alias) + 3
                        and SequenceMatcher(None, alias, line).ratio() >= 0.68
                    )
                    for alias in section_aliases
                    for line in compact_lines
                ):
                    found.add(section)
            headings_by_page[page_number] = found
    finally:
        document.close()

    target_pages = [
        page for page, headings in headings_by_page.items() if record_section in headings
    ]
    if not target_pages:
        return None
    start = min(target_pages)
    later_section_pages = [
        page
        for page, headings in headings_by_page.items()
        if page > start and any(section != record_section for section in headings)
    ]
    end = min(later_section_pages) - 1 if later_section_pages else len(headings_by_page)
    return tuple(range(start, max(start, end) + 1))


def _horizontal_table_lines(image) -> list[int]:
    """페이지 폭을 길게 가로지르는 표 선의 y좌표를 찾습니다."""
    import numpy as np

    grayscale = image.mean(axis=2) if image.ndim == 3 else image
    dark_counts = (grayscale < 120).sum(axis=1)
    candidates = np.flatnonzero(dark_counts >= image.shape[1] * 0.45).tolist()
    groups: list[list[int]] = []
    for y in candidates:
        if not groups or y > groups[-1][-1] + 1:
            groups.append([y])
        else:
            groups[-1].append(y)
    return [round(sum(group) / len(group)) for group in groups]


def _order_korean_record_ocr(result: list, image) -> list:
    """표 가운데 검출된 활동 영역명을 해당 표 행의 시작 위치로 옮깁니다."""
    section_names = {"자율활동", "자율자치활동", "동아리활동", "진로활동"}
    horizontal_lines = _horizontal_table_lines(image)
    sortable = []
    for original_index, row in enumerate(result):
        if len(row) < 2 or not row[0]:
            continue
        box = row[0]
        top = min(float(point[1]) for point in box)
        left = min(float(point[0]) for point in box)
        center_y = sum(float(point[1]) for point in box) / len(box)
        compact = re.sub(r"\s+", "", str(row[1]))
        sort_y = top
        if compact in section_names:
            upper_lines = [line for line in horizontal_lines if line < center_y]
            lower_lines = [line for line in horizontal_lines if line > center_y]
            if upper_lines and lower_lines:
                sort_y = max(upper_lines) + 0.01
                left = -1.0
        sortable.append((sort_y, left, original_index, row))
    return [row for _, _, _, row in sorted(sortable)]


def _is_korean_document(document_type: str) -> bool:
    """Return whether the Korean OCR engine should be preferred."""
    return (
        document_type in {"student_record", "school_record_guide_2026"}
        or bool(re.search(r"[\uac00-\ud7a3]", document_type))
    )


def _with_ocr_confidence(report: QualityReport, confidence: float) -> QualityReport:
    """Use OCR confidence only as a small quality-score tie breaker."""
    report.score = min(1.0, report.score + min(0.08, max(0.0, confidence) * 0.08))
    return report


def process_pdf(
    pdf_path: str | Path,
    *,
    document_type: str,
    expected_terms_by_page: Optional[dict[int, list[str]]] = None,
    config: Optional[PipelineConfig] = None,
    vision_hook: Optional[VISION_HOOK] = None,
    page_numbers: Optional[Iterable[int]] = None,
) -> list[PageResult]:
    """Extract every selected page using parser and OCR fallbacks."""
    config = config or PipelineConfig()
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(path)

    doc = pymupdf.open(path)
    selected_pages = sorted(set(page_numbers or range(1, doc.page_count + 1)))
    invalid_pages = [page for page in selected_pages if page < 1 or page > doc.page_count]
    if invalid_pages:
        doc.close()
        raise ValueError(f"PDF page is outside the document range: {invalid_pages}")

    first_page = min(selected_pages)
    last_page = max(selected_pages)
    pdftotext_pages = (
        [] if config.prefer_pymupdf
        else _run_pdftotext(path, first_page, last_page)
    )
    pypdf_reader = None if config.prefer_pymupdf else open_pypdf(path)
    results: list[PageResult] = []

    for pno in selected_pages:
        expected_terms = (expected_terms_by_page or {}).get(pno, [])
        pdftotext_index = pno - first_page
        text = _clean_text(
            pdftotext_pages[pdftotext_index]
            if pdftotext_index < len(pdftotext_pages) else ""
        )
        q = quality_check(text, expected_terms=expected_terms, min_chars=config.min_chars)
        method = "pdftotext-layout"

        # Text PDFs can expose different font maps to different parsers.
        if not q.ok and not config.prefer_pymupdf:
            candidate = extract_pypdf_text(pypdf_reader, pno)
            cq = quality_check(candidate, expected_terms=expected_terms, min_chars=config.min_chars)
            if cq.score > q.score:
                text, q, method = candidate, cq, "pypdf-text"

        if not q.ok:
            candidate = extract_pymupdf_text(doc, pno, sort=config.pymupdf_sort)
            cq = quality_check(candidate, expected_terms=expected_terms, min_chars=config.min_chars)
            if cq.score > q.score:
                text, q, method = candidate, cq, "pymupdf-text"

        if not q.ok:
            candidate = extract_pymupdf_blocks(doc, pno)
            cq = quality_check(candidate, expected_terms=expected_terms, min_chars=config.min_chars)
            if cq.score > q.score:
                text, q, method = candidate, cq, "pymupdf-blocks"

        image_bytes: bytes | None = None
        korean_document = _is_korean_document(document_type)

        # Prefer EasyOCR's ko+en model for scanned Korean documents.
        if not q.ok and config.enable_ocr and config.enable_korean_ocr and korean_document:
            image_bytes = render_page(doc, pno, config.dpi)
            try:
                candidate, ocr_score = easyocr_korean_text(image_bytes)
                cq = _with_ocr_confidence(
                    quality_check(candidate, expected_terms=expected_terms, min_chars=40),
                    ocr_score,
                )
                if cq.korean_ratio > q.korean_ratio + 0.10 or cq.score > q.score:
                    text, q, method = candidate, cq, "easyocr-korean"
            except RuntimeError:
                pass

        # Use RapidOCR only when the prior result is still insufficient.
        if not q.ok and config.enable_ocr:
            image_bytes = image_bytes or render_page(doc, pno, config.dpi)
            try:
                candidate, ocr_score = rapidocr_text(image_bytes)
                cq = _with_ocr_confidence(
                    quality_check(candidate, expected_terms=expected_terms, min_chars=40),
                    ocr_score,
                )
                if cq.score > q.score:
                    text, q, method = candidate, cq, "rapidocr"
            except RuntimeError:
                pass

        if not q.ok and vision_hook:
            image_bytes = image_bytes or render_page(doc, pno, config.dpi)
            metadata = {
                "document_type": document_type,
                "page_number": pno,
                "printed_page": detect_printed_page(text, pno),
                "extraction_method": method,
                "quality_score": q.score,
            }
            candidate = vision_hook(image_bytes, metadata)
            cq = quality_check(candidate, expected_terms=expected_terms, min_chars=40)
            if cq.score >= q.score:
                text, q, method = _clean_text(candidate), cq, "vision"

        printed_page = detect_printed_page(text, pno)
        results.append(
            PageResult(
                page_number=pno,
                printed_page=printed_page,
                text=text,
                method=method,
                quality=q,
                metadata={
                    "document_type": document_type,
                    "source": str(path.resolve()),
                    "page": pno - 1,
                    "page_label": str(printed_page or pno),
                    "source_file": path.name,
                    "page_number": pno,
                    "printed_page": printed_page,
                    "extraction_method": method,
                    "quality_score": round(q.score, 4),
                    "needs_review": not q.ok,
                },
            )
        )

    doc.close()
    return results

def detect_printed_page(text: str, fallback: Optional[int] = None) -> Optional[int]:
    """본문의 '70' 같은 인쇄 페이지 번호를 찾을 때 쓰는 보수적 휴리스틱."""
    # 대학 모집요강은 인쇄 쪽수를 머리말·꼬리말의 시작 또는 끝에 두는 경우가 많습니다.
    # 장·절 번호와 혼동하지 않도록 PDF 물리 페이지와 가까운 숫자만 채택합니다.
    if fallback is not None:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        candidates = []
        for line in lines[:3] + lines[-3:]:
            for pattern in (r"^(\d{1,3})\b", r"\b(\d{1,3})$"):
                match = re.search(pattern, line)
                if match:
                    value = int(match.group(1))
                    if value > 0 and abs(value - fallback) <= 10:
                        candidates.append(value)
        if candidates:
            return min(candidates, key=lambda value: abs(value - fallback))
    patterns = [
        r"수시모집요(?:강|⊙).*?\s(\d{1,3})\b",
        r"학교생활기록부.*?(?:p\.?|페이지)\s*(\d{1,3})",
    ]
    for p in patterns:
        m = re.search(p, text, re.S | re.I)
        if m:
            return int(m.group(1))
    return fallback


def page_results_to_documents(results: Iterable[PageResult]):
    return [
        Document(page_content=result.text, metadata=result.metadata)
        for result in results
        if result.text.strip()
    ]


def targeted_pages_for_document(document_type: str) -> dict[int, list[str]]:
    """이전 호출부 호환용. 페이지 선택은 config.py에서 관리합니다."""
    return {}


def summarize_page_failures(results: Iterable[PageResult]) -> dict:
    results = list(results)
    return {
        "total_pages": len(results),
        "ok_pages": sum(r.quality.ok for r in results),
        "review_pages": sum(not r.quality.ok for r in results),
        "methods": {
            m: sum(r.method == m for r in results)
            for m in [
                "pdftotext-layout",
                "pypdf-text",
                "pymupdf-text",
                "pymupdf-blocks",
                "easyocr-korean",
                "rapidocr",
                "vision",
            ]
        },
        "worst_pages": [
            {"page": r.page_number, "score": round(r.quality.score, 3), "method": r.method}
            for r in sorted(results, key=lambda x: x.quality.score)[:10]
        ],
    }


def build_source_documents(
    pdf_path: str | Path,
    *,
    document_type: str,
    expected_terms_by_page: Optional[dict[int, list[str]]] = None,
    vision_hook: Optional[VISION_HOOK] = None,
    config: Optional[PipelineConfig] = None,
    page_numbers: Optional[Iterable[int]] = None,
):
    """현재 retriever.py에서 바로 사용할 수 있는 LangChain Documents 생성."""
    results = process_pdf(
        pdf_path,
        document_type=document_type,
        expected_terms_by_page=expected_terms_by_page,
        vision_hook=vision_hook,
        config=config,
        page_numbers=page_numbers,
    )
    return page_results_to_documents(results), summarize_page_failures(results)
