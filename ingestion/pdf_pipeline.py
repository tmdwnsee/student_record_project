"""RAG용 PDF 단계별 추출 파이프라인.

순서:
1) pdftotext -layout
2) 텍스트 품질 검사
3) PyMuPDF text
4) PyMuPDF blocks
5) RapidOCR
6) Vision hook (선택)

문서별로 페이지 단위 Document를 만든 뒤, 페이지 메타데이터를 보존한 상태로
후속 chunking/retrieval에 넘기는 것을 목표로 합니다.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
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


def process_pdf(
    pdf_path: str | Path,
    *,
    document_type: str,
    expected_terms_by_page: Optional[dict[int, list[str]]] = None,
    config: Optional[PipelineConfig] = None,
    vision_hook: Optional[VISION_HOOK] = None,
    page_numbers: Optional[Iterable[int]] = None,
) -> list[PageResult]:
    """PDF 전체를 페이지별로 단계적 추출.

    Vision hook은 RapidOCR까지도 품질이 낮은 페이지에서만 호출됩니다.
    hook signature: hook(image_bytes, metadata) -> text
    """
    config = config or PipelineConfig()
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(path)

    doc = pymupdf.open(path)
    selected_pages = sorted(set(page_numbers or range(1, doc.page_count + 1)))
    invalid_pages = [page for page in selected_pages if page < 1 or page > doc.page_count]
    if invalid_pages:
        doc.close()
        raise ValueError(f"PDF 범위를 벗어난 페이지입니다: {invalid_pages}")
    first_page = min(selected_pages)
    last_page = max(selected_pages)
    pdftotext_pages = (
        [] if config.prefer_pymupdf
        else _run_pdftotext(path, first_page, last_page)
    )
    results: list[PageResult] = []

    for pno in selected_pages:
        expected_terms = (expected_terms_by_page or {}).get(pno, [])

        # 1) pdftotext -layout
        pdftotext_index = pno - first_page
        text = _clean_text(
            pdftotext_pages[pdftotext_index]
            if pdftotext_index < len(pdftotext_pages) else ""
        )
        q = quality_check(text, expected_terms=expected_terms, min_chars=config.min_chars)
        method = "pdftotext-layout"

        # 2) PyMuPDF text
        if not q.ok:
            candidate = extract_pymupdf_text(doc, pno, sort=config.pymupdf_sort)
            cq = quality_check(candidate, expected_terms=expected_terms, min_chars=config.min_chars)
            if cq.score > q.score:
                text, q, method = candidate, cq, "pymupdf-text"

        # 3) PyMuPDF blocks
        if not q.ok:
            candidate = extract_pymupdf_blocks(doc, pno)
            cq = quality_check(candidate, expected_terms=expected_terms, min_chars=config.min_chars)
            if cq.score > q.score:
                text, q, method = candidate, cq, "pymupdf-blocks"

        # 4) RapidOCR
        if not q.ok and config.enable_ocr:
            image_bytes = render_page(doc, pno, config.dpi)
            try:
                candidate, ocr_score = rapidocr_text(image_bytes)
                cq = quality_check(candidate, expected_terms=expected_terms, min_chars=40)
                # OCR 자체 confidence를 품질 점수의 보조 신호로 사용
                cq.score = max(cq.score, min(1.0, ocr_score))
                if cq.score > q.score:
                    text, q, method = candidate, cq, "rapidocr"
            except RuntimeError:
                # 프로덕션에서는 로그를 남기고 Vision hook으로 이어질 수 있게 둠
                pass

        # 5) Vision 연결 지점
        if not q.ok and vision_hook:
            image_bytes = render_page(doc, pno, config.dpi)
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
            for m in ["pdftotext-layout", "pymupdf-text", "pymupdf-blocks", "rapidocr", "vision"]
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
