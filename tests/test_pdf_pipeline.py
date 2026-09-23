"""PDF 단계별 추출 파이프라인의 재현 가능한 단위 테스트."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

from ingestion.pdf_pipeline import (
    _order_korean_record_ocr,
    _pages_for_record_section,
    _record_sections_in_text,
    _run_pdftotext,
    detect_printed_page,
    page_results_to_documents,
    process_pdf,
)


class PdfPipelineTests(unittest.TestCase):
    def test_activity_phrase_in_body_is_not_a_page_heading(self):
        self.assertEqual(
            _record_sections_in_text("동아시아 시민 진로탐색 동아리 활동을 통해 사회 문제를 파악함."),
            set(),
        )
        self.assertEqual(_record_sections_in_text("동 아 리 활 동"), {"동아리활동"})

    def test_repeated_activity_sections_across_grades_are_all_selected(self):
        headings = {
            1: {"동아리활동"},
            2: set(),
            3: {"진로활동"},
            4: {"동아리활동"},
            5: set(),
            6: {"자율자치활동"},
        }
        self.assertEqual(
            _pages_for_record_section(headings, "동아리활동"),
            (1, 2, 4, 5),
        )

    def test_repeated_selection_is_generic_for_every_activity_type(self):
        headings = {
            1: {"세부능력특기사항"},
            2: {"진로활동"},
            3: {"자율자치활동"},
            4: {"진로활동"},
        }
        self.assertEqual(_pages_for_record_section(headings, "진로활동"), (2, 4))

    def test_activity_label_is_moved_to_start_of_its_table_row(self):
        import numpy as np

        image = np.full((200, 300, 3), 255, dtype=np.uint8)
        image[[20, 100, 180], :, :] = 0
        result = [
            ([[120, 30], [280, 30], [280, 45], [120, 45]], "진로 본문 앞부분", 0.9),
            ([[20, 60], [90, 60], [90, 75], [20, 75]], "진로활동", 0.9),
            ([[120, 80], [280, 80], [280, 95], [120, 95]], "진로 본문 뒷부분", 0.9),
            ([[120, 110], [280, 110], [280, 125], [120, 125]], "자율 본문 앞부분", 0.9),
            ([[20, 140], [90, 140], [90, 155], [20, 155]], "자율활동", 0.9),
        ]
        ordered = _order_korean_record_ocr(result, image)
        self.assertEqual([row[1] for row in ordered], [
            "진로활동", "진로 본문 앞부분", "진로 본문 뒷부분",
            "자율활동", "자율 본문 앞부분",
        ])

    def test_printed_page_is_detected_from_header_near_physical_page(self):
        text = "학생부종합전형 전형요소별 평가 안내    61\n1. 학생부종합전형 서류평가"
        self.assertEqual(detect_printed_page(text, 63), 61)

    def test_missing_pdftotext_falls_back_without_error(self):
        with patch("ingestion.pdf_pipeline.subprocess.run", side_effect=FileNotFoundError):
            self.assertEqual(_run_pdftotext(Path("missing-command.pdf")), [])

    def test_process_pdf_preserves_rag_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "sample.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "Student record activity and evaluation details " * 8)
            document.save(pdf_path)
            document.close()

            with patch("ingestion.pdf_pipeline._run_pdftotext", return_value=[]), patch(
                "ingestion.pdf_pipeline.rapidocr_text", side_effect=RuntimeError
            ):
                results = process_pdf(pdf_path, document_type="test")

            documents = page_results_to_documents(results)
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].metadata["source"], str(pdf_path.resolve()))
            self.assertEqual(documents[0].metadata["page"], 0)
            self.assertIn(results[0].method, {"pypdf-text", "pymupdf-text", "pymupdf-blocks"})

    def test_process_pdf_reads_only_selected_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "three-pages.pdf"
            document = pymupdf.open()
            for index in range(3):
                page = document.new_page()
                page.insert_text((72, 72), f"Page {index + 1} evaluation details " * 10)
            document.save(pdf_path)
            document.close()

            with patch("ingestion.pdf_pipeline._run_pdftotext", return_value=[]):
                results = process_pdf(
                    pdf_path, document_type="test", page_numbers=(2,)
                )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].page_number, 2)
            self.assertEqual(results[0].metadata["page"], 1)

    def test_readable_pdftotext_does_not_run_fallbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "sample.pdf"
            document = pymupdf.open()
            document.new_page()
            document.save(pdf_path)
            document.close()
            readable = "학교생활기록부 활동 내용과 구체적인 평가 기록입니다. " * 10

            with patch("ingestion.pdf_pipeline._run_pdftotext", return_value=[readable]), patch(
                "ingestion.pdf_pipeline.extract_pymupdf_text"
            ) as pymupdf_text, patch("ingestion.pdf_pipeline.rapidocr_text") as ocr:
                results = process_pdf(pdf_path, document_type="test")

            self.assertEqual(results[0].method, "pdftotext-layout")
            pymupdf_text.assert_not_called()
            ocr.assert_not_called()

    def test_pypdf_is_used_before_pymupdf_when_it_recovers_text(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "font-map.pdf"
            document = pymupdf.open()
            document.new_page()
            document.save(pdf_path)
            document.close()
            recovered = "\ud559\uad50\uc0dd\ud65c\uae30\ub85d\ubd80 \ud65c\ub3d9 \ub0b4\uc6a9\uacfc \uad6c\uccb4\uc801\uc778 \ud3c9\uac00 \uae30\ub85d\uc785\ub2c8\ub2e4. " * 10

            with patch("ingestion.pdf_pipeline._run_pdftotext", return_value=[]), patch(
                "ingestion.pdf_pipeline.extract_pypdf_text", return_value=recovered
            ), patch("ingestion.pdf_pipeline.extract_pymupdf_text") as pymupdf_text:
                results = process_pdf(pdf_path, document_type="student_record")

            self.assertEqual(results[0].method, "pypdf-text")
            pymupdf_text.assert_not_called()

    def test_korean_record_uses_easyocr_before_rapidocr(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "scanned-record.pdf"
            document = pymupdf.open()
            document.new_page()
            document.save(pdf_path)
            document.close()
            korean = "\ucc3d\uc758\uc801 \uccb4\ud5d8\ud65c\ub3d9 \uc0c1\ud669 \uc9c4\ub85c\ud65c\ub3d9 \ubbf8\ub514\uc5b4 \uae30\ud68d\uc790\uc758 \uc724\ub9ac\uc640 \ud0dc\ub3c4\ub97c \ud0d0\uad6c\ud568. " * 5
            with patch("ingestion.pdf_pipeline._run_pdftotext", return_value=[]), patch(
                "ingestion.pdf_pipeline.rapidocr_text"
            ) as rapidocr, patch(
                "ingestion.pdf_pipeline.easyocr_korean_text", return_value=(korean, 0.92)
            ):
                results = process_pdf(pdf_path, document_type="student_record")

            self.assertEqual(results[0].method, "easyocr-korean")
            self.assertIn("\uc9c4\ub85c\ud65c\ub3d9", results[0].text)
            rapidocr.assert_not_called()



if __name__ == "__main__":
    unittest.main()
