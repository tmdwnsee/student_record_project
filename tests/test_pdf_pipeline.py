"""PDF 단계별 추출 파이프라인의 재현 가능한 단위 테스트."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

from ingestion.pdf_pipeline import _run_pdftotext, page_results_to_documents, process_pdf


class PdfPipelineTests(unittest.TestCase):
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
            self.assertIn(results[0].method, {"pymupdf-text", "pymupdf-blocks"})

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


if __name__ == "__main__":
    unittest.main()
