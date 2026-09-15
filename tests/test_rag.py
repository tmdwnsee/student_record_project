"""API 비용 없이 중복 저장, 인용 검증, 화면 동작을 확인합니다."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from streamlit.testing.v1 import AppTest

from config import PROJECT_ROOT
from rag.attachment import ChunkAssessment, extract_context, prepare_record_context, split_record
from rag.chain import (
    Evidence, ReviewResult, SAMPLE_DRAFT, _conservative_rewrite, _looks_like_record_sentence,
    extract_college_criteria, review_draft, select_evidence, validate_evidence,
)
from ingestion.build_index import sync_vectorstore


class CountingEmbeddings(Embeddings):
    def __init__(self):
        self.embedded_count = 0

    def embed_documents(self, texts):
        self.embedded_count += len(texts)
        return [[float(len(text)), 1.0, 0.5] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0, 0.5]


class RagTests(unittest.TestCase):
    def test_revised_text_rejects_guidance_and_evidence(self):
        self.assertTrue(_looks_like_record_sentence("Python으로 데이터를 분석하고 문제 해결 과정을 수행함."))
        for text in (
            "탐구역량이 드러나도록 작성할 것.",
            "대학 평가기준과 반영 비율을 고려해야 함.",
            "student_record_rule.pdf 근거를 참고함.",
        ):
            with self.subTest(text=text):
                self.assertFalse(_looks_like_record_sentence(text))
        self.assertFalse(_looks_like_record_sentence(
            "Python으로 데이터를 수집하고 시각화하여 논리적 사고력을 기름.",
            "Python으로 데이터를 분석함.",
        ))
        self.assertEqual(
            _conservative_rewrite("데이터 분석 프로젝트를 진행하며 Python으로 데이터를 분석함."),
            "데이터 분석 프로젝트에서 Python을 활용해 데이터를 분석함",
        )

    def test_previous_record_text_extraction_and_validation(self):
        self.assertEqual(extract_context("record.txt", "기존 활동".encode("utf-8")), "기존 활동")
        for name, content in (("record.txt", b""), ("record.txt", b"\xff"), ("record.docx", b"x")):
            with self.subTest(name=name, content=content), self.assertRaises(ValueError):
                extract_context(name, content)

    def test_uploaded_pdf_uses_shared_pdf_pipeline(self):
        pages = [SimpleNamespace(text="첫 페이지"), SimpleNamespace(text="둘째 페이지")]
        with patch("rag.attachment.process_pdf", return_value=pages) as process:
            text = extract_context("record.pdf", b"fake pdf bytes")
        self.assertEqual(text, "첫 페이지\n둘째 페이지")
        self.assertEqual(process.call_args.kwargs["document_type"], "student_record")

    def test_whole_record_is_read_and_only_relevant_passages_are_selected(self):
        full_text = "앞부분 " + "가" * 6_000 + " 끝부분 활동"
        chunks = split_record(full_text)
        self.assertGreater(len(chunks), 1)
        self.assertIn("끝부분 활동", chunks[-1])

        assessments = [
            ChunkAssessment(activity_summary="관계없는 내용", relevance=0),
            ChunkAssessment(activity_summary="관련 활동", relevance=3),
            ChunkAssessment(activity_summary="다른 활동", relevance=0),
        ]
        with patch("rag.attachment.split_record", return_value=["처음", "중간 원문", "마지막"]), patch(
            "rag.attachment.ChatOllama"
        ) as model_class:
            model = model_class.return_value.with_structured_output.return_value
            model.invoke.side_effect = assessments
            context = prepare_record_context("전체 생기부", "새 초안")
        self.assertEqual(model.invoke.call_count, 3)
        self.assertIn("중간 원문", context)
        self.assertNotIn("처음", context)
        self.assertNotIn("마지막", context)

    def test_previous_record_is_passed_as_context_without_changing_search_query(self):
        with patch("rag.chain.load_vectorstores", return_value=(object(), object())), patch(
            "rag.chain.prepare_record_context", return_value="선택된 맥락"
        ) as prepare, patch("rag.chain.retrieve_college_context", return_value=[]) as college, patch(
            "rag.chain.retrieve_guideline_context", return_value=[]
        ) as guideline, patch("rag.chain.generate_review") as generate:
            review_draft("새 초안", "기존 생기부 내용", "성균관대학교", "소프트웨어학과")
        prepare.assert_called_once_with("기존 생기부 내용", "새 초안")
        self.assertEqual(college.call_args.args[1], "새 초안")
        self.assertEqual(college.call_args.args[2], "성균관대학교")
        self.assertEqual(college.call_args.args[3], "소프트웨어학과")
        self.assertEqual(guideline.call_args.args[1], "새 초안")
        self.assertEqual(generate.call_args.args, ("새 초안", [], [], "선택된 맥락", "성균관대학교", "소프트웨어학과"))

    def test_college_criteria_are_extracted_from_source(self):
        document = Document(
            page_content="학업역량(40%)\n탐구역량(40%)\n잠재역량(20%)\n공동체의식(100점)",
            metadata={"source": "college_table.pdf", "page": 71},
        )
        criteria, evidence_ids = extract_college_criteria(
            [document], "Python으로 데이터를 분석함.", "성균관대학교",
        )
        self.assertEqual(
            [(item.area, item.weight) for item in criteria],
            [("학업역량", "40%"), ("탐구역량", "40%"), ("잠재역량", "20%")],
        )
        self.assertEqual(evidence_ids, [1])
        self.assertNotIn("공동체의식", [item.area for item in criteria])

    def test_persistence_deduplication_and_changed_pdf(self):
        embedding = CountingEmbeddings()
        documents = [Document(page_content="평가 기준", metadata={"source": "college_table.pdf", "page": 71})]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            store = sync_vectorstore(documents, Path(directory), "test_collection", embedding)
            original_ids = store.get(include=[])["ids"]
            reopened = sync_vectorstore(documents, Path(directory), "test_collection", embedding)
            self.assertEqual(reopened.get(include=[])["ids"], original_ids)
            self.assertEqual(embedding.embedded_count, 1)
            changed = [Document(page_content="변경된 평가 기준", metadata=documents[0].metadata)]
            updated = sync_vectorstore(changed, Path(directory), "test_collection", embedding)
            self.assertEqual(len(updated.get(include=[])["ids"]), 1)
            self.assertNotEqual(updated.get(include=[])["ids"], original_ids)
            self.assertEqual(embedding.embedded_count, 2)

    def test_evidence_must_match_retrieved_source_page_and_text(self):
        document = Document(page_content="교사가 직접 관찰한 내용", metadata={"source": "student_record_rule.pdf", "page": 10})
        valid = Evidence(source="student_record_rule.pdf", page=10, content="직접 관찰한 내용")
        validate_evidence([valid], [document])
        for changes in ({"source": "college_table.pdf"}, {"page": 11}, {"content": "없는 근거"}, {"content": " "}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_evidence([valid.model_copy(update=changes)], [document])

    def test_streamlit_empty_input_and_result(self):
        result = ReviewResult(
            original_text=SAMPLE_DRAFT, revised_text="데이터 분석 프로젝트에서 Python으로 데이터를 분석하였다.",
            revision_reason="검토 이유", guideline_evidence=[], college_evidence=[], caution="확인 필요",
        )
        with patch("rag.chain.review_draft", return_value=result) as review:
            app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run()
            self.assertFalse(app.exception)
            app.button[0].click().run()
            self.assertTrue(app.warning)
            review.assert_not_called()
            self.assertEqual(app.selectbox[0].value, "성균관대학교")
            app.text_input[0].set_value("소프트웨어학과")
            app.text_area[0].set_value(SAMPLE_DRAFT)
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(len(app.subheader), 6)
            review.assert_called_once()
            # 초안을 바꾸면 이전 초안의 검토 결과를 표시하지 않습니다.
            app.text_area[0].set_value("수정된 초안").run()
            self.assertEqual(len(app.subheader), 0)

    def test_evidence_selection_rejects_invalid_ids(self):
        document = Document(page_content="평가표 원문", metadata={"source": "college_table.pdf", "page": 71})
        evidence = select_evidence([1, 1], [document])
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].content, document.page_content)
        self.assertEqual(evidence[0].page, 71)
        for index in (0, -1, 2):
            with self.subTest(index=index), self.assertRaises(ValueError):
                select_evidence([index], [document])


if __name__ == "__main__":
    unittest.main()
