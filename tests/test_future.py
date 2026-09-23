import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest
from config import PROJECT_ROOT
from ingestion.prepare_documents import annotate_curriculum_documents, normalize_course_key
from rag.future import (
    ActivityStep,
    CompactFutureActivity,
    _naturalize_evidence_references,
    _fallback_connection_reason,
    _to_advisory_style,
    generate_future_guide,
    next_semester,
)
from rag.retriever import retrieve_curriculum_context


class FutureTests(unittest.TestCase):
    def test_connection_reason_uses_weight_basis_and_reason_without_raw_excerpt(self):
        criterion = SimpleNamespace(area="탐구역량", weight="40%")
        activity = {"title": "디지털 미디어 리터러시 심화 탐구", "past_evidence_numbers": [1]}
        matches = [{"experience_title": "기초연기 활동 경험"}]
        result = _fallback_connection_reason("성균관대학교", criterion, activity, matches)
        self.assertIn("희망 대학인 성균관대학교의 탐구역량 반영 비율 40%", result)
        self.assertIn("기존 생기부에서 확인한 기초연기 활동 경험을 기반으로", result)
        self.assertIn("필요가 있기 때문에", result)
        self.assertTrue(result.endswith("확장해 보는 것을 추천드립니다."))
        self.assertNotIn("…", result)

    def test_internal_evidence_numbers_are_naturalized(self):
        text = (
            "과거 근거 1에서 확인한 미디어 탐색과 근거 2에서 수행한 진로검사를 "
            "현재 탐구로 확장하는 방향이 좋습니다."
        )
        result = _naturalize_evidence_references(text)
        self.assertNotIn("근거 1", result)
        self.assertNotIn("근거 2", result)
        self.assertIn("기존 생기부에서는", result)

    def test_malformed_evidence_reference_and_hanja_are_cleaned(self):
        result = _naturalize_evidence_references(
            "기존 생기부 경험와 3 의 민주적 의사결정 및 스크립트朗誦 경험을 연결합니다."
        )
        self.assertEqual(
            result,
            "기존 생기부의 민주적 의사결정 및 스크립트낭송 경험을 연결합니다.",
        )
        self.assertNotRegex(result, r"[\u3400-\u9fff]")

    def test_future_plan_declarative_tone_is_changed_to_advice(self):
        result = _to_advisory_style("자료를 수집합니다. 분석의 정확도를 높입니다.")
        self.assertNotIn("수집합니다", result)
        self.assertNotIn("높입니다", result)
        self.assertIn("수집하는 것을 추천드립니다", result)
        self.assertIn("정확도를 높이는 것을 권합니다", result)

    def build_result(self, section="세부능력특기사항", subject="확률과 통계"):
        model = Mock()
        plan = CompactFutureActivity(title="다음 학기 활동",
            goal="분석 방법을 보완해 보는 것을 권합니다.",
            steps=[ActivityStep(action="동료와 자료를 비교하고 분석 방법을 정리해 보는 것을 추천드립니다.", output="비교표") for _ in range(3)])
        def structured(schema, **kwargs):
            model.invoke.return_value = schema(**{name: plan for name in schema.model_fields})
            return model
        model.with_structured_output.side_effect = structured
        document = Document(
            page_content="관련 없는 전형 안내\n학업역량(40%)\n탐구역량(40%)\n잠재역량(20%)",
            metadata={"source": "college.pdf", "page": 2},
        )
        curriculum_document = Document(
            page_content="확률과 통계에서 자료를 수집·정리하고 결과를 분석하는 통계적 과정을 학습한다.",
            metadata={"source": "교육과정.pdf", "page": 45, "course_name": "확률과통계", "course_key": "확률과통계"},
        )
        with patch("rag.future.ChatOllama", return_value=model), patch("rag.future.load_college_vectorstore"), patch(
            "rag.future.retrieve_college_context", return_value=[document]
        ), patch("rag.future.load_curriculum_vectorstore"), patch(
            "rag.future.retrieve_curriculum_context", return_value=[curriculum_document]
        ), patch("rag.future.prepare_record_context", return_value=("기존 활동 근거", [{
            "experience_title": "자료 비교·분석 경험",
            "original": "기존 활동\n원문", "connection_reason": "초안의 분석 방법과 관련됨",
            "similarity_score": 0.8, "llm_relevance": 3, "text_integrity": 2,
        }])) as context:
            result = generate_future_guide("초안", "성균관대학교", "통계학과", current_grade=2,
                current_semester=2, section_type=section, subject=subject, previous_record="기록")
        return result, model, context

    def test_next_semester(self):
        self.assertEqual(next_semester(1, 1), "1학년 2학기")
        self.assertEqual(next_semester(2, 2), "3학년 1학기")
        for grade, semester in [(3, 1), (3, 2), (0, 1), (1, 3)]:
            with self.assertRaises(ValueError):
                next_semester(grade, semester)

    def test_first_year_first_semester_uses_only_current_draft(self):
        model = Mock()
        plan = CompactFutureActivity(
            title="다음 학기 활동",
            goal="탐구 과정을 보완해 보는 것을 권합니다.",
            steps=[ActivityStep(action="수업 자료를 비교하고 탐구 과정을 단계별로 기록해 보는 것을 추천드립니다.", output="탐구 기록") for _ in range(3)],
        )

        def structured(schema, **kwargs):
            model.invoke.return_value = schema(**{name: plan for name in schema.model_fields})
            return model

        model.with_structured_output.side_effect = structured
        document = Document(
            page_content="학업역량(40%)\n탐구역량(40%)\n잠재역량(20%)",
            metadata={"source": "college.pdf", "page": 2},
        )
        curriculum_document = Document(
            page_content="수학 교과에서 자료를 분석하고 탐구 과정을 기록한다.",
            metadata={"source": "교육과정.pdf", "page": 30, "course_name": "공통수학1"},
        )
        with patch("rag.future.ChatOllama", return_value=model), patch(
            "rag.future.load_college_vectorstore"
        ), patch("rag.future.retrieve_college_context", return_value=[document]), patch(
            "rag.future.load_curriculum_vectorstore"
        ), patch("rag.future.retrieve_curriculum_context", return_value=[curriculum_document]), patch(
            "rag.future.prepare_record_context"
        ) as context:
            result = generate_future_guide(
                "현재 자기평가 경험", "성균관대학교", "통계학과",
                current_grade=1, current_semester=1, section_type="세부능력특기사항", subject="수학",
            )

        context.assert_not_called()
        self.assertFalse(result["uses_previous_record"])
        self.assertEqual(result["record_context"], "")
        self.assertEqual(result["record_matches"], [])
        prompt = str(model.invoke.call_args.args[0])
        self.assertIn("[현재 경험: 자기평가보고서]", prompt)
        self.assertIn("기존 생기부가 없으므로", prompt)
        self.assertNotIn("[과거 경험: 기존 생기부 관련 구간]", prompt)

    def test_subject_and_target_reach_single_generation(self):
        result, model, context = self.build_result()
        self.assertEqual(model.invoke.call_count, 1)
        prompt = str(model.invoke.call_args.args[0])
        for text in ["확률과 통계", "3학년 1학기", "세부능력특기사항", "통계학과", "기존 활동 근거", "현재 교육과정 참고 자료", "자료를 수집·정리", "성취기준, 단원", "~해 보는 것을 추천드립니다", "과거·현재 경험의 주제나 방법", "평가영역마다 다른 초점", "기존 활동과 무관한 활동을 처음부터 새로 제시하지 마라"]:
            self.assertIn(text, prompt)
        self.assertEqual(len(result["future_activities"]), 3)
        self.assertEqual([a["weight"] for a in result["future_activities"]], ["40%", "40%", "20%"])
        self.assertNotIn("college_evidence", result)
        self.assertEqual(context.call_args.kwargs["max_selected_chunks"], 3)
        self.assertTrue(context.call_args.kwargs["return_matches"])
        self.assertEqual(context.call_args.kwargs["layout_noise_terms"], ["확률과 통계"])
        self.assertEqual(context.call_args.kwargs["record_section"], "세부능력특기사항")
        self.assertEqual(context.call_args.kwargs["subject"], "확률과 통계")
        self.assertGreaterEqual(len(context.call_args.kwargs["retrieval_queries"]), 4)
        self.assertEqual(result["future_activities"][0]["past_evidence_numbers"], [1])

    def test_curriculum_pages_are_tagged_as_one_course_card(self):
        documents = [
            Document(page_content="과목명\n확률과통계\n이 과목은 어떤 과목인가요?", metadata={"page": 45}),
            Document(page_content="자료를 수집하고 분석하는 활동", metadata={"page": 46}),
            Document(page_content="관련 학과 전체 안내", metadata={"page": 47}),
        ]
        result = annotate_curriculum_documents(documents)
        self.assertEqual(normalize_course_key("확률과 통계"), "확률과통계")
        self.assertEqual(result[0].metadata["course_key"], "확률과통계")
        self.assertEqual(result[1].metadata["course_key"], "확률과통계")
        self.assertNotIn("course_key", result[2].metadata)

    def test_exact_subject_retrieval_wins_over_semantic_search(self):
        store = Mock()
        store.get.return_value = {
            "documents": ["두 번째 청크", "첫 번째 청크"],
            "metadatas": [
                {"page": 46, "start_index": 0, "course_name": "확률과통계"},
                {"page": 45, "start_index": 0, "course_name": "확률과통계"},
            ],
        }
        result = retrieve_curriculum_context(
            store, subject="확률과 통계", department="통계학과", section_type="세부능력특기사항",
            target_semester="2학년 1학기", student_draft="자료를 분석함",
        )
        self.assertEqual([item.page_content for item in result], ["첫 번째 청크", "두 번째 청크"])
        store.similarity_search.assert_not_called()
        self.assertEqual(store.get.call_args.kwargs["where"], {"course_key": "확률과통계"})

    def test_activity_without_subject_uses_department_and_draft_in_one_query(self):
        store = Mock()
        store.get.return_value = {"documents": [], "metadatas": []}
        store.similarity_search.return_value = [
            Document(page_content="매체 표현을 비판적으로 분석함", metadata={"page": 29, "course_name": "매체의사소통"}),
            Document(page_content="같은 과목의 후속 설명", metadata={"page": 30, "course_name": "매체의사소통"}),
            Document(page_content="같은 과목의 중복 청크", metadata={"page": 30, "course_name": "매체의사소통"}),
            Document(page_content="일반 목차", metadata={"page": 3}),
            Document(page_content="언어 자료의 표현 효과를 탐구함", metadata={"page": 31, "course_name": "언어생활탐구"}),
        ]
        result = retrieve_curriculum_context(
            store, subject="", department="미디어커뮤니케이션학과", section_type="동아리활동",
            target_semester="2학년 1학기", student_draft="뉴스 매체의 표현 방식을 비교하고 토론함",
        )
        self.assertEqual([item.metadata["course_name"] for item in result], [
            "매체의사소통", "매체의사소통", "언어생활탐구",
        ])
        query = store.similarity_search.call_args.args[0]
        self.assertIn("미디어커뮤니케이션학과", query)
        self.assertIn("뉴스 매체의 표현 방식", query)

    def test_ui_inputs_results_and_invalidation(self):
        result, _, _ = self.build_result()
        uploaded = SimpleNamespace(name="record.txt", getvalue=lambda: "기록".encode())
        with patch("rag.future.generate_future_guide", return_value=result) as generate:
            app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run()
            self.assertFalse(app.tabs)
            app.button[0].click().run()
            generate.assert_not_called()
            with patch("streamlit.file_uploader", return_value=uploaded):
                app.text_input(key="guide_department").set_value("통계학과")
                app.text_input(key="guide_subject").set_value("확률과 통계")
                app.text_area[0].set_value("초안")
                app.selectbox(key="guide_grade").set_value(2)
                app.selectbox(key="guide_semester").set_value(2)
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertEqual([x.value for x in app.subheader], ["1. 활동 가이드", "2. 근거"])
                rendered_values = [
                    str(element.value)
                    for collection in (app.markdown, app.text, app.caption)
                    for element in collection
                ]
                self.assertFalse(any("관련 없는 전형 안내" in value for value in rendered_values))
                self.assertTrue(any("자료 비교·분석 경험" in item.value for item in app.markdown))
                self.assertTrue(any("기존 생기부 원문" in item.value for item in app.markdown))
                self.assertTrue(any("기존 활동 원문" in item.value for item in app.markdown))
                self.assertFalse(any("과거 · 기존 생기부" in item.value for item in app.markdown))
                self.assertTrue(any("현재 · 자기평가보고서" in item.value for item in app.markdown))
                self.assertTrue(any("추천 이유 · 대학 평가 기준과 경험 연결" in item.value for item in app.markdown))
                markdown_values = [item.value for item in app.markdown]
                record_block = next(
                    value for value in markdown_values
                    if "기존 생기부 원문" in value and "관련 있다고 판단한 이유" in value
                )
                self.assertLess(record_block.index("기존 생기부 원문"), record_block.index("관련 있다고 판단한 이유"))
                self.assertEqual(generate.call_args.kwargs["current_grade"], 2)
                app.button[0].click().run()
                self.assertEqual(generate.call_count, 1)
                self.assertNotIn("봉사활동", app.selectbox(key="guide_section_type").options)
                app.selectbox(key="guide_section_type").set_value("동아리활동").run()
                self.assertFalse(app.subheader)
                self.assertNotIn("guide_subject", [x.key for x in app.text_input])
                self.assertEqual(app.selectbox(key="guide_grade").options, ["1학년", "2학년"])

    def test_first_year_first_semester_hides_record_upload(self):
        with patch("streamlit.file_uploader") as uploader:
            app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run()
        self.assertFalse(app.exception)
        uploader.assert_not_called()
        self.assertTrue(any("기존 생기부 없이" in item.value for item in app.caption))


if __name__ == "__main__":
    unittest.main()
