import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest
from config import PROJECT_ROOT
from rag.future import ActivityStep, FutureActivity, generate_future_guide, next_semester


class FutureTests(unittest.TestCase):
    def build_result(self, section="세부능력특기사항", subject="확률과 통계"):
        model = Mock()
        plan = FutureActivity(title="다음 학기 활동", rationale="초안의 경험을 확장", goal="분석 방법 보완",
            department_connection="학과의 학습 분야와 연결 제안", success_check="비교 방법과 한계 기록",
            steps=[ActivityStep(action="동료와 자료를 비교하고 분석 방법을 정리한다.", output="비교표") for _ in range(3)])
        def structured(schema, **kwargs):
            model.invoke.return_value = schema(**{name: plan for name in schema.model_fields})
            return model
        model.with_structured_output.side_effect = structured
        document = Document(page_content="학업역량(40%)\n탐구역량(40%)\n잠재역량(20%)", metadata={"source": "college.pdf", "page": 2})
        with patch("rag.future.ChatOllama", return_value=model), patch("rag.future.load_college_vectorstore"), patch(
            "rag.future.retrieve_college_context", return_value=[document]
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
        plan = FutureActivity(
            title="다음 학기 활동", rationale="자기평가보고서의 현재 경험을 확장", goal="탐구 과정 보완",
            department_connection="학과 분야와 연결", success_check="탐구 과정과 결과물 확인",
            steps=[ActivityStep(action="수업 자료를 비교하고 탐구 과정을 단계별로 기록한다.", output="탐구 기록") for _ in range(3)],
        )

        def structured(schema, **kwargs):
            model.invoke.return_value = schema(**{name: plan for name in schema.model_fields})
            return model

        model.with_structured_output.side_effect = structured
        document = Document(
            page_content="학업역량(40%)\n탐구역량(40%)\n잠재역량(20%)",
            metadata={"source": "college.pdf", "page": 2},
        )
        with patch("rag.future.ChatOllama", return_value=model), patch(
            "rag.future.load_college_vectorstore"
        ), patch("rag.future.retrieve_college_context", return_value=[document]), patch(
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
        self.assertIn("기존 생기부가 없다", prompt)
        self.assertNotIn("[과거 경험: 기존 생기부 관련 구간]", prompt)

    def test_subject_and_target_reach_single_generation(self):
        result, model, context = self.build_result()
        self.assertEqual(model.invoke.call_count, 1)
        prompt = str(model.invoke.call_args.args[0])
        for text in ["확률과 통계", "3학년 1학기", "세부능력특기사항", "통계학과", "기존 활동 근거", "기존 활동과 무관한 활동을 처음부터 새로 제시하지 마라"]:
            self.assertIn(text, prompt)
        self.assertEqual(len(result["future_activities"]), 3)
        self.assertEqual([a["weight"] for a in result["future_activities"]], ["40%", "40%", "20%"])
        self.assertEqual(context.call_args.kwargs["max_selected_chunks"], 3)
        self.assertTrue(context.call_args.kwargs["return_matches"])
        self.assertEqual(context.call_args.kwargs["layout_noise_terms"], ["확률과 통계"])

    def test_volunteering_ignores_stale_subject(self):
        result, model, _ = self.build_result("봉사활동", "생명과학")
        prompt = str(model.invoke.call_args.args[0])
        self.assertNotIn("생명과학", prompt)
        self.assertIn("도움이 필요한 대상", prompt)
        self.assertEqual(result["subject"], "")

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
                self.assertTrue(any("자료 비교·분석 경험" in item.value for item in app.markdown))
                self.assertTrue(any("기존 생기부 원문" in item.value for item in app.markdown))
                self.assertTrue(any("기존 활동 원문" in item.value for item in app.markdown))
                markdown_values = [item.value for item in app.markdown]
                record_block = next(
                    value for value in markdown_values
                    if "기존 생기부 원문" in value and "관련 있다고 판단한 이유" in value
                )
                self.assertLess(record_block.index("기존 생기부 원문"), record_block.index("관련 있다고 판단한 이유"))
                self.assertEqual(generate.call_args.kwargs["current_grade"], 2)
                app.button[0].click().run()
                self.assertEqual(generate.call_count, 1)
                app.selectbox(key="guide_section_type").set_value("봉사활동").run()
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
