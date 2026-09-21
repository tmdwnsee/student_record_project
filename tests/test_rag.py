"""API 비용 없이 중복 저장, 인용 검증, 화면 동작을 확인합니다."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from streamlit.testing.v1 import AppTest

from config import PROJECT_ROOT
from rag.attachment import (
    CandidateVerdict,
    RecordSelection,
    _repair_layout_intrusions,
    _retrieve_record_candidates,
    extract_context,
    prepare_record_context,
    split_record,
    split_record_units,
)
from rag.chain import SAMPLE_DRAFT, review_draft
from rag.criteria import extract_college_criteria
from rag.retriever import GUIDELINE_CANDIDATES_PER_QUERY, retrieve_guideline_context
from rag.reviewer import ReviewResult
from rag.validation import (
    Evidence,
    conservative_rewrite,
    looks_like_record_sentence,
    select_evidence,
    validate_evidence,
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


class RecordEmbeddings:
    @staticmethod
    def _vector(text):
        return [
            float("데이터" in text),
            float("공공" in text or "자료" in text),
            float("봉사" in text or "수상" in text),
        ]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


class RagTests(unittest.TestCase):
    def test_guideline_queries_adapt_to_draft_instead_of_always_using_example_rules(self):
        generic_store = Mock()
        generic_store.similarity_search.return_value = []
        retrieve_guideline_context(generic_store, "수학 문제의 풀이 과정을 비교하고 발표함", "세특")
        generic_query_count = generic_store.similarity_search.call_count
        self.assertEqual(generic_query_count, 2)

        research_store = Mock()
        research_store.similarity_search.return_value = []
        retrieve_guideline_context(research_store, "연구 결과를 논문으로 작성하고 학회에서 발표함", "세특")
        self.assertEqual(research_store.similarity_search.call_count, generic_query_count)
        self.assertEqual(
            [call.kwargs["k"] for call in research_store.similarity_search.call_args_list],
            [5, 6],
        )

    def test_guideline_reranking_metadata_is_available_for_streamlit(self):
        store = Mock()
        store.similarity_search.return_value = [
            Document(
                page_content="학생의 구체적인 활동내용과 개별적 특성이 드러나야 함",
                metadata={"source": "student_record_rule.pdf", "page": 23},
            )
        ]
        result = retrieve_guideline_context(store, "탐구 활동을 수행함", "세특")
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].metadata["retrieval_reasons"])
        self.assertEqual(result[0].metadata["guideline_scopes"], ["공통", "세특"])

    def test_guideline_result_shows_the_triggering_original_sentence(self):
        store = Mock()
        store.similarity_search.return_value = [
            Document(
                page_content="논문을 학회지에 등재하거나 학회에서 발표한 사실은 기재할 수 없음",
                metadata={"source": "student_record_rule.pdf", "page": 23},
            )
        ]
        draft = "실험을 수행함. 실험 결과를 논문으로 작성하여 학회에서 발표함."
        result = retrieve_guideline_context(store, draft, "세특")
        self.assertEqual(result[0].metadata["guideline_scopes"], ["공통", "세특"])

    def test_query_hit_without_rule_keywords_is_not_linked_to_draft(self):
        store = Mock()
        store.similarity_search.return_value = [
            Document(
                page_content="교육지원청 담당 부서에 문의하는 절차",
                metadata={"source": "student_record_rule.pdf", "page": 25},
            )
        ]
        result = retrieve_guideline_context(store, "논문을 작성하여 학회에서 발표함.", "세특")
        self.assertTrue(result)
        self.assertEqual(result[0].metadata["guideline_scopes"], ["공통", "세특"])

    def test_unseen_evaluative_sentence_is_connected_semantically(self):
        store = Mock()
        store.similarity_search.return_value = [
            Document(
                page_content="교사가 직접 관찰한 사실을 기록하고 단순 사실을 과장하거나 부풀리지 않아야 함",
                metadata={"source": "student_record_rule.pdf", "page": 24},
            )
        ]
        sentence = "한국 현대문학에 대한 완벽한 이해력을 갖추고 다른 학생보다 뛰어난 능력을 보여주었다."
        result = retrieve_guideline_context(store, sentence, "세특")
        self.assertEqual(result[0].metadata["guideline_scopes"], ["공통", "세특"])

    def test_draft_sentences_are_linked_only_to_their_matching_rule_category(self):
        store = Mock()
        store.similarity_search.return_value = [
            Document(
                page_content=(
                    "활동내용에 따른 개별적 특성이 드러나야 함. "
                    "구체적인 특정 대학명과 상호명은 기재할 수 없음."
                ),
                metadata={"source": "student_record_rule.pdf", "page": 23},
            )
        ]
        named = "성균관대학교 진학을 목표로 문학 활동에 참여함."
        vague = "여러 활동에 매우 성실하게 참여함."
        result = retrieve_guideline_context(store, f"{named} {vague}", "세특")
        self.assertEqual(result[0].metadata["guideline_scopes"], ["공통", "세특"])

    def test_ordinary_good_score_does_not_trigger_exam_award_rule(self):
        store = Mock()
        store.similarity_search.return_value = []
        retrieve_guideline_context(store, "모둠원이 좋은 점수를 받을 수 있도록 도와줌.", "세특")
        queries = [call.args[0] for call in store.similarity_search.call_args_list]
        self.assertFalse(any("공인어학시험 성적" in query for query in queries))

    def test_revised_text_rejects_guidance_and_evidence(self):
        self.assertTrue(looks_like_record_sentence("Python으로 데이터를 분석하고 문제 해결 과정을 수행함."))
        for text in (
            "탐구역량이 드러나도록 작성할 것.",
            "대학 평가기준과 반영 비율을 고려해야 함.",
            "student_record_rule.pdf 근거를 참고함.",
        ):
            with self.subTest(text=text):
                self.assertFalse(looks_like_record_sentence(text))
        self.assertTrue(looks_like_record_sentence(
            "Python으로 데이터를 수집하고 시각화하여 논리적 사고력을 기름."
        ))
        self.assertEqual(
            conservative_rewrite("데이터 분석 프로젝트를 진행하며 Python으로 데이터를 분석함."),
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

    def test_record_context_uses_embedding_search_and_llm_validation(self):
        record = (
            "관련 없는 수상 내용. 데이터 분석 프로젝트를 시작함. "
            "공공 자료 두 종류를 비교함. 분석의 한계를 기록함. 관련 없는 봉사 내용."
        )
        query = "데이터 분석 공공 자료 비교"
        self.assertEqual(len(split_record_units(record)), 5)
        with patch("rag.attachment.get_embeddings", return_value=RecordEmbeddings()):
            candidates = _retrieve_record_candidates(record, query)
        self.assertTrue(candidates)
        self.assertGreater(candidates[0]["similarity_score"], 0)
        selected_numbers = [
            index for index, candidate in enumerate(candidates, 1)
            if "데이터 분석" in candidate["original"] or "공공 자료" in candidate["original"]
        ]
        selection = RecordSelection(verdicts=[
            CandidateVerdict(
                candidate_number=number,
                relevance=3,
                text_integrity=2,
                connection_reason="원문의 자료 비교 경험이 초안의 데이터 분석 방법과 관련됨",
            )
            for number in selected_numbers
        ])
        with patch("rag.attachment.get_embeddings", return_value=RecordEmbeddings()), patch(
            "rag.attachment.ChatOllama"
        ) as model_class:
            model = model_class.return_value.with_structured_output.return_value
            model.invoke.return_value = selection
            context, matches = prepare_record_context(record, query, return_matches=True)
        self.assertEqual(model.invoke.call_count, 1)
        self.assertTrue(matches)
        self.assertIn("데이터 분석 프로젝트를 시작함.", context)
        self.assertIn("공공 자료 두 종류를 비교함.", context)
        self.assertNotIn("관련 없는 수상 내용", context)
        self.assertNotIn("관련 없는 봉사 내용", context)
        for match in matches:
            self.assertIn(match["original"], record)
            self.assertIn("similarity_score", match)
            self.assertNotIn("hit_score", match)
            self.assertNotIn("mmr_score", match)

    def test_pdf_line_wrap_does_not_cut_a_record_sentence(self):
        record = "데이터 분석 과정에서\n두 자료를 비교하여 결론을 도출함. 다음 활동을 계획함."
        units = split_record_units(record)
        self.assertEqual(len(units), 2)
        self.assertEqual(units[0], "데이터 분석 과정에서\n두 자료를 비교하여 결론을 도출함.")

    def test_record_context_rejects_weak_llm_match(self):
        selection = RecordSelection(verdicts=[
            CandidateVerdict(
                candidate_number=1,
                relevance=1,
                text_integrity=2,
                connection_reason="주제 이름만 비슷하고 직접 관련되는 경험은 없음",
            )
        ])
        with patch("rag.attachment.get_embeddings", return_value=RecordEmbeddings()), patch(
            "rag.attachment.ChatOllama"
        ) as model_class:
            model_class.return_value.with_structured_output.return_value.invoke.return_value = selection
            context, matches = prepare_record_context(
                "데이터 분석과 무관한 자료 조사 활동을 수행함.",
                "데이터 분석 활동",
                return_matches=True,
            )
        self.assertFalse(matches)
        self.assertIn("관련성과 원문 무결성 검증을 모두 통과한 활동을 찾지 못했습니다", context)

    def test_record_context_rejects_layout_corrupted_text(self):
        broken = "자료를 선별한 뒤 데이터 저널리즘의 장 미디어 콘텐츠 기초 면 구성을 설계함."
        selection = RecordSelection(verdicts=[
            CandidateVerdict(
                candidate_number=1,
                relevance=3,
                text_integrity=2,
                connection_reason="데이터 저널리즘 주제가 초안과 관련됨",
            )
        ])
        with patch("rag.attachment.get_embeddings", return_value=RecordEmbeddings()), patch(
            "rag.attachment.ChatOllama"
        ) as model_class:
            model_class.return_value.with_structured_output.return_value.invoke.return_value = selection
            context, matches = prepare_record_context(broken, "데이터 저널리즘", return_matches=True)
        self.assertFalse(matches)
        self.assertNotIn(broken, context)

    def test_course_name_inserted_by_pdf_table_is_removed_and_word_is_restored(self):
        broken = "데이터 저널리즘의 장 미디어 콘텐츠 기초 면 구성을 설계함."
        repaired, changed = _repair_layout_intrusions(broken, ["미디어 콘텐츠 기초"])
        self.assertTrue(changed)
        self.assertEqual(repaired, "데이터 저널리즘의 장면 구성을 설계함.")

    def test_review_draft_uses_selected_record_section(self):
        with patch("rag.chain.load_guideline_vectorstore", return_value=object()), patch(
            "rag.chain.retrieve_guideline_context", return_value=[]
        ) as guideline, patch("rag.chain.generate_review") as generate:
            review_draft("새 초안", "세특")
        guideline.assert_called_once_with(guideline.call_args.args[0], "새 초안", "세특")
        generate.assert_called_once_with(
            student_draft="새 초안",
            record_section="세특",
            guideline_results=[],
        )

    def test_college_criteria_are_extracted_from_source(self):
        document = Document(
            page_content=(
                "학업수월성(200점) 우리대학에 입학할 만한 충분한 학업능력을 보여주는가\n"
                "학업역량(40%) - 학업 관련 활동 및 성취수준, 학업 태도, 학업 여건 등 학업충실성(200점)\n"
                "탐구확장성(200점) 관심 분야에 대한 호기심과 이를 탐구하기 위한 노력이 있는가\n"
                "탐구역량(40%) - 진로 탐색 의지, 지적 호기심과 탐구 의지\n"
                "탐구주도성(200점) - 배움에 대한 관심 및 열의, 활동 내용 등\n"
                "미래성장성(100점) 자기주도적 리더가 될 자질 및 발전가능성이 있는가\n"
                "잠재역량(20%) - 자기주도성, 리더십, 이타성, 소통 능력, 성실성 등 공동체의식(100점)"
            ),
            metadata={"source": "college_table.pdf", "page": 71},
        )
        criteria, evidence_ids = extract_college_criteria(
            [document], "관심 분야의 자료를 조사하고 Python으로 데이터를 분석함.", "성균관대학교",
        )
        self.assertEqual(
            [(item.area, item.weight) for item in criteria],
            [("학업역량", "40%"), ("탐구역량", "40%"), ("잠재역량", "20%")],
        )
        self.assertEqual(evidence_ids, [1])
        self.assertNotIn("공동체의식", [item.area for item in criteria])
        self.assertEqual(criteria[0].subcriteria, ["학업수월성", "학업충실성"])
        self.assertEqual(criteria[1].subcriteria, ["탐구확장성", "탐구주도성"])
        self.assertEqual(criteria[2].subcriteria, ["미래성장성", "공동체의식"])
        self.assertEqual(
            criteria[1].evaluation_question,
            "관심 분야에 대한 호기심과 이를 탐구하기 위한 노력이 있는가",
        )
        self.assertEqual(
            criteria[1].evaluation_points,
            ["진로 탐색 의지, 지적 호기심과 탐구 의지", "배움에 대한 관심 및 열의, 활동 내용 등"],
        )
        self.assertEqual(criteria[1].draft_evidence, ["관심 분야의 탐구 과정"])
        self.assertEqual(criteria[1].missing_aspects, ["탐구 범위의 확장"])
        self.assertIn("조사·분석 방법", criteria[1].revision_direction)
        self.assertIn("탐구확장성, 탐구주도성", criteria[1].recommendation)

    def test_dongguk_student_comprehensive_criteria_are_found_without_fixed_page(self):
        details = Document(
            page_content=(
                "서류종합평가\n○ 평가항목 및 내용 : 평가서류의 내용을 평가항목별로 종합평가\n"
                "■ 입학 후 학업을 수행할 수 있는 기초수학역량\n기초학업역량\n"
                "■ 기초교과 중심의 종합적인 학업역량\n학업역량\n"
                "■ 학업 수행과정에서의 주도적인 태도와 탐구능력\n학습의 주도성\n"
                "■ 전공 관련 교과목의 학업 이수 과정\n전공수학역량\n"
                "■ 전공 관련 교과목의 학업 성취도\n전공적합성\n"
                "■ 진로탐색 활동 노력 및 탐구과정\n전공관심도 및 진로탐색노력\n"
                "■ 학교생활의 다양한 영역에서 주도적으로 역할을 수행한 경험\n역할의 주도성\n"
                "■ 공동체의 목표달성을 위해 협력한 경험\n협업소통능력\n인성 및 사회성"
            ),
            metadata={"source": "동국대학교_모집요강.pdf", "page": 92, "start_index": 0},
        )
        weights = Document(
            page_content=(
                "■ 학생부종합 Do Dream, 불교추천인재\n평가항목 세부평가항목 반영비율\n"
                "기초학업역량\n학업역량 30% 30점\n학습의 주도성\n"
                "전공수학역량\n전공적합성 50% 50점\n전공관심도 및 진로탐색노력\n"
                "역할의 주도성\n인성 및 사회성 20% 20점\n협업소통능력\n계 100%"
            ),
            metadata={"source": "동국대학교_모집요강.pdf", "page": 93, "start_index": 0},
        )
        criteria, evidence_ids = extract_college_criteria(
            [details, weights], "문학 작품을 조사하고 친구들과 토론함", "동국대학교",
        )
        self.assertEqual(
            [(item.area, item.weight) for item in criteria],
            [("학업역량", "30%"), ("전공적합성", "50%"), ("인성 및 사회성", "20%")],
        )
        self.assertEqual(criteria[1].subcriteria, ["전공수학역량", "전공관심도 및 진로탐색노력"])
        self.assertIn("전공 관련 교과목의 학업 이수 과정", criteria[1].evaluation_points)
        self.assertFalse(any("전공 관련" in point for point in criteria[0].evaluation_points))
        self.assertEqual(criteria[1].evidence_pages, [92, 93])
        self.assertEqual(set(evidence_ids), {1, 2})

    def test_hierarchical_subcriteria_weights_are_aggregated_for_general_department(self):
        first_page = Document(
            page_content=(
                "학생부종합전형 서류평가\n평가 요소·비율 및 평가 항목\n"
                "학업역량\n학업성취도 (25%)\n학업태도 및 탐구력 (15%)\n"
                "- 대학 수학에 필요한 기본 교과목의\n교과 성적은 적절한가? "
                "그 외 교과목의 성취는 어느 정도인가?\n"
                "- 자기주도적으로 학습하려는 의지가 있는가?\n"
                "진로역량\n전공(계열) 관련 교과 이수 노력 및 성취도 (25%)\n"
                "진로 탐색 활동과 경험 (15%)\n- 관심 분야 활동에 참여한 경험이 있는가?\n"
                "자기주도역량\n자기주도 교과 이수 노력 및 성취도 (25%)\n"
                "자기주도 진로 탐색 활동과 경험 (15%)"
            ),
            metadata={
                "source": "경희대학교_모집요강.pdf", "page": 62,
                "printed_page": 61, "start_index": 0,
            },
        )
        second_page = Document(
            page_content=(
                "공동체역량\n협업과 소통능력, 리더십 (10%)\n"
                "- 공동의 과제를 수행한 경험이 있는가?\n"
                "나눔과 배려, 성실성과 규칙준수 (10%)\n"
                "- 자신이 맡은 역할에 최선을 다했는가?"
            ),
            metadata={
                "source": "경희대학교_모집요강.pdf", "page": 63,
                "printed_page": 62, "start_index": 0,
            },
        )
        criteria, _ = extract_college_criteria(
            [first_page, second_page], "문학 작품을 조사하고 모둠 토론에 참여함",
            "경희대학교", "국어국문학과",
        )
        self.assertEqual(
            [(item.area, item.weight) for item in criteria],
            [("학업역량", "40%"), ("진로역량", "40%"), ("공동체역량", "20%")],
        )
        self.assertEqual(criteria[0].subcriteria, ["학업성취도", "학업태도 및 탐구력"])
        self.assertIn(
            "대학 수학에 필요한 기본 교과목의 교과 성적은 적절한가? 그 외 교과목의 성취는 어느 정도인가?",
            criteria[0].evaluation_points,
        )
        self.assertFalse(any("관심 분야 활동" in point for point in criteria[0].evaluation_points))
        self.assertEqual(criteria[0].printed_pages, [61])
        self.assertEqual(criteria[2].printed_pages, [62])

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
