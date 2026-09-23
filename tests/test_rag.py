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
from rag.attachment import (
    CandidateVerdict,
    OcrCorrection,
    OcrCorrectionBatch,
    RecordSelection,
    _repair_layout_intrusions,
    _retrieve_record_candidates,
    _safe_corrected_ocr,
    _obvious_text_corruption,
    _ocr_correction_suggestions,
    _grounded_experience_title,
    _fallback_experience_title,
    _section_ranges,
    _subject_ranges,
    clean_record_ocr_text,
    extract_context,
    prepare_record_context,
    split_record,
    split_record_units,
)
from rag.criteria import extract_college_criteria
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
    def test_experience_title_does_not_expose_a_truncated_ocr_sentence(self):
        original = "(기초연기) (26시간) 뮤지컬의 구성요소에 대해 알아보고 배역을 분석함."
        self.assertEqual(
            _fallback_experience_title(original, "동아리활동"),
            "기초연기 활동 경험",
        )
        self.assertEqual(
            _fallback_experience_title("자료를 조사하고 발표함.", "진로활동"),
            "진로활동에서 수행한 관련 경험",
        )

    def test_korean_ocr_suffix_suggestions_preserve_normal_words(self):
        text = "모두록 납득 시키논 뒤 회장으로 선출 팀. 서울 기록 토론 이름"
        suggestions = _ocr_correction_suggestions(text)
        self.assertIn("모두록→모두를", suggestions)
        self.assertIn("시키논→시키는", suggestions)
        self.assertIn("선출 팀→선출됨", suggestions)
        self.assertFalse(any(
            word in suggestion
            for word in ("서울", "기록", "토론", "이름")
            for suggestion in suggestions
        ))

    def test_ocr_display_correction_must_preserve_numbers_and_shape(self):
        original = "진로적성검사(2022.05.13.)틀 진행하여 미디어 진로름 탄색함."
        corrected = "진로적성검사(2022.05.13.)를 진행하여 미디어 진로를 탐색함."
        self.assertEqual(_safe_corrected_ocr(original, corrected), corrected)
        changed_date = "진로적성검사(2023.05.13.)를 진행하여 미디어 진로를 탐색함."
        self.assertEqual(_safe_corrected_ocr(original, changed_date), original)

    def test_ocr_display_correction_allows_spaces_inserted_inside_dates(self):
        original = (
            "학교 내 통합학급 홍보 촬영 활동(2021.06. 22.-202 1.07.14.)에서 "
            "진행자 역할올 맡아, 명확한 발음과 자연스러운 표정올 구사하여 스크 립트름 읽음."
        )
        corrected = (
            "학교 내 통합학급 홍보 촬영 활동(2021.06.22.-2021.07.14.)에서 "
            "진행자 역할을 맡아, 명확한 발음과 자연스러운 표정을 구사하여 스크립트를 읽음."
        )
        self.assertEqual(_safe_corrected_ocr(original, corrected), corrected)

    def test_ocr_display_correction_rejects_new_hanja(self):
        original = "진행자 역할올 맡아 스크 립트름 읽음."
        corrected_with_hanja = "진행자 역할을 맡아 스크립트朗誦."
        self.assertEqual(_safe_corrected_ocr(original, corrected_with_hanja), original)

    def test_ocr_display_correction_removes_new_obvious_spacing_errors(self):
        original = (
            "모두록 납득 시키논 공약을 통하여 1학기 학급자치회 회장 "
            "(2021.03.02.-2021.08.16.)으로 선출 팀."
        )
        corrected = (
            "모두를 납득 시키는 공약을 통하여 1 학기 학급자치회 회장 "
            "(2021.03.02.-2021.08.16.) 으로 선출됨."
        )
        self.assertEqual(
            _safe_corrected_ocr(original, corrected),
            "모두를 납득시키는 공약을 통하여 1학기 학급자치회 회장 "
            "(2021.03.02.-2021.08.16.)으로 선출됨.",
        )

    def test_missing_combined_ocr_correction_is_retried_with_focused_request(self):
        original = (
            "학교 내 통합학급 홍보 촬영 활동(2021.06. 22.-202 1.07.14.)에서 "
            "진행자 역할올 맡아, 명확한 발음과 자연스러운 표정올 구사하여 스크 립트름 읽음."
        )
        corrected = (
            "학교 내 통합학급 홍보 촬영 활동(2021.06.22.-2021.07.14.)에서 "
            "진행자 역할을 맡아, 명확한 발음과 자연스러운 표정을 구사하여 스크립트를 읽음."
        )
        correction = OcrCorrectionBatch(corrections=[
            OcrCorrection(candidate_number=1, corrected_original=corrected)
        ])
        with patch("rag.attachment.get_embeddings", return_value=CountingEmbeddings()), patch(
            "rag.attachment.ChatOllama"
        ) as model_class:
            model = model_class.return_value.with_structured_output.return_value
            model.invoke.side_effect = [correction]
            context, matches = prepare_record_context(
                original,
                "통합학급 홍보 영상 진행자 활동",
                return_matches=True,
            )
        self.assertEqual(model.invoke.call_count, 1)
        self.assertEqual(matches[0]["display_original"], corrected)
        self.assertIn(corrected, context)

    def test_clean_record_skips_the_extra_ocr_language_model_call(self):
        with patch("rag.attachment.get_embeddings", return_value=CountingEmbeddings()), patch(
            "rag.attachment.ChatOllama"
        ) as model_class:
            _, matches = prepare_record_context(
                "신문 기사와 공공 자료를 비교하여 미디어 표현 방식을 분석함.",
                "미디어 자료 비교 분석",
                return_matches=True,
            )
        self.assertTrue(matches)
        model_class.assert_not_called()

    def test_hanja_inserted_by_ocr_correction_is_retried_in_hangul(self):
        original = "진행자 역할올 맡아 스크 립트름 읽음."
        corrected = "진행자 역할을 맡아 스크립트를 읽음."
        hanja_correction = OcrCorrectionBatch(corrections=[
            OcrCorrection(candidate_number=1, corrected_original="진행자 역할을 맡아 스크립트朗誦.")
        ])
        hangul_correction = OcrCorrectionBatch(corrections=[
            OcrCorrection(candidate_number=1, corrected_original=corrected)
        ])
        with patch("rag.attachment.get_embeddings", return_value=CountingEmbeddings()), patch(
            "rag.attachment.ChatOllama"
        ) as model_class:
            model = model_class.return_value.with_structured_output.return_value
            model.invoke.side_effect = [hanja_correction, hangul_correction]
            _, matches = prepare_record_context(
                original,
                "진행자 역할과 스크립트 낭독",
                return_matches=True,
            )
        self.assertEqual(model.invoke.call_count, 2)
        self.assertEqual(matches[0]["display_original"], corrected)

    def test_normal_korean_one_syllable_words_and_line_wraps_are_not_corruption(self):
        wrapped = (
            "미디어학과의 진로를 어떻게 탐색해야 할\n"
            "지 고민하고, 언어와 매체 영역 그 중 관련 지문을 읽을 때 비판적으로 생각함."
        )
        self.assertEqual(_obvious_text_corruption(wrapped), "")
        self.assertTrue(_obvious_text_corruption("창 의 적 체 험 활 동 상 황 표 머리글"))

    def test_current_draft_terms_cannot_become_past_experience_title(self):
        original = "미디어학과 진로를 탐색하고 사회에 대한 관심과 분석 능력이 필요함을 알게 됨."
        self.assertFalse(_grounded_experience_title(
            "공익 미디어 프로젝트 윤리 헌장 제정", original
        ))
        self.assertTrue(_grounded_experience_title(
            "미디어학과 진로 탐색과 사회 분석", original
        ))

    def test_record_ranges_separate_activity_sections_and_subject(self):
        record = (
            "동아리활동\n영화 콘텐츠를 제작함.\n"
            "세부능력 및 특기사항\n수학\n통계 자료를 비교하고 해석함.\n"
            "국어\n영화 서사의 표현 방식을 분석함.\n"
            "진로활동\n직업인을 조사함."
        )
        section_ranges = _section_ranges(record, "세부능력특기사항")
        selected = " ".join(record[start:end] for start, end in section_ranges)
        self.assertIn("통계 자료", selected)
        self.assertNotIn("영화 콘텐츠", selected)
        self.assertNotIn("직업인을 조사", selected)
        subject_ranges = _subject_ranges(record, "수학")
        self.assertTrue(subject_ranges)
        subject_text = " ".join(record[start:end] for start, end in subject_ranges)
        self.assertIn("통계 자료", subject_text)
        self.assertNotIn("영화 서사", subject_text)

    def test_activity_name_inside_career_sentence_is_not_a_section_heading(self):
        record = (
            "진로활동\n희망분야 기획 및 광고 관리자\n"
            "동아시아 시민 진로탐색 동아리 활동을 통해 사회 문제를 파악함.\n"
            "자율활동\n학급회의에 참여함."
        )
        ranges = _section_ranges(record, "진로활동")
        selected = " ".join(record[start:end] for start, end in ranges)
        self.assertIn("기획 및 광고 관리자", selected)
        self.assertIn("사회 문제를 파악함", selected)
        self.assertNotIn("학급회의", selected)

    def test_record_units_support_korean_record_endings_without_periods(self):
        self.assertEqual(
            split_record_units("자료를 조사함 결과를 표로 정리함 한계를 성찰함"),
            ["자료를 조사함", "결과를 표로 정리함", "한계를 성찰함"],
        )

    def test_record_units_do_not_split_dates_into_fragments(self):
        text = (
            "진로적성검사(2022. 05. 13.)를 진행하여 강점과 약점을 분석하고 목표를 설정함. "
            "미디어 분야의 진로를 탐색함."
        )
        units = split_record_units(text)
        self.assertEqual(len(units), 2)
        self.assertIn("2022. 05. 13.", units[0])
        self.assertNotIn("05.", units)

    def test_record_ocr_cleanup_removes_table_noise_but_keeps_activity(self):
        text = (
            "강남영상미디어고등학교\n2026년 9월 21일\n반\n번호\n12\n2\n"
            "진로활동\n희망분야\n미디어학과 진학\n"
            "사회 문제를 분석하고 해결 방법을 탐구함.\n"
            "문서확인번호 : 1234 (신청인 : 학생)"
        )
        cleaned = clean_record_ocr_text(text)
        self.assertIn("진로활동", cleaned)
        self.assertIn("사회 문제를 분석", cleaned)
        self.assertNotIn("고등학교", cleaned)
        self.assertNotIn("문서확인번호", cleaned)
        self.assertNotRegex(cleaned, r"(?m)^12$")

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
        self.assertEqual(process.call_args.kwargs["config"].dpi, 200)

    def test_whole_record_is_read_and_only_relevant_passages_are_selected(self):
        full_text = "앞부분 " + "가" * 6_000 + " 끝부분 활동"
        chunks = split_record(full_text)
        self.assertGreater(len(chunks), 1)
        self.assertIn("끝부분 활동", chunks[-1])

    def test_record_context_uses_embedding_search_without_clean_text_rewrite(self):
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
        self.assertEqual(model.invoke.call_count, 0)
        self.assertTrue(matches)
        self.assertIn("데이터 분석 프로젝트를 시작함.", context)
        self.assertIn("공공 자료 두 종류를 비교함.", context)
        self.assertNotIn("관련 없는 수상 내용", context)
        self.assertNotIn("관련 없는 봉사 내용", context)
        for match in matches:
            self.assertIn(match["original"], record)
            self.assertIn("similarity_score", match)
            self.assertIn("query_hit_count", match)
            self.assertIn("rrf_score", match)
            self.assertIn("mmr_score", match)

    def test_pdf_line_wrap_does_not_cut_a_record_sentence(self):
        record = "데이터 분석 과정에서\n두 자료를 비교하여 결론을 도출함. 다음 활동을 계획함."
        units = split_record_units(record)
        self.assertEqual(len(units), 2)
        self.assertEqual(units[0], "데이터 분석 과정에서\n두 자료를 비교하여 결론을 도출함.")

    def test_strong_retrieval_is_not_vetoed_by_weak_llm_rating(self):
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
                "데이터 분석을 위해 자료 조사 활동을 수행함.",
                "데이터 분석 활동",
                return_matches=True,
            )
        self.assertTrue(matches)
        self.assertIn("데이터 분석을 위해 자료 조사 활동", context)
        self.assertNotIn("주제 이름만 비슷하고 직접 관련되는 경험은 없음", context)

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

    def test_uos_uses_student_comprehensive_type_two_weights(self):
        weights = Document(
            page_content=(
                "학생부종합전형I(면접형)\n"
                "전형단계 구분 학업역량 잠재역량 사회역량 계\n"
                "서류평가 35%(175점) 40%(200점) 25%(125점) 100%(500점)\n"
                "학생부종합전형II(서류형)\n"
                "전형단계 구분 학업역량 잠재역량 사회역량 계\n"
                "일괄 서류평가 30%(300점) 50%(500점) 20%(200점) 100%(1,000점)\n"
                "학업역량 고교 기초 학업 능력 대학 전공 기초 소양\n"
                "잠재역량 다학제적 전공수학 열의 통합적인 문제해결 역량\n"
                "사회역량 공동체 및 시민윤리의식 협동학습능력"
            ),
            metadata={"source": "서울시립대학교_모집요강.pdf", "page": 45, "printed_page": 44},
        )
        details = Document(
            page_content=(
                "다. 평가 준거\n"
                "주요 교과 학업성취도 및 성적 추이\n"
                "관심 분야 탐구 및 교육활동 경험의 우수성, 지속성, 다양성\n"
                "협력 등 팀워크 사례"
            ),
            metadata={"source": "서울시립대학교_모집요강.pdf", "page": 46, "printed_page": 45},
        )
        criteria, _ = extract_college_criteria(
            [weights, details], "교과 탐구와 모둠 협업을 수행함", "서울시립대학교",
        )
        self.assertEqual(
            [(item.area, item.weight) for item in criteria],
            [("학업역량", "30%"), ("잠재역량", "50%"), ("사회역량", "20%")],
        )
        self.assertEqual(
            criteria[1].subcriteria,
            ["다학제적 전공수학 열의", "통합적인 문제해결 역량"],
        )
        self.assertEqual(criteria[2].evaluation_points, ["협력 등 팀워크 사례"])
        self.assertEqual(criteria[2].printed_pages, [44, 45])

    def test_new_college_profiles_use_their_official_weights(self):
        cases = (
            (
                "명지대학교", "미디어커뮤니케이션학과",
                "명지인재서류 서류형 학업성취도 학업태도 진로설계역량 진로학업역량 "
                "진로탐색역량 성실성과 규칙준수 협업과 소통능력",
                [("학업역량", "30%"), ("진로역량", "50%"), ("공동체역량", "20%")],
            ),
            (
                "건국대학교", "컴퓨터공학부",
                "학생부종합전형 전체 1,000점 학업성취도 학업태도 탐구력 "
                "전공(계열) 관련 교과 이수 노력 전공(계열) 관련 교과 성취도 "
                "진로 탐색 활동과 경험 협업과 소통능력 나눔과 배려 성실성과 규칙준수 리더십",
                [("학업역량", "30%"), ("진로역량", "40%"), ("공동체역량", "30%")],
            ),
            (
                "건국대학교", "KU자유전공학부",
                "학생부종합전형 전체 1,000점 학업성취도 학업태도 자기주도성 "
                "창의적 문제해결력 경험의 다양성 탐구력 협업과 소통능력 나눔과 배려 "
                "성실성과 규칙준수 리더십",
                [("학업역량", "20%"), ("성장역량", "50%"), ("공동체역량", "30%")],
            ),
            (
                "가톨릭대학교", "심리학과",
                "서류종합평가요소별 반영비율 주요 평가 관점 학업성취도 학업태도 탐구력 "
                "전공(계열) 관련 교과 이수 노력 전공(계열) 관련 교과 성취도 "
                "진로 탐색 활동과 경험 협업과 소통능력 나눔과 배려 성실성과 규칙준수 리더십",
                [("학업역량", "40%"), ("진로역량", "35%"), ("공동체역량", "25%")],
            ),
        )
        for university, department, text, expected in cases:
            with self.subTest(university=university, department=department):
                document = Document(
                    page_content=text,
                    metadata={"source": f"{university}_모집요강.pdf", "page": 1},
                )
                criteria, evidence_ids = extract_college_criteria(
                    [document], "교과 탐구를 계획하고 친구들과 협력함", university, department,
                )
                self.assertEqual([(item.area, item.weight) for item in criteria], expected)
                self.assertEqual(evidence_ids, [1])

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

if __name__ == "__main__":
    unittest.main()
