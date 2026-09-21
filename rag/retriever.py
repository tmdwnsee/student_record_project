"""대학 모집요강과 생기부 작성요령을 검색합니다."""

import re
import unicodedata

from langchain_chroma import Chroma
from langchain_core.documents import Document

COLLEGE_RESULT_COUNT = 8
COLLEGE_DETAIL_RESULT_COUNT = 5
GUIDELINE_RESULT_COUNT = 5
GUIDELINE_CANDIDATES_PER_QUERY = 12
RRF_RANK_CONSTANT = 10
SENTENCE_QUERY_PREFIX = "원문 문장::"
EVALUATIVE_CLAIM_PATTERN = (
    r"완벽|모든|전부|최고|매우|(?:크게|많이|매우)\s*향상|뛰어나|우수|탁월|훌륭|"
    r"다른\s+.+보다|누구보다|역량을?\s*(?:모두|완전히)?\s*갖추|"
    r"능력을?\s*(?:보여|갖추|향상)|성실|열심|잘함|생각함"
)
NAMED_ENTITY_PATTERN = (
    r"[가-힣A-Za-z0-9]+(?:대학교|대학|기관|센터|회사|재단|문고|스토어|쇼핑몰)|"
    r"[가-힣A-Za-z0-9]+\s+(?:운동화|신발|노트북|태블릿|휴대폰|스마트폰|이어폰|카메라|의류|가방|음료)|"
    r"[가-힣A-Za-z0-9]+(?:와|과)\s*[가-힣A-Za-z0-9]+(?:을|를)\s*(?:이용|사용|활용)"
)
TEST_AWARD_PATTERN = (
    r"공인\s*어학\s*시험|토익|토플|텝스|아이엘츠|"
    r"(?:교내|교외|전국|국제)?\s*(?:대회|공모전)\s*(?:수상|입상)|"
    r"수상\s*실적|자격증|인증\s*취득"
)
ADMINISTRATIVE_NOISE_TERMS = (
    "부당 요구", "입력 및 정정 권한", "입력 및 정정 업무", "부정청탁", "담당 부서로 질의",
    "학교생활기록부 작성에 필요한 보조부", "학업성적관리위원회에서는",
)
MIN_GUIDELINE_SCORE = 0.22

COLLEGE_QUERY = """학생부종합전형 서류평가 평가영역 반영비율 배점 퍼센트
평가요소 학업역량 탐구역량 잠재역량 문제해결능력 기준"""

GUIDELINE_QUERY = """학교생활기록부 작성 원칙 기재 시 주의사항 교사가 직접 관찰 평가한 내용
활동 기록 구체적 사실 표현 금지사항"""

COMMON_GUIDELINE_QUERY = """
학교생활기록부 작성 시 유의사항
학교생활기록부 기재 금지 사항
학교생활기록부 서술형 항목
교사가 직접 관찰 평가한 내용
학생이 직접 작성한 자료 활용
학교생활기록부 작성 원칙
"""


SECTION_GUIDELINE_QUERIES = {
    "세특": """
교과학습발달상황
세부능력 및 특기사항
과목별 세부능력 및 특기사항
성취기준
성취수준
학습활동 참여도
학습활동 태도
성취과정
성취특성
""",

    "자율자치활동": """
창의적 체험활동상황
자율 자치 활동
자율자치활동
특기사항
학급담임교사
구체적 활동 내용
""",

    "동아리활동": """
창의적 체험활동상황
동아리활동
동아리 활동
특기사항
동아리 담당교사
구체적 활동 내용
""",

    "진로활동": """
창의적 체험활동상황
진로활동
진로희망분야
진로검사
진로상담
관심분야
진로희망 관련 활동내용
진로 특성
""",

    "행특": """
행동특성 및 종합의견
행동특성
종합의견
학년 동안 지속적으로 관찰
학생을 총체적으로 이해
학급담임교사
""",
}


def retrieve_college_context(store: Chroma, student_draft: str, university: str = "", department: str = "") -> list[Document]:
    """평가 비율과 평가요소가 서로 다른 청크에 있어도 함께 검색합니다."""
    queries = [
        f"{COLLEGE_QUERY}\n검토할 학생 활동: {student_draft}\n희망 대학: {university}\n희망 학과: {department}",
        "학생부종합 서류종합평가 서류평가 평가항목 세부평가항목 평가기준 배점 반영비율 최고점 최저점",
        "학생부종합 평가영역 평가요소 평가내용 학업역량 전공적합성 진로역량 자기주도역량 공동체역량 탐구역량 성장가능성 인성 사회성",
        "학생부종합 학업 수행과정 탐구능력 진로탐색 전공 관심 역할 주도성 협업 소통 평가",
        "학생부종합 평가 항목별 반영 비율 30% 40% 50% 20% 계 100%",
    ]
    results: list[Document] = []
    seen = set()
    for index, query in enumerate(queries):
        limit = COLLEGE_RESULT_COUNT if index == 0 else COLLEGE_DETAIL_RESULT_COUNT
        for document in store.similarity_search(query, k=limit):
            key = (
                str(document.metadata.get("source", "")),
                document.metadata.get("page"),
                document.page_content,
            )
            if key not in seen:
                seen.add(key)
                results.append(document)

    # 표가 여러 청크로 나뉘면 검색된 일부 청크만으로 행 관계를 복원하기 어렵습니다.
    # 평가기준 신호가 발견된 페이지는 해당 페이지의 청크 전체를 추가합니다.
    evaluation_pages = sorted({
        document.metadata.get("page")
        for document in results
        if document.metadata.get("page") is not None
        and re.search(r"서류종합평가|서류평가|평가항목|평가요소|평가기준|반영\s*비율", document.page_content)
    })
    if evaluation_pages:
        page_data = store.get(
            where={"page": {"$in": evaluation_pages}},
            include=["documents", "metadatas"],
        )
        for content, metadata in zip(page_data.get("documents", []), page_data.get("metadatas", [])):
            document = Document(page_content=content, metadata=metadata)
            key = (str(metadata.get("source", "")), metadata.get("page"), content)
            if key not in seen:
                seen.add(key)
                results.append(document)
    return results


def _compact(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", unicodedata.normalize("NFKC", text).lower())


def _guideline_queries(student_draft: str) -> list[tuple[str, str, float, tuple[str, ...]]]:
    """공통 검색에 초안에서 감지된 규정 범주만 조건부로 추가합니다."""
    queries = [
        ("초안 의미 유사도", f"학교생활기록부 기재 시 다음 문장의 문제와 관련된 작성요령:\n{student_draft}", 2.0, ()),
        ("관찰·사실성", "학교생활기록부 교사가 직접 관찰 평가 허위 과장 부풀려 기재 금지 객관적 사실", 1.5,
         ("직접 관찰", "과장", "부풀려", "허위")),
        ("구체성·개별성", "학교생활기록부 활동내용 구체적 사실 과정 결과 개별적 특성이 드러나는 사항 누가기록", 1.3,
         ("개별적 특성", "활동내용", "누가기록")),
        ("핵심 작성원칙", GUIDELINE_QUERY, 1.0, ("작성 시 유의사항", "작성 원칙")),
    ]

    # 특정 표현 사전에 의존하지 않고 각 문장을 작성요령과 직접 비교합니다.
    for sentence in _split_draft_sentences(student_draft)[:10]:
        queries.append((
            f"{SENTENCE_QUERY_PREFIX}{sentence}",
            f"학교생활기록부에 다음 문장을 기록할 때 적용되는 기재 원칙과 주의사항:\n{sentence}",
            1.7,
            (),
        ))

    named_entity_signal = re.search(NAMED_ENTITY_PATTERN, student_draft)
    if named_entity_signal:
        queries.append((
            "특정 명칭 규정",
            "학교생활기록부 구체적인 특정 대학명 기관명 상호명 상품명 강사명 기재 금지",
            1.6,
            ("특정 대학명", "기관명", "상호명", "강사명", "기재할 수 없음"),
        ))

    if re.search(TEST_AWARD_PATTERN, student_draft):
        queries.append((
            "시험·수상·자격 규정",
            "학교생활기록부 공인어학시험 성적 수상실적 교내외 대회 자격증 기재 불가",
            1.5,
            ("공인어학시험", "수상 실적", "대회", "자격증"),
        ))

    if re.search(r"논문|학회|특허|실용신안|상표|출간|저서", student_draft):
        queries.append((
            "논문·지식재산 규정",
            "학교생활기록부 논문 학회 발표 도서 출간 특허 지식재산권 기재 불가",
            1.5,
            ("논문", "학회", "도서출간", "지식재산권"),
        ))

    if re.search(r"부모|아버지|어머니|가족|직업|직장|주소|소득", student_draft):
        queries.append((
            "개인·가족정보 규정",
            "학교생활기록부 부모 친인척 사회 경제적 지위 직업 직장 개인정보 기재 금지",
            1.5,
            ("부모", "친인척", "직업명", "직장명", "사회･경제적 지위"),
        ))

    return queries


def _split_draft_sentences(student_draft: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", student_draft) if part.strip()]


def _sentences_matching(student_draft: str, pattern: str) -> list[str]:
    return [sentence for sentence in _split_draft_sentences(student_draft) if re.search(pattern, sentence)]


def _draft_matches_by_reason(student_draft: str) -> dict[str, list[str]]:
    """검색 범주를 작동시킨 초안의 실제 문장을 반환합니다."""
    return {
        "특정 명칭 규정": _sentences_matching(
            student_draft,
            NAMED_ENTITY_PATTERN,
        ),
        "시험·수상·자격 규정": _sentences_matching(student_draft, TEST_AWARD_PATTERN),
        "논문·지식재산 규정": _sentences_matching(student_draft, r"논문|학회|특허|실용신안|상표|출간|저서"),
        "개인·가족정보 규정": _sentences_matching(student_draft, r"부모|아버지|어머니|가족|직업|직장|주소|소득"),
        "관찰·사실성": _sentences_matching(
            student_draft,
            EVALUATIVE_CLAIM_PATTERN + r"|좋은\s*성과|잘했|적극적|노력",
        ),
        "구체성·개별성": _sentences_matching(
            student_draft,
            EVALUATIVE_CLAIM_PATTERN + r"|다양한\s*활동|좋은\s*성과|여러\s*활동|많은\s*활동",
        ),
    }


def _attach_draft_matches(documents: list[Document], student_draft: str) -> list[Document]:
    matches_by_reason = _draft_matches_by_reason(student_draft)
    enriched = []
    for document in documents:
        matches = []
        matches_by_category = {}
        for reason in document.metadata.get("retrieval_keyword_groups", {}):
            category_matches = matches_by_reason.get(reason, [])
            if category_matches:
                matches_by_category[reason] = list(category_matches)
                matches.extend(category_matches)
        keyword_groups = document.metadata.get("retrieval_keyword_groups", {})
        if keyword_groups:
            semantic_matches = document.metadata.get("semantic_draft_matches", [])
            if set(keyword_groups) & {"관찰·사실성", "구체성·개별성"}:
                semantic_matches = [
                    sentence for sentence in semantic_matches
                    if re.search(EVALUATIVE_CLAIM_PATTERN, sentence)
                ]
            for sentence in semantic_matches:
                matches.append(sentence)
                for reason in set(keyword_groups) & {"관찰·사실성", "구체성·개별성"}:
                    matches_by_category.setdefault(reason, []).append(sentence)
        metadata = dict(document.metadata)
        metadata["draft_matches"] = list(dict.fromkeys(matches))
        metadata["draft_matches_by_category"] = {
            reason: list(dict.fromkeys(sentences))
            for reason, sentences in matches_by_category.items()
        }
        enriched.append(Document(page_content=document.page_content, metadata=metadata))
    return enriched


def rerank_guideline_candidates(
    ranked_results: list[tuple[str, float, tuple[str, ...], list[Document]]],
    *,
    limit: int = GUIDELINE_RESULT_COUNT,
) -> list[Document]:
    """Weighted RRF와 규정 핵심어 점수로 작성요령 후보를 재정렬합니다."""
    documents, scores, reasons, keywords, keyword_groups, semantic_matches = {}, {}, {}, {}, {}, {}
    for label, weight, terms, results in ranked_results:
        for rank, document in enumerate(results, 1):
            key = (str(document.metadata.get("source", "")), document.metadata.get("page"), document.page_content)
            documents[key] = document
            scores[key] = scores.get(key, 0.0) + weight / (RRF_RANK_CONSTANT + rank)
            reasons.setdefault(key, set()).add(label)
            if label.startswith(SENTENCE_QUERY_PREFIX) and rank <= 2:
                semantic_matches.setdefault(key, set()).add(label.removeprefix(SENTENCE_QUERY_PREFIX))
            content = _compact(document.page_content)
            matched = [term for term in terms if _compact(term) in content]
            if matched:
                scores[key] += 0.03 * weight * len(matched)
                reasons[key].add(f"핵심어: {', '.join(matched)}")
                keywords.setdefault(key, set()).update(matched)
                keyword_groups.setdefault(key, {}).setdefault(label, set()).update(matched)

    for key, document in documents.items():
        noise = [term for term in ADMINISTRATIVE_NOISE_TERMS if term in document.page_content]
        if noise:
            scores[key] -= 0.12
            reasons[key].add(f"행정 안내 감점: {', '.join(noise)}")

    selected, page_counts = [], {}
    for key in sorted(scores, key=scores.get, reverse=True):
        if scores[key] < MIN_GUIDELINE_SCORE:
            continue
        document = documents[key]
        page_key = (str(document.metadata.get("source", "")), document.metadata.get("page"))
        if page_counts.get(page_key, 0) >= 2:
            continue
        metadata = dict(document.metadata)
        metadata["retrieval_score"] = round(scores[key], 6)
        metadata["retrieval_reasons"] = sorted(reasons[key])
        metadata["retrieval_keywords"] = sorted(keywords.get(key, set()))
        metadata["retrieval_keyword_groups"] = {
            label: sorted(terms) for label, terms in keyword_groups.get(key, {}).items()
        }
        metadata["semantic_draft_matches"] = sorted(semantic_matches.get(key, set()))
        selected.append(Document(page_content=document.page_content, metadata=metadata))
        page_counts[page_key] = page_counts.get(page_key, 0) + 1
        if len(selected) == limit:
            break
    return selected


def retrieve_guideline_context(
    store: Chroma,
    student_draft: str,
    record_section: str,
) -> list[Document]:

    if record_section not in SECTION_GUIDELINE_QUERIES:
        raise ValueError(
            f"지원하지 않는 생기부 항목입니다: "
            f"{record_section}"
        )

    common_query = f"""
{COMMON_GUIDELINE_QUERY}

현재 검토할 초안:
{student_draft}
"""

    section_query = f"""
{SECTION_GUIDELINE_QUERIES[record_section]}

현재 검토할 초안:
{student_draft}
"""

    common_results = store.similarity_search(
        common_query,
        k=5,
    )

    section_results = store.similarity_search(
        section_query,
        k=6,
    )

    results = []
    seen = {}

    for scope, documents in (
        ("공통", common_results),
        (record_section, section_results),
    ):

        for document in documents:

            key = (
                str(
                    document.metadata.get(
                        "source",
                        ""
                    )
                ),
                document.metadata.get("page"),
                document.page_content,
            )

            # 공통/항목 검색에서 같은 chunk가 나온 경우
            if key in seen:

                existing = seen[key]

                scopes = existing.metadata.get(
                    "guideline_scopes",
                    []
                )

                existing.metadata[
                    "guideline_scopes"
                ] = list(
                    dict.fromkeys(
                        scopes + [scope]
                    )
                )

                continue

            metadata = dict(
                document.metadata
            )

            metadata[
                "guideline_scopes"
            ] = [scope]

            metadata[
                "retrieval_reasons"
            ] = [
                f"{scope} 작성요령 검색"
            ]

            copied = Document(
                page_content=document.page_content,
                metadata=metadata,
            )

            seen[key] = copied
            results.append(copied)

    return results
