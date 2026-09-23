"""대학 모집요강과 교육과정을 검색합니다."""

import re

from langchain_chroma import Chroma
from langchain_core.documents import Document

from ingestion.prepare_documents import normalize_course_key

COLLEGE_RESULT_COUNT = 8
COLLEGE_DETAIL_RESULT_COUNT = 5

COLLEGE_QUERY = """학생부종합전형 서류평가 평가영역 반영비율 배점 퍼센트
평가요소 학업역량 탐구역량 잠재역량 문제해결능력 기준"""


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

def retrieve_curriculum_context(
    store: Chroma,
    *,
    subject: str,
    department: str,
    section_type: str,
    target_semester: str,
    student_draft: str,
    limit: int = 6,
) -> list[Document]:
    """정확한 과목 카드를 우선하고, 없으면 입력 경험과 학과로 의미 검색합니다."""
    results: list[Document] = []
    seen: set[tuple] = set()

    def add(document: Document) -> None:
        key = (
            str(document.metadata.get("source", "")),
            document.metadata.get("page"),
            document.metadata.get("start_index", 0),
            document.page_content,
        )
        if key not in seen:
            seen.add(key)
            results.append(document)

    subject_key = normalize_course_key(subject)
    if subject_key:
        exact = store.get(
            where={"course_key": subject_key},
            include=["documents", "metadatas"],
        )
        exact_documents = [
            Document(page_content=content, metadata=metadata)
            for content, metadata in zip(exact.get("documents", []), exact.get("metadatas", []))
        ]
        exact_documents.sort(key=lambda item: (
            int(item.metadata.get("page", -1)),
            int(item.metadata.get("start_index", 0)),
        ))
        for document in exact_documents[:limit]:
            add(document)
        if results:
            return results

    # 짧은 자연어 질의가 필드명과 활동 구분까지 섞은 질의보다 과목 의미를 더 잘 보존합니다.
    query = (
        f"{department} {student_draft} {subject} "
        "관련 고등학교 교과 과목 학습 내용"
    ).strip()
    semantic_results = store.similarity_search(query, k=limit * 2)
    best_course_key = next(
        (str(document.metadata.get("course_key", "")).strip() for document in semantic_results
         if str(document.metadata.get("course_key", "")).strip()),
        "",
    )
    if best_course_key:
        exact = store.get(
            where={"course_key": best_course_key},
            include=["documents", "metadatas"],
        )
        best_course_documents = [
            Document(page_content=content, metadata=metadata)
            for content, metadata in zip(exact.get("documents", []), exact.get("metadatas", []))
        ]
        best_course_documents.sort(key=lambda item: (
            int(item.metadata.get("page", -1)), int(item.metadata.get("start_index", 0)),
        ))
        return best_course_documents[:limit]

    course_counts: dict[str, int] = {}
    for document in semantic_results:
        course_name = str(document.metadata.get("course_name", "")).strip()
        if not course_name or course_counts.get(course_name, 0) >= 2:
            continue
        add(document)
        course_counts[course_name] = course_counts.get(course_name, 0) + 1
        if len(results) >= limit:
            break
    return results
