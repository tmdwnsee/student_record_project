"""작성요령과 대학 평가 기준을 서로 다른 목적의 query로 검색합니다."""

from langchain_chroma import Chroma
from langchain_core.documents import Document

COLLEGE_QUERY = "학생부종합전형 서류평가 평가 영역 및 반영 비율 평가요소 학업역량 탐구역량 잠재역량 문제해결능력 기준"
GUIDELINE_QUERY = "학교생활기록부 작성 원칙 기재 시 주의사항 교사가 직접 관찰 평가한 내용 활동 기록 구체적 사실 표현 금지사항"


def retrieve_college_context(store: Chroma, student_draft: str) -> list[Document]:
    query = f"{COLLEGE_QUERY}\n검토할 학생 활동: {student_draft}"
    return store.as_retriever(search_kwargs={"k": 5}).invoke(query)


def retrieve_guideline_context(store: Chroma, student_draft: str) -> list[Document]:
    query = f"{GUIDELINE_QUERY}\n기록하려는 학생 활동: {student_draft}"
    return store.as_retriever(search_kwargs={"k": 5}).invoke(query)


def print_search_results(results: list[Document]) -> None:
    for document in results:
        print("PAGE:", document.metadata.get("page"))
        print("PAGE LABEL:", document.metadata.get("page_label"))
        print("SOURCE:", document.metadata.get("source"))
        print(document.page_content[:500])
        print("-" * 50)
