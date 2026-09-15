# rag/retriever.py

from langchain_chroma import Chroma
from langchain_core.documents import Document


COLLEGE_QUERY = """
학생부종합전형
서류평가
평가 영역 및 반영 비율
평가요소
학업역량
탐구역량
잠재역량
문제해결능력
기준
"""


GUIDELINE_QUERY = """
학교생활기록부 작성 원칙
기재 시 주의사항
교사가 직접 관찰 평가한 내용
활동 기록
구체적 사실
표현 금지사항
"""


def retrieve_college_context(
    store: Chroma,
    student_draft: str,
) -> list[Document]:

    query = f"""
{COLLEGE_QUERY}

검토할 학생 활동:
{student_draft}
"""

    return store.similarity_search(
        query,
        k=5,
    )


def retrieve_guideline_context(
    store: Chroma,
    student_draft: str,
) -> list[Document]:

    query = f"""
{GUIDELINE_QUERY}

기록하려는 학생 활동:
{student_draft}
"""

    return store.similarity_search(
        query,
        k=5,
    )