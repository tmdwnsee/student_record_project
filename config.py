# config.py

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIRECTORY = PROJECT_ROOT / "data"
VECTORSTORE_DIRECTORY = PROJECT_ROOT / "vectorstores"

SKKU_PDF = DATA_DIRECTORY / "성균관대학교_모집요강.pdf"
DONGGUK_PDF = DATA_DIRECTORY / "동국대학교_모집요강.pdf"
KYUNGHEE_PDF = DATA_DIRECTORY / "경희대학교_모집요강.pdf"
UOS_PDF = DATA_DIRECTORY / "서울시립대학교_모집요강.pdf"
MYONGJI_PDF = DATA_DIRECTORY / "명지대학교_모집요강.pdf"
KONKUK_PDF = DATA_DIRECTORY / "건국대학교_모집요강.pdf"
CATHOLIC_PDF = DATA_DIRECTORY / "가톨릭대학교_모집요강.pdf"
GUIDELINE_PDF = DATA_DIRECTORY / "student_record_rule.pdf"

COLLEGE_GUIDES = {
    "성균관대학교": SKKU_PDF,
    "동국대학교": DONGGUK_PDF,
    "경희대학교": KYUNGHEE_PDF,
    "서울시립대학교": UOS_PDF,
    "명지대학교": MYONGJI_PDF,
    "건국대학교": KONKUK_PDF,
    "가톨릭대학교": CATHOLIC_PDF,
}

# PDF 뷰어 기준(표지 포함, 1부터 시작) 평가기준 페이지입니다.
COLLEGE_GUIDE_PAGES = {
    "성균관대학교": (72,),
    "동국대학교": (93, 94),
    "경희대학교": (63, 64),
    "서울시립대학교": (46, 47),
    "명지대학교": (98,),
    "건국대학교": (69, 70),
    "가톨릭대학교": (101,),
}

# 복잡한 다단 표는 좌표 정렬보다 PDF 내부 읽기 순서가 열 관계를 더 잘 보존합니다.
COLLEGE_NATURAL_TEXT_ORDER = {"경희대학교"}

COLLEGE_VECTORSTORES = {
    university: VECTORSTORE_DIRECTORY / pdf_path.stem
    for university, pdf_path in COLLEGE_GUIDES.items()
}
COLLEGE_COLLECTIONS = {
    university: f"college_{index}"
    for index, university in enumerate(COLLEGE_GUIDES, start=1)
}

# 이전 import 경로와 작성요령 관련 코드의 호환성을 유지합니다.
COLLEGE_PDF = SKKU_PDF
COLLEGE_VECTORSTORE = COLLEGE_VECTORSTORES["성균관대학교"]
COLLEGE_COLLECTION = COLLEGE_COLLECTIONS["성균관대학교"]
GUIDELINE_VECTORSTORE = VECTORSTORE_DIRECTORY / "guideline"

GUIDELINE_COLLECTION = "guideline_collection"

EMBEDDING_MODEL = "BAAI/bge-m3"
MODEL = "qwen3.5:9b"
OLLAMA_BASE_URL = "http://localhost:11434"
