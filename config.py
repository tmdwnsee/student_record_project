# config.py

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIRECTORY = PROJECT_ROOT / "data"
VECTORSTORE_DIRECTORY = PROJECT_ROOT / "vectorstores"

COLLEGE_PDF = DATA_DIRECTORY / "college_table.pdf"
GUIDELINE_PDF = DATA_DIRECTORY / "student_record_rule.pdf"

# 현재는 성균관대학교만 지원합니다. 대학을 추가할 때 data/의 PDF 경로를 등록합니다.
COLLEGE_GUIDES = {
    "성균관대학교": COLLEGE_PDF,
}

COLLEGE_VECTORSTORE = VECTORSTORE_DIRECTORY / "college"
GUIDELINE_VECTORSTORE = VECTORSTORE_DIRECTORY / "guideline"

COLLEGE_COLLECTION = "college_collection"
GUIDELINE_COLLECTION = "guideline_collection"

EMBEDDING_MODEL = "BAAI/bge-m3"
MODEL = "qwen3.5:9b"
OLLAMA_BASE_URL = "http://localhost:11434"
