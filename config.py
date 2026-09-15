# config.py

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIRECTORY = PROJECT_ROOT / "data"
VECTORSTORE_DIRECTORY = PROJECT_ROOT / "vectorstores"

COLLEGE_PDF = DATA_DIRECTORY / "college_table.pdf"
GUIDELINE_PDF = DATA_DIRECTORY / "student_record_rule.pdf"

COLLEGE_VECTORSTORE = VECTORSTORE_DIRECTORY / "college"
GUIDELINE_VECTORSTORE = VECTORSTORE_DIRECTORY / "guideline"

COLLEGE_COLLECTION = "college_collection"
GUIDELINE_COLLECTION = "guideline_collection"

EMBEDDING_MODEL = "text-embedding-3-small"


def check_api_key():
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError(
            ".env에 OPENAI_API_KEY를 설정하세요."
        )