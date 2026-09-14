"""업로드한 기존 생기부에서 검토용 텍스트를 추출합니다."""

from io import BytesIO

from pypdf import PdfReader

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_CONTEXT_CHARS = 20_000


def extract_context(filename: str, content: bytes) -> str:
    if not content or len(content) > MAX_FILE_BYTES:
        raise ValueError("첨부 파일은 비어 있지 않은 10MB 이하 파일이어야 합니다.")
    suffix = filename.lower().rsplit(".", 1)[-1]
    try:
        if suffix == "pdf":
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted:
                raise ValueError("암호가 걸린 PDF는 사용할 수 없습니다.")
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif suffix == "txt":
            text = content.decode("utf-8-sig")
        else:
            raise ValueError("PDF 또는 UTF-8 TXT 파일을 첨부하세요.")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("첨부 파일에서 텍스트를 읽지 못했습니다. 텍스트가 포함된 PDF 또는 UTF-8 TXT를 사용하세요.") from error
    text = text.strip()
    if not text:
        raise ValueError("첨부 파일에 추출할 텍스트가 없습니다. 스캔 이미지 PDF는 지원하지 않습니다.")
    return text[:MAX_CONTEXT_CHARS]
