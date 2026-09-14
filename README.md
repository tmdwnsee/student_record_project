# 두 PDF 기반 RAG 테스트

두 PDF의 로딩 → chunking → Embedding/Chroma → 개별 검색 → LLM 구조화 응답 →
Streamlit 화면까지 구현했습니다. 다중 대학, 로그인, 사용자 DB, Multi-Agent는 포함하지 않습니다.

## 실행 (Windows PowerShell / Python 3.12)

기존 `.venv`가 없다면 먼저 `py -3.12 -m venv .venv`를 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

브라우저에서 http://localhost:8501 에 접속해 초안을 입력하고 **분석하기**를 누릅니다.
`.env`는 프로젝트 경로에서 읽으며 기존 파일을 보존했습니다. 다음 설정을 사용합니다.

```dotenv
OPENAI_API_KEY=본인의_API_키
OPENAI_MODEL=gpt-4o-mini
```

`OPENAI_MODEL`은 선택 사항이며 기본값은 `gpt-4o-mini`입니다.
최초 분석 시 두 PDF의 chunk를 `text-embedding-3-small`로 임베딩합니다.
검색 시 query 임베딩, 답변 생성 시 LLM 호출이 발생합니다. API 사용료가 발생할 수 있습니다.
PDF chunk, 검색 query, 학생 초안과 검색 근거가 OpenAI API에 전달됩니다.
`.env`와 vectorstores는 Git에서 제외합니다. 학생 초안은 vectorstore에 저장하지 않습니다.
앱은 `data/`의 두 PDF를 읽습니다.

## 실제 검증 결과

- `college_table.pdf`: 77개 Document
- `student_record_rule.pdf`: 223개 Document
- 두 파일의 모든 Document에 `source`와 정수형 `page`가 있는지 검사 완료
- 대학 PDF의 metadata `page=69~72` 텍스트 출력 완료
- 평가비율 표가 있는 원본 페이지를 이미지로 확인하고 추출 텍스트와 대조 완료

| metadata page (0부터) | PDF 파일 내 순서 (1부터) | page_label / 인쇄 쪽수 | 내용 |
| --- | --- | --- | --- |
| 69 | 70 | 68 | 체육 실적 기준 |
| 70 | 71 | 69 | 무용학과 전공실기 복장 안내 |
| 71 | 72 | 70 | 학생부종합전형 서류평가 방법 및 평가비율 표 |
| 72 | 73 | 71 | 서류평가 절차 |

따라서 요청의 '70페이지'는 인쇄된 쪽수이며, 해당 페이지의 실제 loader metadata는
`page=71`, `page_label='70'`입니다. PDF 파일에서 순서대로 세면 72번째입니다.
평가표에는 학업역량 40%, 탐구역량 40%, 잠재역량 20%가 있습니다.
이 대응은 실제 파일을 확인한 결과이며 다른 문서에 같은 보정값을 적용하면 안 됩니다.
이 파일도 표지의 page_label은 77, 78이므로 전체 페이지에 단순한 공식을 적용하지 않습니다.

텍스트에서 일부 머리글이 깨지고 표의 셀 경계가 평문으로 합쳐집니다.
현재 단계에서는 평가비율과 관련 항목이 추출되는 것을 확인했습니다.
표 구조 보존과 검색 정확도는 후속 단계에서 별도로 확인해야 합니다.

## 노트북에서 개별 확인

필요하면 프로젝트 루트에서 실행하는 노트북의 셀에 다음을 사용할 수 있습니다.

```python
from rag.loader import DATA_DIRECTORY, load_pdf, print_document_metadata, inspect_pdf_page

college_documents = load_pdf(DATA_DIRECTORY / "college_table.pdf")
guideline_documents = load_pdf(DATA_DIRECTORY / "student_record_rule.pdf")
print(len(college_documents))
print(len(guideline_documents))
print_document_metadata(college_documents)
print_document_metadata(guideline_documents)

for page in range(69, 73):
    inspect_pdf_page(college_documents, page)

# 원본 metadata를 모두 출력하려면:
# for document in college_documents:
#     print(document.metadata)
```

loader는 [LangChain 공식 PyPDFLoader API](https://reference.langchain.com/python/langchain-community/document_loaders/pdf/PyPDFLoader)의
`langchain_community.document_loaders.PyPDFLoader`를 사용합니다.

## 4단계: chunking

`rag/loader.py`의 `split_documents()`가 두 PDF를 각각 분할합니다.
[RecursiveCharacterTextSplitter](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter)를
`chunk_size=700`, `chunk_overlap=100`으로 사용합니다. 길이는 토큰이 아닌 문자 수이며,
실제 겹치는 길이는 문단·줄 경계에 따라 100자보다 작을 수 있습니다. 페이지끼리는 합치지 않습니다.
텍스트가 없는 페이지는 chunk를 만들지 않습니다.

```python
from rag.loader import split_documents, validate_chunk_metadata

college_chunks = split_documents(college_documents)
guideline_chunks = split_documents(guideline_documents)
validate_chunk_metadata(college_documents, college_chunks)
validate_chunk_metadata(guideline_documents, guideline_chunks)
print(college_chunks[0].metadata)
print(guideline_chunks[0].metadata)
```

`app.py`를 실행하면 chunk 수, 첫 chunk metadata, 원본 metadata 보존 검사 결과와
평가표 `page=71`의 모든 chunk 본문을 출력합니다. `source`, `page`, `page_label`을 포함한
전체 metadata를 원본과 비교합니다. 표의 의미를 고려한 분할은 아니므로 출력된 경계도 확인해야 합니다.
분할한 chunk는 5단계에서 Embedding + Chroma 저장에 사용합니다.

실제 실행 결과: 대학 PDF 176개, 작성요령 PDF 517개 chunk가 생성되었고 모든 chunk의
metadata가 원본과 일치했습니다. 평가표 페이지는 681자와 612자의 두 chunk로 나뉩니다.
첫 chunk에 학업역량 40%, 탐구역량 40%, 잠재역량 20%가 함께 포함되고,
둘째 chunk에 평가요소 및 항목이 포함되는 것을 확인했습니다.

## 5~9단계 구조

| 파일 | 역할 |
| --- | --- |
| `rag/loader.py` | 페이지 로딩, metadata 점검, 700/100 chunking |
| `rag/vectorstore.py` | API 설정 확인, 두 저장소 생성 및 중복 방지 |
| `rag/retriever.py` | 목적을 나눈 query, 각각 k=5 검색, 검색 결과 출력 |
| `rag/chain.py` | 근거별 context, 시스템 프롬프트, Pydantic 출력과 인용 검증 |
| `app.py` | Streamlit UI 및 단계별 콘솔 실행 |

대학은 `vectorstores/college`의 `college_collection`, 작성요령은
`vectorstores/guideline`의 `guideline_collection`에 저장합니다.
모델명·내용·metadata의 SHA-256을 chunk ID로 사용해 이미 저장된 chunk는 다시 임베딩하지 않습니다.
PDF가 바뀌면 새 chunk 저장 성공 후 이전 chunk를 제거합니다. 중간 실패 시 재실행하면 누락분을 추가합니다.
UI는 PDF 수정 시간과 크기를 기준으로 저장소 준비 결과를 캐시합니다.
테스트용 로컬 앱이므로 인덱스를 갱신하는 콘솔 명령과 UI 분석을 동시에 실행하지 마세요.

작성요령은 기재 원칙과 관찰·표현 기준, 대학은 서류평가 영역·비율·역량을 검색합니다.
두 query에 학생 활동을 덧붙이며 별도 LLM query rewrite는 사용하지 않습니다.
검색된 두 문서군을 구분해 LLM에 제공하고, Pydantic `ReviewSelection`으로 수정안·이유·근거 번호를 받습니다.
코드가 선택된 근거의 원문과 metadata를 가져와 최종 `ReviewResult`를 구성합니다.
수정 이유 하단에도 선택한 실제 파일명과 metadata page를 코드가 붙입니다.
존재하지 않는 근거 번호나 원문과 불일치하는 인용이 있으면 결과 표시를 중단합니다.
원문은 사용자 입력 그대로 유지합니다. 활동을 지어내지 않고, 근거 부족은 주의사항에 적도록 지시합니다.
인용 일치 검사는 문장 전체의 해석이나 수정안의 타당성까지 보증하지 않으므로 최종 검토가 필요합니다.

## 단계별 실행

```powershell
# 1~4단계: API 호출 없이 PDF와 chunk 확인 (기본 실행도 동일)
.\.venv\Scripts\python.exe -X utf8 app.py --stage pdf

# 5단계: 저장소 생성 또는 변경분 반영
.\.venv\Scripts\python.exe -X utf8 app.py --stage index

# 5~6단계: 저장소 준비 후 두 검색 결과의 본문/source/page 확인
.\.venv\Scripts\python.exe -X utf8 app.py --stage retrieve

# 1~8단계: 샘플 초안으로 전체 실행, 결과 JSON 저장
.\.venv\Scripts\python.exe -X utf8 app.py --stage all --output tmp/rag_review.json

# 사용자 초안으로 실행
.\.venv\Scripts\python.exe -X utf8 app.py --stage all --draft "검토할 학생 활동 내용"

# API 호출 없는 회귀 테스트
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

CLI와 UI 오류는 가능한 한 단계명을 표시합니다. 인증 오류 원문에 키가 포함될 수 있어
SDK traceback을 화면에 그대로 표시하지 않습니다. API 연결 실패 시 네트워크, API 키,
계정 잔액과 모델 접근 권한을 확인하세요.

구현 참고: [Chroma 공식 연동 문서](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma),
[OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).

## 전체 실행 검증 결과 (2026-09-12)

- 실제 OpenAI 임베딩 저장: 대학 176개, 작성요령 517개
- 재실행 시 두 collection 모두 신규 0개로 기존 임베딩 재사용 확인
- 샘플 초안의 대학 검색 첫 결과: 평가비율 표 `page=71`
- 작성요령과 대학 각각 5개 검색 결과 출력 확인
- 실제 Streamlit 분석 버튼 → 검색 → LLM → 결과 표시까지 AppTest 실행 성공
- 최종 응답에 작성요령 `page=23`, 대학 `page=71` 근거가 포함됨
- 원문, 수정안, 수정 이유, 두 문서 근거, 주의사항의 6개 화면 영역 확인
- API를 호출하지 않는 회귀 테스트 4개 통과, `pip check` 의존성 충돌 없음

실행 결과를 파일로 저장하려면 `--output tmp/rag_review.json` 옵션을 사용합니다.
샘플 수정안: “데이터 분석 프로젝트를 진행하며 Python을 활용해 데이터를 분석함.”
이는 한 가지 샘플의 실행 검증이며, 다른 초안의 검색 정확도는 추가 사례로 평가해야 합니다.
