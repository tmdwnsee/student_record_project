# 생기부 초안 검토 서비스

학교생활기록부 초안을 입력하면 생기부 작성요령과 대학 모집요강 PDF에서 관련 근거를 검색하고, 원문의 의미를 유지하는 수정안과 수정 이유를 제안하는 Streamlit 앱입니다. 검색된 근거의 파일명과 페이지를 결과에 표시합니다. 생성된 문장은 최종 제출 전에 직접 검토해야 합니다.

## 프로젝트 구조

| 경로 | 역할 |
| --- | --- |
| `app.py` | Streamlit 화면과 단계별 콘솔 실행 |
| `rag/loader.py` | PDF 로딩, 페이지 확인, 문서 분할 |
| `rag/vectorstore.py` | OpenAI 임베딩과 Chroma 검색 저장소 관리 |
| `rag/retriever.py` | 두 PDF에서 관련 내용 검색 |
| `rag/chain.py` | 검색 근거를 이용한 수정안 생성 및 근거 검증 |
| `data/college_table.pdf` | 대학 모집요강 원본 |
| `data/student_record_rule.pdf` | 생기부 작성요령 원본 |
| `tests/test_rag.py` | API 호출 없이 실행하는 회귀 테스트 |

## 설치 및 실행

Windows PowerShell과 Python 3.12를 기준으로 합니다. 프로젝트 폴더에서 실행하세요.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

프로젝트 루트에 `.env` 파일을 만들고 본인의 API 키를 설정합니다.

```dotenv
OPENAI_API_KEY=본인의_API_키
OPENAI_MODEL=gpt-4o-mini
```

`OPENAI_MODEL`은 선택 사항입니다. 기본값은 `gpt-4o-mini`이며, 임베딩에는 `text-embedding-3-small`을 사용합니다. 최초 분석 시 PDF 내용을 임베딩하고, 분석할 때 검색 및 응답 생성을 위해 OpenAI API를 호출하므로 사용료가 발생할 수 있습니다. 학생 초안과 검색된 PDF 내용도 API로 전달됩니다.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

브라우저에서 <http://localhost:8501>에 접속해 초안을 입력하고 **분석하기**를 누릅니다. 종료하려면 터미널에서 `Ctrl+C`를 누르세요.

## 콘솔 실행과 테스트

```powershell
# PDF 로딩과 문서 분할 확인: API 키 불필요
.\.venv\Scripts\python.exe -X utf8 app.py --stage pdf

# 검색 저장소 생성: API 키 필요
.\.venv\Scripts\python.exe -X utf8 app.py --stage index

# 관련 문서 검색
.\.venv\Scripts\python.exe -X utf8 app.py --stage retrieve --draft "검토할 학생 활동 내용"

# 수정안 생성 및 선택적으로 JSON 저장
.\.venv\Scripts\python.exe -X utf8 app.py --stage all --draft "검토할 학생 활동 내용" --output tmp/rag_review.json

# API 호출 없는 테스트
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

PDF는 `data/`에서 읽습니다. 검색 인덱스는 `vectorstores/`에 생성되며, PDF가 변경되면 해당 내용을 갱신합니다. `.env`, `.venv`, `vectorstores/`, `tmp/`는 `.gitignore`로 Git 업로드에서 제외합니다. API 키를 저장소에 추가하지 마세요.

## GitHub에 변경사항 올리기

이 폴더는 이미 `origin` 원격 저장소와 연결되어 있습니다. 변경사항을 확인하고 커밋한 뒤 올리세요.

```powershell
git status
git add .
git commit -m "Update documentation"
git push origin main
```

`git status`에서 `.env`와 `.venv`가 추가 대상에 없는지 확인하세요.
