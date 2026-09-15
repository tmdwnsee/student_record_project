# 자기평가보고서 초안 검토 서비스

자기평가보고서 초안을 입력하면 생기부 작성요령과 대학 모집요강 PDF에서 관련 근거를 검색하고, 원문의 의미를 유지하는 수정안과 수정 이유를 제안하는 Streamlit 앱입니다. 검색된 근거의 파일명과 페이지를 결과에 표시합니다. 생성된 문장은 최종 제출 전에 직접 검토해야 합니다.

희망 대학교와 학과를 입력하면 `data/`에 등록된 해당 대학 모집요강을 검색합니다. 현재는 성균관대학교만 선택할 수 있습니다. 확인된 학생부 평가영역과 반영비율을 표로 보여주고, 생기부 작성 전문가와 입학처 평가자의 관점에서 현재 초안의 부족한 점과 수정 방향을 제안합니다.

기존 생기부를 선택적으로 첨부해 활동 맥락을 참고할 수 있습니다. PDF와 UTF-8 TXT를 지원하며 최대 10MB입니다. PDF는 모집요강과 동일한 단계별 추출 파이프라인을 사용하므로 일반 텍스트 추출에 실패하면 PyMuPDF와 RapidOCR를 순서대로 시도합니다. 전체 텍스트를 약 5,000자씩 나눠 각 구간의 활동과 새 초안과의 관련성을 확인한 뒤, 관련성이 높은 최대 3개 구간의 요약과 원문을 최종 검토에 전달합니다. 결과의 **검토에 참고한 기존 생기부 구간**에서 실제 선택된 내용을 확인할 수 있습니다. 첨부 내용은 공식 근거로 인용하거나 검색 인덱스에 저장하지 않습니다.

## 프로젝트 구조

| 경로 | 역할 |
| --- | --- |
| `app.py` | Streamlit 입력과 결과 화면 |
| `ingestion/pdf_pipeline.py` | pdftotext, PyMuPDF, OCR 순서의 PDF 추출 |
| `ingestion/prepare_documents.py` | 추출된 PDF에 문서 유형을 지정하고 검색용 청크로 분할 |
| `ingestion/build_index.py` | 검색 인덱스 생성 및 갱신 |
| `storage/vectorstore.py` | 생성된 Chroma 검색 저장소 로딩 |
| `config.py` | 대학별 PDF 및 저장소 경로 설정 |
| `rag/retriever.py` | 두 PDF에서 관련 내용 검색 |
| `rag/chain.py` | 검색 근거를 이용한 수정안 생성 및 근거 검증 |
| `rag/attachment.py` | 기존 생기부 첨부 파일의 텍스트 추출 |
| `data/college_table.pdf` | 대학 모집요강 원본 |
| `data/student_record_rule.pdf` | 생기부 작성요령 원본 |
| `tests/test_rag.py` | 외부 모델 호출 없이 실행하는 회귀 테스트 |

## 설치 및 실행

Windows PowerShell과 Python 3.12를 기준으로 합니다. 프로젝트 폴더에서 실행하세요.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Ollama를 설치하고 로컬 LLM을 준비합니다.

```powershell
ollama pull qwen3.5:9b
```

답변 생성에는 Ollama의 `qwen3.5:9b`, 검색용 임베딩에는 CPU에서 실행되는 `BAAI/bge-m3`를 사용합니다. `BAAI/bge-m3`는 인덱스를 처음 생성할 때 Hugging Face에서 자동으로 내려받습니다. OpenAI API 키는 필요하지 않습니다.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

브라우저에서 <http://localhost:8501>에 접속해 초안을 입력하고 **분석하기**를 누릅니다. 종료하려면 터미널에서 `Ctrl+C`를 누르세요.

PDF가 처음 등록되거나 변경됐거나 임베딩 모델을 변경했다면 기존 `vectorstores/` 폴더를 지운 후 검색 인덱스를 다시 생성합니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m ingestion.build_index
```

## 테스트

```powershell
# 외부 모델 호출 없는 테스트
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

PDF는 `data/`에서 읽습니다. 검색 인덱스는 `vectorstores/`에 생성되며, PDF가 변경되면 해당 내용을 갱신합니다. `.env`, `.venv`, `vectorstores/`, `tmp/`는 `.gitignore`로 Git 업로드에서 제외합니다.

## GitHub에 변경사항 올리기

이 폴더는 이미 `origin` 원격 저장소와 연결되어 있습니다. 변경사항을 확인하고 커밋한 뒤 올리세요.

```powershell
git status
git add .
git commit -m "Update documentation"
git push origin main
```

`git status`에서 `.env`와 `.venv`가 추가 대상에 없는지 확인하세요.
