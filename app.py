"""Streamlit UI와 단계별 콘솔 점검 진입점."""

import argparse
import sys

from dotenv import load_dotenv

from rag.loader import (
    DATA_DIRECTORY,
    PROJECT_ROOT,
    inspect_pdf_page,
    load_pdf,
    print_document_metadata,
    split_documents,
    validate_chunk_metadata,
)


def inspect_pdfs() -> None:
    # 기존 환경 변수를 덮어쓰지 않습니다. 현재 단계는 API 키 없이 실행됩니다.
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    print("[2단계] 대학 모집요강 로딩")
    college_documents = load_pdf(DATA_DIRECTORY / "college_table.pdf")
    print("대학 모집요강 페이지 수:", len(college_documents))
    print_document_metadata(college_documents)

    print("\n[2단계] 생기부 작성요령 로딩")
    guideline_documents = load_pdf(DATA_DIRECTORY / "student_record_rule.pdf")
    print("생기부 작성요령 페이지 수:", len(guideline_documents))
    print_document_metadata(guideline_documents)

    print("\n[3단계] 대학 PDF metadata page=69~72 확인")
    # 특정 번호를 평가표 페이지라고 가정하지 않고 실제 내용을 출력합니다.
    for page in range(69, 73):
        inspect_pdf_page(college_documents, page)

    print("\n[4단계] chunking (700자 / overlap 100자)")
    college_chunks = split_documents(college_documents)
    guideline_chunks = split_documents(guideline_documents)
    validate_chunk_metadata(college_documents, college_chunks)
    validate_chunk_metadata(guideline_documents, guideline_chunks)
    print("대학 모집요강 chunk 수:", len(college_chunks))
    print("생기부 작성요령 chunk 수:", len(guideline_chunks))
    print("대학 첫 chunk metadata:", college_chunks[0].metadata)
    print("작성요령 첫 chunk metadata:", guideline_chunks[0].metadata)
    print("모든 chunk의 원본 metadata 보존 확인 완료")

    # 평가표가 chunk 경계에서 어떻게 나뉘는지 전체 내용을 확인합니다.
    print("\n[4단계] 평가표 page=71의 모든 chunk")
    evaluation_chunks = [
        chunk for chunk in college_chunks if chunk.metadata["page"] == 71
    ]
    for index, chunk in enumerate(evaluation_chunks, start=1):
        print(f"\nCHUNK {index} / 길이 {len(chunk.page_content)}자")
        print("METADATA:", chunk.metadata)
        print(chunk.page_content)


def run_cli() -> None:
    from rag.chain import SAMPLE_DRAFT, generate_review
    from rag.retriever import (
        print_search_results, retrieve_college_context, retrieve_guideline_context,
    )
    from rag.vectorstore import build_vectorstores

    parser = argparse.ArgumentParser(description="두 PDF 기반 RAG 단계별 테스트")
    parser.add_argument("--stage", choices=["pdf", "index", "retrieve", "all"], default="pdf")
    parser.add_argument("--draft", default=SAMPLE_DRAFT)
    parser.add_argument("--output", type=str, help="최종 응답 JSON 저장 경로 (선택)")
    args = parser.parse_args()
    if args.stage in ("pdf", "all"):
        inspect_pdfs()
    if args.stage == "pdf":
        return
    if not args.draft.strip():
        raise ValueError("자기평가보고서 초안을 입력하세요.")
    college_store, guideline_store = build_vectorstores()
    if args.stage == "index":
        return
    print("\n[6단계] 대학 모집요강 검색 결과")
    college_results = retrieve_college_context(college_store, args.draft)
    print_search_results(college_results)
    print("\n[6단계] 생기부 작성요령 검색 결과")
    guideline_results = retrieve_guideline_context(guideline_store, args.draft)
    print_search_results(guideline_results)
    if args.stage == "retrieve":
        return
    result = generate_review(args.draft, guideline_results, college_results)
    print("\n[7~8단계] 최종 구조화 응답 (근거 원문 대조 완료)")
    print(result.model_dump_json(indent=2))
    if args.output:
        from pathlib import Path
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def run_streamlit() -> None:
    import streamlit as st
    from rag.chain import review_draft
    from rag.vectorstore import build_vectorstores

    st.set_page_config(page_title="자기평가보고서 초안 검토", page_icon="📝")
    st.title("자기평가보고서 초안 검토")
    st.caption("작성요령과 대학 모집요강의 근거를 바탕으로 초안을 검토합니다.")

    @st.cache_resource(show_spinner=False)
    def get_stores(pdf_versions):
        # PDF가 바뀌면 캐시도 갱신합니다. 학생 초안은 저장소에 넣지 않습니다.
        return build_vectorstores()

    student_draft = st.text_area("자기평가보고서 초안", height=160, placeholder="검토할 활동 기록을 입력하세요.")
    if st.button("분석하기", type="primary"):
        st.session_state.pop("review_result", None)
        if not student_draft.strip():
            st.warning("자기평가보고서 초안을 입력하세요.")
        else:
            try:
                with st.spinner("관련 근거를 찾아 초안을 검토하고 있습니다..."):
                    pdf_versions = tuple(
                        (path.stat().st_mtime_ns, path.stat().st_size)
                        for path in (DATA_DIRECTORY / "college_table.pdf", DATA_DIRECTORY / "student_record_rule.pdf")
                    )
                    college_store, guideline_store = get_stores(pdf_versions)
                    st.session_state["review_result"] = review_draft(
                        student_draft, college_store, guideline_store,
                    )
            except (ValueError, RuntimeError) as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"분석 준비 실패 ({type(error).__name__}). PDF 파일과 실행 환경을 확인하세요.")

    result = st.session_state.get("review_result")
    if result is not None and result.original_text == student_draft:
        for title, value in (
            ("원문", result.original_text), ("수정안", result.revised_text),
            ("수정 이유", result.revision_reason),
        ):
            st.subheader(title)
            st.write(value)
        for title, evidence in (
            ("생기부 작성요령 근거", result.guideline_evidence),
            ("대학 모집요강 근거", result.college_evidence),
        ):
            st.subheader(title)
            if not evidence:
                st.write("검색 결과에서 관련 근거를 확인하지 못했습니다.")
            for item in evidence:
                st.caption(f"{item.source} · page: {item.page} (0부터 시작하는 PDF 인덱스)")
                st.write(item.content)
        st.subheader("주의사항")
        st.write(result.caution or "추가 주의사항 없음")


if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx(suppress_warning=True) is not None:
        run_streamlit()
    else:
        try:
            run_cli()
        except Exception as error:
            # 인증 실패 시 SDK 예외 원문에 API 키가 포함될 수 있어 traceback은 출력하지 않습니다.
            if isinstance(error, (ValueError, RuntimeError, FileNotFoundError)):
                print(str(error), file=sys.stderr)
            else:
                print(f"[실행 실패] {type(error).__name__}: API 설정 및 네트워크를 확인하세요.", file=sys.stderr)
            sys.exit(1)
