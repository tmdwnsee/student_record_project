from pathlib import Path
from pdf_pipeline import process_pdf, summarize_page_failures, targeted_pages_for_document

BASE = Path('/mnt/data')


def test_student_pdf():
    p = BASE / 'SYN_0123_미디어커뮤니케이션학과_박준호.pdf'
    results = process_pdf(p, document_type='student_record')
    print('student', summarize_page_failures(results))
    assert results and any('박준호' in r.text for r in results)


def test_skku_2027_target_page():
    p = BASE / '[성균관대학교] 2027학년도 수시 모집요강.pdf'
    results = process_pdf(p, document_type='skku_2027', expected_terms_by_page=targeted_pages_for_document('skku_2027'))
    page72 = results[71]
    print('skku p72', page72.method, page72.quality.score)
    assert '학업역량' in page72.text
    assert '탐구역량' in page72.text
    assert '잠재역량' in page72.text


def test_school_guide_detects_ocr_needed():
    p = BASE / '2026_학교생활기록부_기재요령_주요_개정사항(고등학교).pdf'
    results = process_pdf(p, document_type='school_record_guide_2026')
    summary = summarize_page_failures(results)
    print('guide', summary)
    # 이 테스트 환경에는 RapidOCR가 없을 수 있으므로, 이미지형 PDF라는 사실만 검증
    assert summary['total_pages'] == 52
    assert summary['review_pages'] > 0
