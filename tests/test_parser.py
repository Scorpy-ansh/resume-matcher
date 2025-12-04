# tests/test_parser.py
from resume_parser import parse_jd_text
def test_parse_jd_text_basic():
    jd = "Required: Must have Python and SQL.\nNice to have: Docker."
    parsed = parse_jd_text(jd)
    assert isinstance(parsed, dict)
    assert 'raw' in parsed
