# src/parser.py
import pdfplumber
import io
import re
import nltk
nltk.download('punkt', quiet=True)

def _extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    text_lines = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_lines.append(page_text)
    return "\n".join(text_lines)

def _clean_text(text: str) -> str:
    text = text.replace('\r', '\n')
    text = re.sub(r'\n\s+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def parse_resume_file(uploaded_file) -> str:
    filename = uploaded_file.name.lower()
    raw = None
    try:
        if filename.endswith('.pdf'):
            raw = _extract_text_from_pdf_bytes(uploaded_file.getvalue())
        else:
            raw = uploaded_file.getvalue().decode('utf-8', errors='ignore')
    except Exception:
        try:
            raw = uploaded_file.read().decode('utf-8', errors='ignore')
        except Exception:
            raw = ""
    return _clean_text(raw or "")

def parse_jd_text(jd_text: str) -> dict:
    text = _clean_text(jd_text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    responsibilities = []
    must = []
    nice = []
    for l in lines:
        low = l.lower()
        if any(k in low for k in ['responsibilit', 'responsible', 'responsibilities', 'you will']):
            responsibilities.append(l)
        if any(k in low for k in ['must', 'required', 'mandatory', 'required skills', 'must have']):
            must.append(l)
        elif any(k in low for k in ['prefer', 'nice to have', 'nice-to-have', 'optional', 'desired']):
            nice.append(l)
    if not responsibilities:
        responsibilities = [l for l in lines if 'skill' in l.lower() or 'experience' in l.lower()][:10]
    return {"raw": text, "responsibilities": responsibilities, "must": must, "nice": nice}
