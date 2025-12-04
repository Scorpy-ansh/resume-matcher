# src/utils.py
import re
import pandas as pd

PHONE_RE = re.compile(r'(\+?\d{1,3}[\s-]?)?(\d{10}|\d{5}[\s-]\d{5}|\d{3}[\s-]\d{3}[\s-]\d{4})')
EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+', re.I)

def redact_pii(text: str) -> str:
    t = PHONE_RE.sub('[PHONE]', text)
    t = EMAIL_RE.sub('[EMAIL]', t)
    return t

def explain_match(resume_text: str, jd_text: str, resume_skills: dict, tfidf_terms: list, sbert_top_sentences: list) -> str:
    skills = resume_skills.get('all', [])[:8]
    top_terms = [t for t, _ in tfidf_terms[:5]]
    top_sent = sbert_top_sentences[0][0] if sbert_top_sentences else ""
    reasons = []
    if skills:
        reasons.append("Contains skills: " + ", ".join(skills[:5]))
    if top_terms:
        reasons.append("Keyword overlap: " + ", ".join(top_terms[:5]))
    if top_sent:
        reasons.append("Relevant sentence: " + (top_sent[:160] + ("..." if len(top_sent) > 160 else "")))
    return "  ".join(reasons[:3])

def save_results_csv(df):
    csv = df.to_csv(index=False)
    return csv.encode('utf-8')
