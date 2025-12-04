import streamlit as st
from src.resume_parser import parse_resume_file, parse_jd_text
from src.skills import SkillsExtractor, load_skills_dict
from src.matchers import TfidfMatcher, SbertMatcher, combined_ranking
from src.utils import redact_pii, explain_match, save_results_csv
import pandas as pd

st.set_page_config(layout="wide", page_title="Resume Matcher")
st.title("Resume Screening / Job Description Matcher")

st.sidebar.header("Settings")

ngram_min = st.sidebar.number_input(
    "Keyword Match: Minimum Word Group Size",
    min_value=1, max_value=5, value=1
)

ngram_max = st.sidebar.number_input(
    "Keyword Match: Maximum Word Group Size",
    min_value=1, max_value=5, value=2
)

sbert_model = st.sidebar.selectbox(
    "Job Fit Model",
    ["all-MiniLM-L6-v2"]
)

tfidf_weight = st.sidebar.slider(
    "Weight for Keyword Match",
    0.0, 1.0, 0.4
)

sbert_weight = st.sidebar.slider(
    "Weight for Job Fit",
    0.0, 1.0, 0.6
)

must_have_threshold = st.sidebar.slider(
    "Minimum Required Skills %",
    0, 100, 60
)

redact_display = st.sidebar.checkbox(
    "Hide personal info (email, phone)"
)


skills_path = "src/skills.json"
skills = load_skills_dict(skills_path)
skills_extractor = SkillsExtractor(skills)


uploaded_files = st.file_uploader("Upload resumes (PDF or txt)", accept_multiple_files=True, type=['pdf', 'txt'])
st.header("Job Description")
jd_text = st.text_area("Paste job description text here", height=200)
jd_file = st.file_uploader("Or upload JD text file (txt)", type=["txt"])
if jd_file and not jd_text:
    jd_text = jd_file.getvalue().decode('utf-8')

if st.button("Run matching"):
    if not uploaded_files:
        st.error("Upload at least one resume.")
        st.stop()
    if not jd_text or jd_text.strip() == "":
        st.error("Provide a job description.")
        st.stop()

    resumes = []
    for f in uploaded_files:
        parsed = parse_resume_file(f)
        resumes.append({"filename": f.name, "text": parsed})
    jd_parsed = parse_jd_text(jd_text)
    jd_skills = skills_extractor.extract_from_jd(jd_text)

    tfidf = TfidfMatcher(ngram_range=(ngram_min, ngram_max))
    sbert = SbertMatcher(model_name=sbert_model)
    corpus = [jd_text] + [r['text'] for r in resumes]
    tfidf.fit(corpus)
    sbert.fit(corpus)

    results = []
    for r in resumes:
        tfidf_score, tfidf_terms = tfidf.score_document(jd_text, r['text'])
        sbert_score, sbert_top = sbert.score_document(jd_text, r['text'], top_k_sentences=10)
        resume_skills = skills_extractor.extract_from_text(r['text'])
        must_have_met_pct = 0.0
        if jd_skills.get('must'):
            matched_must = set(jd_skills['must']).intersection(set(resume_skills['all']))
            must_have_met_pct = 100 * len(matched_must) / max(1, len(jd_skills['must']))
        combined = combined_ranking(tfidf_score, sbert_score, tfidf_weight, sbert_weight)
        explanation = explain_match(r['text'], jd_text, resume_skills, tfidf_terms, sbert_top)
        results.append({
            "filename": r['filename'],
            "text": r['text'],
            "tfidf_score": float(tfidf_score),
            "sbert_score": float(sbert_score),
            "combined_score": float(combined),
            "matched_skills": sorted(resume_skills['all']),
            "must_have_coverage_pct": must_have_met_pct,
            "top_sentences": [s for s,_ in sbert_top][:5],
            "explanation": explanation
        })

    df = pd.DataFrame(results).sort_values("combined_score", ascending=False).reset_index(drop=True)
    st.header("Ranking Results")
    st.write(f"TF-IDF weight: {tfidf_weight:.2f} — SBERT weight: {sbert_weight:.2f}")

    display_df = df[[
        "filename",
        "tfidf_score",
        "sbert_score",
        "combined_score",
        "must_have_coverage_pct"
    ]].copy()

    display_df = display_df.round(3)

    display_df = display_df.rename(columns={
        "filename": "Resume",
        "tfidf_score": "Words Match Score",
        "sbert_score": "Job Fit Score",
        "combined_score": "Final Fit Score",
        "must_have_coverage_pct": "Required Skills %"
    })

    st.dataframe(display_df)

    export_df = df.rename(columns={
        "filename": "Resume",
        "tfidf_score": "Words Match Score",
        "sbert_score": "Job Fit Score",
        "combined_score": "Final Fit Score",
        "must_have_coverage_pct": "Required Skills %"
    }).copy()

    csv_bytes = save_results_csv(export_df)
    st.download_button("Download results CSV", data=csv_bytes, file_name="resume_matching_results.csv")

    st.header("Detailed Results")
    for idx, row in df.iterrows():
        st.subheader(f"{idx+1}. {row['filename']} — Final Fit Score {row['combined_score']:.3f}")

        display_text = redact_pii(row['text']) if redact_display else row['text']
        left, right = st.columns([1,1])

        with left:
            st.markdown("**Matched Skills**")
            st.write(", ".join(row['matched_skills']) if row['matched_skills'] else "None found")

            st.markdown("**Required Skills %**")
            st.write(f"{row['must_have_coverage_pct']:.1f}%")

            st.markdown("**Most Relevant Sentences (Job Fit)**")
            for s in row['top_sentences']:
                st.write(f"- {s}")

        with right:
            st.markdown("**Explanation (auto-generated)**")
            st.write(row['explanation'])

            st.markdown("**Resume Text (Redacted View)**")
            st.text(display_text[:2000] + ("..." if len(display_text) > 2000 else ""))
