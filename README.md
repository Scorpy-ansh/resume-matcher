# Resume Screening / Job Description Matcher

Quick setup:
1. Create venv and activate:
   python3.10 -m venv venv
   source venv/bin/activate   # Windows: .\venv\Scripts\Activate.ps1
2. Install:
   pip install -r requirements.txt
3. Download models:
   python -m spacy download en_core_web_sm
   python -c "import nltk; import nltk as n; n.download('punkt'); n.download('stopwords')"
4. Run app:
   streamlit run app.py

Files:
- app.py (Streamlit)
- src/ (parser, skills, matchers, utils)
- sample_data/ (5 resumes, 2 JDs)
- tests/ (pytest)
- evaluations/evaluate.py

Notes:
- Edit src/skills.json to change skills dictionary.
- PDF OCR not included (pdfplumber handles text PDFs).
