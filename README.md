# 📄 Resume Matcher — AI-Powered Resume & Job Description Matching

[![Open Deployed App](https://img.shields.io/badge/Live%20Demo-Resume%20Matcher-blue)](https://resume-matcher88.streamlit.app/)

A smart AI-based Resume Matcher that compares candidate resumes against job descriptions using both **keyword similarity (TF-IDF)** and **semantic similarity (SBERT)**.  
Built with **Streamlit**, **Python**, **scikit-learn**, and **sentence-transformers**.

This tool ranks resumes, highlights relevant skills, extracts key sentences, and provides detailed reasoning for match scores.

---

# 🚀 Live Demo

### 👉 **Open the deployed app:**  
https://resume-matcher88.streamlit.app/

### 📱 Scan to open:

If `assets/deployed_qr.png` exists in the repo, it will appear here:

![QR Code](assets/deployed_qr.png)

---

# ⭐ Features

### ✔ Upload multiple resumes (PDF or TXT)
- Automatic text extraction  
- Optional PII redaction (emails, phone numbers)  

### ✔ Analyze any Job Description
- Detects responsibilities & skills  
- Identifies **must-have** vs **nice-to-have** skills  

### ✔ Dual Matching Engine  
**1. Words Match Score (TF-IDF)**  
Measures keyword relevance.

**2. Job Fit Score (SBERT)**  
Semantic similarity — understands meaning.

**3. Final Fit Score**  
Weighted combination of both methods.

### ✔ Skills & Explainability
- Extracted skills  
- Required skills %  
- Top relevant sentences  
- Auto-generated explanation  

### ✔ Export Support
- Download ranked results as CSV  

---

# 🧠 How Matching Works

### 🔹 **Words Match Score**
Keyword overlap using TF-IDF (1–2 n-grams).

### 🔹 **Job Fit Score**
Semantic similarity using the SBERT model:

### 🔹 **Final Fit Score**

### 🔹 **Required Skills %**
Percentage of must-have skills present in the resume.

---

# 📦 Project Structure

resume-matcher/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│ ├── parser.py
│ ├── skills.py
│ ├── matchers.py
│ ├── utils.py
│
├── assets/
│ └── deployed_qr.png ← auto-generated QR code (optional)
│
├── tools/
│ └── generate_qr.py
│
├── sample_data/
│ ├── resumes/
│ └── jds/
│
├── tests/
│ ├── test_parser.py
│ ├── test_matcher.py
│
└── notebooks/
└── demo.ipynb

---

# 🔧 Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/resume-matcher.git
cd resume-matcher
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python tools/generate_qr.py
streamlit run app.py
http://localhost:8501
pytest
```
📊 Example Use Cases

HR teams screening applicants

Students matching resumes to internships

Automated resume ranking systems

Job-application optimization

ATS enhancement projects

🛠 Future Enhancements

Cross-encoder re-ranking

OCR support for scanned PDFs

Multi-language resume support

Experience/education extraction

API version for ATS integrations

🤝 Contributing

Pull requests are welcome.
Open an issue if you want improvements or new features.

📄 License

MIT License.

🙏 Acknowledgements

Streamlit

HuggingFace Sentence-Transformers

scikit-learn

spaCy
