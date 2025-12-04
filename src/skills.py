# src/skills.py
import json
import re
import spacy
nlp = spacy.load("en_core_web_sm")

DEFAULT_SKILLS = [
    "python","pandas","numpy","scikit-learn","sklearn","tensorflow","pytorch",
    "sql","postgres","mysql","aws","azure","gcp","docker","kubernetes","streamlit",
    "opencv","nlp","bert","transformer","flask","django","html","css","javascript",
    "matplotlib","seaborn","plotly","hadoop","spark","jupyter","git","rest api"
]

def load_skills_dict(path="src/skills.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except Exception:
        return {"skills": DEFAULT_SKILLS}

class SkillsExtractor:
    def __init__(self, skills_dict):
        self.skills = set([s.lower() for s in skills_dict.get("skills", DEFAULT_SKILLS)])
        escaped = [re.escape(s) for s in sorted(self.skills, key=lambda x: -len(x))]
        self.pattern = re.compile(r'\b(' + '|'.join(escaped) + r')\b', flags=re.I)

    def extract_from_text(self, text: str):
        found = set([m.group(0).lower() for m in self.pattern.finditer(text)])
        doc = nlp(text[:5000])
        ner_found = set([ent.text.lower() for ent in doc.ents if ent.label_ in ('ORG','PRODUCT')])
        all_found = found.union(ner_found)
        return {"all": sorted(all_found)}

    def extract_from_jd(self, jd_text: str):
        must = []
        nice = []
        for line in jd_text.splitlines():
            l = line.strip()
            if not l:
                continue
            ll = l.lower()
            tokens = [m.group(0).lower() for m in self.pattern.finditer(l)]
            if any(k in ll for k in ['must', 'required', 'mandatory']):
                must.extend(tokens)
            elif any(k in ll for k in ['prefer', 'nice to have', 'desired', 'optional']):
                nice.extend(tokens)
        all_tokens = [m.group(0).lower() for m in self.pattern.finditer(jd_text)]
        return {"must": sorted(set(must)), "preferred": sorted(set(nice)), "all": sorted(set(all_tokens))}
