from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re

# Simple safe sentence tokenizer (no NLTK)
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def sent_tokenize_safe(text: str):
    if not text or not text.strip():
        return []
    text = text.strip()
    parts = _SENT_SPLIT_RE.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    # fallback: if only one long chunk, split on newlines or periods
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r'[\r\n]+', text) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in text.split('.') if p.strip()]
    return parts

class TfidfMatcher:
    def __init__(self, ngram_range=(1,2), max_features=10000):
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features, stop_words='english')

    def fit(self, documents: List[str]):
        self.vectorizer.fit(documents)

    def score_document(self, jd_text: str, doc_text: str) -> Tuple[float, List[Tuple[str,float]]]:
        tfidf = self.vectorizer
        vecs = tfidf.transform([jd_text, doc_text])
        sim = float(cosine_similarity(vecs[0], vecs[1])[0,0])
        jd_vec = vecs[0].toarray().ravel()
        doc_vec = vecs[1].toarray().ravel()
        term_scores = {}
        feature_names = tfidf.get_feature_names_out()
        for idx, term in enumerate(feature_names):
            score = min(jd_vec[idx], doc_vec[idx])
            if score > 0:
                term_scores[term] = float(score)
        sorted_terms = sorted(term_scores.items(), key=lambda x: -x[1])[:20]
        return sim, sorted_terms

class SbertMatcher:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        # model load may download weights on first run
        self.model = SentenceTransformer(model_name)

    def fit(self, documents: List[str]):
        # no training; keep reference
        self.documents = documents

    def _embed_sentences(self, text: str):
        sents = sent_tokenize_safe(text)
        if not sents:
            sents = [text] if text else []
        sent_embs = self.model.encode(sents, convert_to_tensor=True)
        return sents, sent_embs

    def score_document(self, jd_text: str, doc_text: str, top_k_sentences=5):
        jd_emb = self.model.encode(jd_text, convert_to_tensor=True)
        sents, sent_embs = self._embed_sentences(doc_text)
        if len(sents) == 0:
            return 0.0, []
        sims = util.cos_sim(jd_emb, sent_embs)[0].cpu().numpy()
        doc_emb = sent_embs.mean(axis=0)
        doc_sim = util.cos_sim(jd_emb, doc_emb)[0].item()
        top_idx = list(np.argsort(-sims)[:top_k_sentences])
        top_sents = [sents[i] for i in top_idx]
        top_scores = [float(sims[i]) for i in top_idx]
        top = list(zip(top_sents, top_scores))
        return float(doc_sim), top

def combined_ranking(tfidf_score: float, sbert_score: float, tfidf_weight: float = 0.4, sbert_weight: float = 0.6) -> float:
    return float(tfidf_weight * tfidf_score + sbert_weight * sbert_score)
