# evaluations/evaluate.py
import os
import numpy as np
from src.matchers import TfidfMatcher, SbertMatcher
def precision_at_k(ranked, ground_truth, k=1):
    topk = set(ranked[:k])
    return sum(1 for x in topk if x in ground_truth) / k

def evaluate_example():
    base = "sample_data"
    resumes = []
    for fn in os.listdir(os.path.join(base, "resumes")):
        with open(os.path.join(base, "resumes", fn), "r", encoding="utf-8") as f:
            resumes.append({"filename": fn, "text": f.read()})
    with open(os.path.join(base, "jds", "jd_ml_engineer.txt"), "r", encoding="utf-8") as f:
        jd = f.read()
    ground_truth = {"resume1.txt", "resume4.txt"}
    tfidf = TfidfMatcher()
    tfidf.fit([jd] + [r['text'] for r in resumes])
    sbert = SbertMatcher()
    sbert.fit([jd] + [r['text'] for r in resumes])
    results = []
    for r in resumes:
        t, _ = tfidf.score_document(jd, r['text'])
        s, _ = sbert.score_document(jd, r['text'])
        comb = 0.4*t + 0.6*s
        results.append((r['filename'], comb))
    ranked = [x for x,_ in sorted(results, key=lambda y: -y[1])]
    p1 = precision_at_k(ranked, ground_truth, k=1)
    p3 = precision_at_k(ranked, ground_truth, k=3)
    tfidf_scores = [tfidf.score_document(jd, r['text'])[0] for r in resumes]
    sbert_scores = [sbert.score_document(jd, r['text'])[0] for r in resumes]
    corr = np.corrcoef(tfidf_scores, sbert_scores)[0,1]
    print("Precision@1:", p1)
    print("Precision@3:", p3)
    print("TFIDF <-> SBERT correlation:", corr)

if __name__ == "__main__":
    evaluate_example()
