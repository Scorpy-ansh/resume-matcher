# tests/test_matchers.py
from src.matchers import TfidfMatcher, SbertMatcher
def test_tfidf_basic():
    docs = ["job about python and sql", "candidate skilled in python", "candidate skilled in java"]
    model = TfidfMatcher(ngram_range=(1,2))
    model.fit(docs)
    sim, terms = model.score_document(docs[0], docs[1])
    assert sim >= 0
def test_sbert_basic():
    docs = ["python sql", "python skills", "java skills"]
    model = SbertMatcher()
    model.fit(docs)
    sim, top = model.score_document(docs[0], docs[1])
    assert sim >= 0
