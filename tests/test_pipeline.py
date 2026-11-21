"""Unit tests for Arbitration RAG-NER, 6-Sector Clustering, and Award Estimation."""

from src.award_estimator import evaluate_award_estimators
from src.dataset_builder import build_arbitration_corpus
from src.rag_ner import chunk_document, evaluate_rag_ner, retrieve_relevant_chunks
from src.sector_clustering import cluster_arbitration_keywords


def test_corpus_and_rag_chunking():
    cases = build_arbitration_corpus(n_cases=30, seed=7, out_path="/tmp/test_arb_cases.json", force_rebuild=True)
    assert len(cases) == 30
    chunks = chunk_document(cases[0]["text"])
    assert len(chunks) >= 2
    top = retrieve_relevant_chunks(chunks, "net arbitral award INR Crore interest", top_k=2)
    assert len(top) <= 2
    assert any("award" in c.lower() for c in top)


def test_rag_ner_extraction_metrics():
    cases = build_arbitration_corpus(n_cases=24, seed=11, out_path="/tmp/test_arb_cases2.json", force_rebuild=True)
    rep = evaluate_rag_ner(cases, use_deepseek=False)
    assert rep.total_predicted > 500
    assert rep.overall_f1 > 0.70


def test_sector_clustering_beats_bow_baseline():
    cases = build_arbitration_corpus(n_cases=36, seed=13, out_path="/tmp/test_arb_cases3.json", force_rebuild=True)
    rep = cluster_arbitration_keywords(cases, n_clusters=6, seed=13)
    assert rep.n_keyword_instances > 700
    assert rep.purity > rep.baseline_bow_purity
    assert rep.purity > 0.75


def test_award_estimation_beats_claim_baseline():
    cases = build_arbitration_corpus(n_cases=60, seed=42, out_path="/tmp/test_arb_cases4.json", force_rebuild=True)
    rep = evaluate_award_estimators(cases, cv_folds=5, seed=42)
    base_mae = rep.model_metrics.loc["Linear_Claim_Baseline", "MAE_INR_Cr"]
    assert rep.best_mae_cr < base_mae
    assert rep.best_mae_cr < 15.0
