import pytest

from smart_faq.data_loader import load_faqs, make_faq_chunks
from smart_faq.prompting import FALLBACK_ANSWER, answer_faq, make_prompt
from smart_faq.reranking import passes_threshold, rerank
from smart_faq.retrieval import search_tfidf


def sample_chunks():
    return make_faq_chunks(load_faqs("data/raw/faqs.csv"))


def test_rerank_adds_keyword_bonus_and_orders_results() -> None:
    results = search_tfidf("how do I cancel billing?", sample_chunks(), top_k=3)
    reranked = rerank("how do I cancel billing?", results)

    assert reranked[0]["source"] == "billing"
    assert reranked[0]["rerank_score"] > reranked[0]["score"]


def test_threshold_fallback_for_unknown_question() -> None:
    response = answer_faq("what is the refund policy?", sample_chunks(), method="tfidf", threshold=0.20)

    assert response["answer"] == FALLBACK_ANSWER
    assert response["sources"] == []
    assert response["best_score"] == 0.0


def test_prompt_contains_grounded_sources() -> None:
    results = search_tfidf("how do I reset my password?", sample_chunks(), top_k=1)
    prompt = make_prompt("how do I reset my password?", results)

    assert "Use only the FAQ context below" in prompt
    assert "Source: account" in prompt
    assert "FAQ Answer: You can reset your password" in prompt


def test_negative_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="threshold cannot be negative"):
        passes_threshold({"score": 0.5}, threshold=-0.1)
