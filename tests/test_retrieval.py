import numpy as np
import pytest

from smart_faq.data_loader import load_faqs, make_faq_chunks
from smart_faq.retrieval import search_semantic, search_tfidf


class FakeEmbeddingModel:
    def encode(self, sentences):
        mapping = {
            "How can I reset my password? You can reset your password by clicking Forgot Password on the login page.": [1.0, 0.0, 0.0],
            "How can I contact support? You can contact support through the help center or by emailing support@example.com.": [0.0, 1.0, 0.0],
            "Can I cancel my subscription? You can cancel your subscription from your account billing settings.": [0.0, 0.0, 1.0],
            "how do I contact support?": [0.0, 1.0, 0.0],
        }
        return np.array([mapping[str(sentence)] for sentence in sentences])


def sample_chunks():
    return make_faq_chunks(load_faqs("data/raw/faqs.csv"))


def test_tfidf_retrieval_returns_password_faq_first() -> None:
    results = search_tfidf("how do I reset my password?", sample_chunks(), top_k=3)

    assert results[0]["source"] == "account"
    assert round(float(results[0]["score"]), 3) == 0.857


def test_semantic_retrieval_uses_injected_model_without_loading_external_model() -> None:
    results = search_semantic("how do I contact support?", sample_chunks(), top_k=2, model=FakeEmbeddingModel())

    assert results[0]["source"] == "support"
    assert float(results[0]["score"]) == pytest.approx(1.0)


def test_retrieval_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="question must not be empty"):
        search_tfidf("   ", sample_chunks())


def test_retrieval_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        search_tfidf("how do I reset my password?", sample_chunks(), top_k=0)
