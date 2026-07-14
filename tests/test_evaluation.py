from smart_faq.data_loader import load_faqs, make_faq_chunks
from smart_faq.evaluation import evaluate


def test_evaluation_matches_original_sample_accuracy() -> None:
    chunks = make_faq_chunks(load_faqs("data/raw/faqs.csv"))

    summary = evaluate(chunks, method="tfidf")

    assert summary["accuracy"] == 1.0
    assert [row["predicted_source"] for row in summary["rows"]] == ["account", "support", "billing"]
