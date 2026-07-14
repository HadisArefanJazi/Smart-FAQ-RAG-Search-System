import pandas as pd
import pytest

from smart_faq.data_loader import clean_text, load_faqs, make_faq_chunks, validate_faqs


def test_load_faqs_falls_back_to_sample_data() -> None:
    df = load_faqs("data/raw/faqs.csv")

    assert list(df.columns) == ["id", "question", "answer", "category"]
    assert len(df) == 3


def test_make_faq_chunks_preserves_original_shape() -> None:
    chunks = make_faq_chunks(load_faqs("data/raw/faqs.csv"))

    assert chunks[0] == {
        "id": 1,
        "source": "account",
        "question": "How can I reset my password?",
        "answer": "You can reset your password by clicking Forgot Password on the login page.",
        "text": "How can I reset my password? You can reset your password by clicking Forgot Password on the login page.",
    }


def test_validate_faqs_rejects_missing_required_columns() -> None:
    df = pd.DataFrame([{"question": "Q", "answer": "A"}])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_faqs(df)


def test_clean_text_matches_original_normalization() -> None:
    assert clean_text("  How   CAN I Reset?  ") == "how can i reset?"
