import logging

from smart_faq.main import load_example_questions, main


def test_cli_single_question_starts_and_logs_answer(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "smart-faq",
            "--question",
            "How do I reset my password?",
        ],
    )

    with caplog.at_level(logging.INFO):
        main()

    assert "You can reset your password" in caplog.text
    assert "Best score:" in caplog.text


def test_load_example_questions_reads_json_examples() -> None:
    questions = load_example_questions("examples/sample_queries.json")

    assert questions[0] == "how do I reset my password?"
    assert "what is the refund policy?" in questions
