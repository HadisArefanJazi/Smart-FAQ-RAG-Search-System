from pathlib import Path


def test_readme_contains_valid_mermaid_architecture_block() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "```mermaid\nflowchart TD" in readme
    assert 'A["FAQ CSV"] --> B["Load and validate data"]' in readme
    assert 'I --> J["Answer and sources"]' in readme
    assert "```" in readme.split("```mermaid", 1)[1]


def test_readme_does_not_reference_old_branch_name() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "refactor/professional-project-structure" not in readme
