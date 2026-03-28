import pytest

from app.db.validator import SQLValidator


@pytest.fixture
def validator() -> SQLValidator:
    return SQLValidator()


def test_valid_select(validator: SQLValidator) -> None:
    result = validator.validate("SELECT * FROM singer")
    assert result.valid is True
    assert result.error is None


def test_insert_rejected(validator: SQLValidator) -> None:
    result = validator.validate("INSERT INTO singer VALUES (1, 'test')")
    assert result.valid is False
    assert result.error is not None


def test_update_rejected(validator: SQLValidator) -> None:
    result = validator.validate("UPDATE singer SET name = 'test' WHERE singer_id = 1")
    assert result.valid is False


def test_delete_rejected(validator: SQLValidator) -> None:
    result = validator.validate("DELETE FROM singer WHERE singer_id = 1")
    assert result.valid is False


def test_drop_rejected(validator: SQLValidator) -> None:
    result = validator.validate("DROP TABLE singer")
    assert result.valid is False


def test_injection_rejected(validator: SQLValidator) -> None:
    result = validator.validate("SELECT * FROM singer; DROP TABLE singer")
    assert result.valid is False


def test_empty_string_rejected(validator: SQLValidator) -> None:
    result = validator.validate("")
    assert result.valid is False


def test_prose_input_rejected(validator: SQLValidator) -> None:
    """LLM returning prose instead of SQL must not crash (Fix 1 — TokenError handling)."""
    result = validator.validate("I need to analyze this step by step to find the answer.")
    assert result.valid is False
    assert result.error is not None
