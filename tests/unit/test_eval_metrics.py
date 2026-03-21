from app.eval.metrics import normalized_exact_match, result_set_match, schema_recall


def test_identical_rows() -> None:
    assert result_set_match([[1, "a"]], [[1, "a"]]) is True


def test_order_insensitive() -> None:
    assert result_set_match([[1], [2], [3]], [[3], [1], [2]]) is True


def test_different_values() -> None:
    assert result_set_match([[1]], [[2]]) is False


def test_multiset_not_set() -> None:
    assert result_set_match([[1], [1]], [[1]]) is False


def test_both_empty() -> None:
    assert result_set_match([], []) is True


def test_one_empty() -> None:
    assert result_set_match([], [[1]]) is False


def test_case_insensitive() -> None:
    assert normalized_exact_match("SELECT * FROM t", "select * from t") is True


def test_trailing_semicolon() -> None:
    assert normalized_exact_match("SELECT 1;", "SELECT 1") is True


def test_collapsed_whitespace() -> None:
    assert normalized_exact_match("SELECT   1", "SELECT 1") is True


def test_different_queries() -> None:
    assert normalized_exact_match("SELECT a FROM t", "SELECT b FROM t") is False


def test_full_recall() -> None:
    assert (
        schema_recall(
            gold_sql="SELECT * FROM singer JOIN concert ON singer.id = concert.id",
            retrieved_tables=["singer", "concert"],
        )
        == 1.0
    )


def test_partial_recall() -> None:
    assert (
        schema_recall(
            gold_sql="SELECT * FROM singer JOIN concert ON singer.id = concert.id",
            retrieved_tables=["singer"],
        )
        == 0.5
    )


def test_zero_recall() -> None:
    assert schema_recall("SELECT * FROM singer", retrieved_tables=["stadium"]) == 0.0


def test_no_tables_in_gold_returns_one() -> None:
    # SELECT 1 has no tables; nothing to retrieve → perfect by definition
    assert schema_recall("SELECT 1", retrieved_tables=[]) == 1.0


def test_alias_resolved() -> None:
    assert (
        schema_recall(
            "SELECT s.name FROM singer AS s",
            retrieved_tables=["singer"],
        )
        == 1.0
    )
