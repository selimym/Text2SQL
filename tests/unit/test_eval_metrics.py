from app.eval.metrics import (
    fewshot_table_overlap,
    normalized_exact_match,
    result_set_match,
    schema_noise_ratio,
    schema_precision,
    schema_recall,
)


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


# schema_precision tests


def test_schema_precision_partial_overlap() -> None:
    assert (
        schema_precision(
            gold_sql="SELECT * FROM table_a",
            retrieved_tables=["table_a", "table_b"],
        )
        == 0.5
    )


def test_schema_precision_empty_retrieved() -> None:
    assert (
        schema_precision(
            gold_sql="SELECT * FROM table_a",
            retrieved_tables=[],
        )
        == 0.0
    )


def test_schema_precision_full_match() -> None:
    assert (
        schema_precision(
            gold_sql="SELECT * FROM table_a",
            retrieved_tables=["table_a"],
        )
        == 1.0
    )


def test_schema_precision_no_match() -> None:
    assert (
        schema_precision(
            gold_sql="SELECT * FROM table_a",
            retrieved_tables=["table_b"],
        )
        == 0.0
    )


# schema_noise_ratio tests


def test_schema_noise_ratio_is_complement() -> None:
    assert (
        schema_noise_ratio(
            gold_sql="SELECT * FROM table_a",
            retrieved_tables=["table_a", "table_b"],
        )
        == 0.5
    )


def test_schema_noise_ratio_empty_retrieved() -> None:
    # When retrieved_tables is empty, schema_noise_ratio returns 0.0 per spec
    assert (
        schema_noise_ratio(
            gold_sql="SELECT * FROM table_a",
            retrieved_tables=[],
        )
        == 0.0
    )


# fewshot_table_overlap tests


def test_fewshot_table_overlap_basic() -> None:
    # gold uses table_a
    # example1 uses table_a and table_b (overlap = 1/2 = 0.5)
    # example2 uses table_a (overlap = 1/1 = 1.0)
    # average = (0.5 + 1.0) / 2 = 0.75
    assert (
        fewshot_table_overlap(
            gold_sql="SELECT * FROM table_a",
            retrieved_sqls=[
                "SELECT * FROM table_a JOIN table_b",
                "SELECT * FROM table_a",
            ],
        )
        == 0.75
    )


def test_fewshot_table_overlap_empty_retrieved() -> None:
    assert (
        fewshot_table_overlap(
            gold_sql="SELECT * FROM table_a",
            retrieved_sqls=[],
        )
        == 0.0
    )


def test_fewshot_table_overlap_no_match() -> None:
    # gold uses table_a
    # example1 uses table_b (no overlap = 0.0)
    # example2 uses table_c (no overlap = 0.0)
    # average = 0.0
    assert (
        fewshot_table_overlap(
            gold_sql="SELECT * FROM table_a",
            retrieved_sqls=[
                "SELECT * FROM table_b",
                "SELECT * FROM table_c",
            ],
        )
        == 0.0
    )
