from app.eval.metrics import result_set_match


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
