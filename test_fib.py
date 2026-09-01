import pytest

from fib import fib


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, 0),
        (1, 1),
        (10, 55),
    ],
)
def test_fib(n: int, expected: int) -> None:
    assert fib(n) == expected


def test_fib_negative() -> None:
    with pytest.raises(ValueError):
        fib(-1)
