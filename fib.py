def fib(n: int) -> int:
    """迭代实现斐波那契数列。

    Args:
        n: 非负整数，表示斐波那契数列的第 n 项。

    Returns:
        斐波那契数列的第 n 项值。

    Raises:
        ValueError: 当 n 为负数时抛出。
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
