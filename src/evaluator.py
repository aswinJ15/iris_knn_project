"""Model evaluation module."""

def evaluate_accuracy(correct: int, total: int) -> float:
    """
    Calculate prediction accuracy rounded to 4 decimal places.

    Parameters
    ----------
    correct : int
        Number of correct predictions. Must be a non‑negative integer.
    total : int
        Total number of predictions. Must be an integer; if non‑positive,
        the function returns 0.0.

    Returns
    -------
    float
        Accuracy as a float rounded to 4 decimal places. Returns 0.0 when
        ``total`` is less than or equal to zero.

    Raises
    ------
    TypeError
        If ``correct`` or ``total`` are not integers.
    ValueError
        If ``correct`` is negative.
    """
    # Type validation
    if not isinstance(correct, int):
        raise TypeError(f"'correct' must be int, got {type(correct).__name__}")
    if not isinstance(total, int):
        raise TypeError(f"'total' must be int, got {type(total).__name__}")

    # Value validation
    if correct < 0:
        raise ValueError("'correct' must be non‑negative")

    if total <= 0:
        return 0.0

    return round(correct / total, 4)
