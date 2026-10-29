"""Model evaluation module."""

def evaluate_accuracy(correct: int, total: int) -> float:
    """Calculate prediction accuracy rounded to 4 decimals."""
    if total <= 0:
        return 0.0
    return round(correct / total, 4)
