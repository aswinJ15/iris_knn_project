import sys, os
sys.path.insert(0, os.path.abspath("src"))
from evaluator import evaluate_accuracy

def test_evaluate_accuracy_basic():
    assert evaluate_accuracy(9, 10) == 0.9
    assert evaluate_accuracy(0, 0) == 0.0
    assert evaluate_accuracy(10, 10) == 1.0
