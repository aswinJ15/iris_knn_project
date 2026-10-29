import sys
import os
sys.path.append(os.path.abspath('src'))

from evaluator import evaluate_accuracy

import pytest

def test_evaluate_accuracy_basic():
    assert evaluate_accuracy(9, 10) == 0.9
    assert evaluate_accuracy(0, 0) == 0.0
    assert evaluate_accuracy(10, 10) == 1.0

def test_evaluate_accuracy_rounding():
    assert evaluate_accuracy(1, 3) == 0.3333

def test_evaluate_accuracy_negative_total():
    assert evaluate_accuracy(5, -2) == 0.0

def test_evaluate_accuracy_invalid_types():
    with pytest.raises(TypeError):
        evaluate_accuracy("5", 10)
    with pytest.raises(TypeError):
        evaluate_accuracy(5, "10")
    with pytest.raises(TypeError):
        evaluate_accuracy(5.0, 10)
    with pytest.raises(TypeError):
        evaluate_accuracy(5, 10.0)

def test_evaluate_accuracy_negative_correct():
    with pytest.raises(ValueError):
        evaluate_accuracy(-1, 10)
