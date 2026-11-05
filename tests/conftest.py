import sys
from pathlib import Path

import pytest

# The framework modules live in the repository root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensor


# Finite differences are only accurate in float64. The seed makes weight
# initialisation and dropout the same on every run.
@pytest.fixture(autouse=True)
def float64_and_seed(monkeypatch):
    monkeypatch.setattr(tensor, "DTYPE", "float64")
    tensor.set_seed(0)
