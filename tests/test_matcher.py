import numpy as np

from waldo.matcher import RefIndex


def _norm(v):
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_classify_picks_best():
    grandma = _norm([1, 0, 0])
    grandpa = _norm([0, 1, 0])
    refs = {"grandma": [grandma], "grandpa": [grandpa]}
    idx = RefIndex(refs)
    assert idx.classify(_norm([0.9, 0.1, 0]), threshold=0.5) == "grandma"
    assert idx.classify(_norm([0.1, 0.9, 0]), threshold=0.5) == "grandpa"


def test_classify_below_threshold():
    refs = {"grandma": [_norm([1, 0, 0])]}
    idx = RefIndex(refs)
    assert idx.classify(_norm([0, 1, 0]), threshold=0.5) is None


def test_classify_batch():
    grandma = _norm([1, 0, 0])
    grandpa = _norm([0, 1, 0])
    refs = {"grandma": [grandma], "grandpa": [grandpa]}
    idx = RefIndex(refs)
    embeddings = np.array([
        _norm([0.9, 0.1, 0]),
        _norm([0.1, 0.9, 0]),
        _norm([0, 0, 1]),  # no match
    ])
    results = idx.classify_batch(embeddings, threshold=0.5)
    assert results == ["grandma", "grandpa", None]


def test_classify_batch_empty():
    refs = {"a": [_norm([1, 0, 0])]}
    idx = RefIndex(refs)
    assert idx.classify_batch(np.zeros((0, 3), dtype=np.float32), threshold=0.5) == []


def test_multiple_refs_per_person():
    refs = {
        "grandma": [_norm([1, 0, 0]), _norm([0.9, 0.1, 0])],
        "grandpa": [_norm([0, 1, 0])],
    }
    idx = RefIndex(refs)
    assert idx.classify(_norm([0.95, 0.05, 0]), threshold=0.5) == "grandma"
    assert idx.classify(_norm([0.1, 0.9, 0]), threshold=0.5) == "grandpa"
