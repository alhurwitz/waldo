import numpy as np

from waldo.matcher import classify_face


def _norm(v):
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_classify_picks_best():
    grandma = _norm([1, 0, 0])
    grandpa = _norm([0, 1, 0])
    refs = {"grandma": [grandma], "grandpa": [grandpa]}
    assert classify_face(_norm([0.9, 0.1, 0]), refs, threshold=0.5) == "grandma"
    assert classify_face(_norm([0.1, 0.9, 0]), refs, threshold=0.5) == "grandpa"


def test_classify_below_threshold():
    refs = {"grandma": [_norm([1, 0, 0])]}
    assert classify_face(_norm([0, 1, 0]), refs, threshold=0.5) is None
