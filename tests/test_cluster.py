import numpy as np

from waldo.cluster import FaceCluster


def _norm(v):
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_same_person_clusters_together():
    c = FaceCluster(threshold=0.5)
    a = _norm([1, 0, 0])
    b = _norm([0.95, 0.05, 0])
    assert c.assign(a) == c.assign(b)


def test_different_people_get_different_clusters():
    c = FaceCluster(threshold=0.5)
    a = _norm([1, 0, 0])
    b = _norm([0, 1, 0])
    assert c.assign(a) != c.assign(b)


def test_centroids_as_refs_filters_singletons():
    c = FaceCluster(threshold=0.5)
    a = _norm([1, 0, 0])
    b = _norm([0.95, 0.05, 0])
    lone = _norm([0, 0, 1])
    c.assign(a)
    c.assign(b)
    c.assign(lone)  # only 1 face in this cluster
    refs = c.centroids_as_refs(min_count=2)
    assert len(refs) == 1  # lone cluster dropped
    assert "person_1" in refs


def test_n_clusters():
    c = FaceCluster(threshold=0.5)
    c.assign(_norm([1, 0, 0]))
    c.assign(_norm([0, 1, 0]))
    c.assign(_norm([0, 0, 1]))
    assert c.n_clusters == 3
