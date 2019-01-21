import numpy as np
import numpy.random as rnd
from libtlda.suba import SemiSubspaceAlignedClassifier, \
    SubspaceAlignedClassifier


def test_init():
    """Test for object type."""
    clf = SubspaceAlignedClassifier()
    assert type(clf) == SubspaceAlignedClassifier
    assert not clf.is_trained


def test_subspace_alignment():
    """Test the alignment between datasets."""
    X = rnd.randn(100, 10)
    Z = np.dot(rnd.randn(100, 10), np.diag(np.arange(1, 11)))
    clf = SubspaceAlignedClassifier()
    V, CX, CZ = clf.subspace_alignment(X, Z, subspace_dim=3)
    assert not np.any(np.isnan(V))
    assert CX.shape[1] == 3
    assert CZ.shape[1] == 3


def test_fit():
    """Test for fitting the model."""
    X = rnd.randn(10, 2)
    y = np.hstack((-np.ones((5,)), np.ones((5,))))
    Z = rnd.randn(10, 2) + 1
    clf = SubspaceAlignedClassifier()
    clf.fit(X, y, Z)
    assert clf.is_trained


def test_fit_semi():
    """Test for fitting the model."""
    X = rnd.randn(10, 2)
    y = np.hstack((np.zeros((5,)), np.ones((5,))))
    Z = rnd.randn(10, 2) + 1
    u = np.array([[0, 0], [9, 1]])
    clf = SemiSubspaceAlignedClassifier()
    clf.fit(X, y, Z, u)
    assert clf.is_trained


def test_predict():
    """Test for making predictions."""
    X = rnd.randn(10, 2)
    y = np.hstack((-np.ones((5,)), np.ones((5,))))
    Z = rnd.randn(10, 2) + 1
    clf = SubspaceAlignedClassifier()
    clf.fit(X, y, Z)
    u_pred = clf.predict(Z)
    labels = np.unique(y)
    assert len(np.setdiff1d(np.unique(u_pred), labels)) == 0


def test_predict_semi():
    """Test for making predictions."""
    X = rnd.randn(10, 2)
    y = np.hstack((np.zeros((5,)), np.ones((5,))))
    Z = rnd.randn(10, 2) + 1
    u = np.array([[0, 0], [9, 1]])
    clf = SemiSubspaceAlignedClassifier()
    clf.fit(X, y, Z, u)
    u_pred = clf.predict(Z)
    labels = np.unique(y)
    assert len(np.setdiff1d(np.unique(u_pred), labels)) == 0
