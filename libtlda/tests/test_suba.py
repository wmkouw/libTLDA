import numpy as np
import numpy.random as rnd
from libtlda.suba import SubspaceAlignedClassifier


def test_init01():
    """Test for object type."""
    clf = SubspaceAlignedClassifier()
    assert type(clf) == SubspaceAlignedClassifier


def test_init02():
    """Test for is_trained model."""
    clf = SubspaceAlignedClassifier()
    assert not clf.is_trained


def test_fit():
    """Test for fitting the model."""
    X = rnd.randn(10, 2)
    y = np.hstack((-np.ones((5,)), np.ones((5,))))
    Z = rnd.randn(10, 2) + 1
    clf = SubspaceAlignedClassifier()
    clf.fit(X, y, Z)
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
