import numpy as np
import numpy.random as rnd
from libtlda.tcpr import TargetContrastivePessimisticClassifier


def test_init():
    """Test for object type."""
    clf = TargetContrastivePessimisticClassifier()
    assert type(clf) == TargetContrastivePessimisticClassifier
    assert not clf.is_trained


def test_fit():
    """Test for fitting the model."""
    X = np.vstack((rnd.randn(5, 2), rnd.randn(5, 2)+1))
    y = np.hstack((np.zeros((5,)), np.ones((5,))))
    Z = np.vstack((rnd.randn(5, 2)-1, rnd.randn(5, 2)+2))
    clf = TargetContrastivePessimisticClassifier(l2=0.1)
    clf.fit(X, y, Z)
    assert clf.is_trained


def test_predict():
    """Test for making predictions."""
    X = np.vstack((rnd.randn(5, 2), rnd.randn(5, 2)+1))
    y = np.hstack((np.zeros((5,)), np.ones((5,))))
    Z = np.vstack((rnd.randn(5, 2)-1, rnd.randn(5, 2)+2))
    clf = TargetContrastivePessimisticClassifier(l2=0.1)
    clf.fit(X, y, Z)
    u_pred = clf.predict(Z)
    labels = np.unique(y)
    assert len(np.setdiff1d(np.unique(u_pred), labels)) == 0
