import numpy as np
import numpy.random as rnd
from libtlda.tca import TransferComponentClassifier


def test_init():
    """Test for object type."""
    clf = TransferComponentClassifier()
    assert type(clf) == TransferComponentClassifier
    assert not clf.is_trained


def test_fit():
    """Test for fitting the model."""
    X = rnd.randn(10, 2)
    y = np.hstack((-np.ones((5,)), np.ones((5,))))
    Z = rnd.randn(10, 2) + 1
    clf = TransferComponentClassifier()
    clf.fit(X, y, Z)
    assert clf.is_trained


def test_predict():
    """Test for making predictions."""
    X = rnd.randn(10, 2)
    y = np.hstack((-np.ones((5,)), np.ones((5,))))
    Z = rnd.randn(10, 2) + 1
    clf = TransferComponentClassifier()
    clf.fit(X, y, Z)
    u_pred = clf.predict(Z)
    labels = np.unique(y)
    assert len(np.setdiff1d(np.unique(u_pred), labels)) == 0
