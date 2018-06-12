import numpy as np
import numpy.random as rnd
from libtlda.iw import ImportanceWeightedClassifier


def test_init01():
    """Test for object type."""
    clf = ImportanceWeightedClassifier()
    assert type(clf) == ImportanceWeightedClassifier


def test_init02():
    """Test for is_trained model."""
    clf = ImportanceWeightedClassifier()
    assert not clf.is_trained


def test_iwe_ratio_Gaussians():
    """Test for estimating ratio of Gaussians."""
    X = rnd.randn(10, 2)
    Z = rnd.randn(10, 2) + 1
    clf = ImportanceWeightedClassifier()
    iw = clf.iwe_ratio_gaussians(X, Z)
    assert np.all(iw >= 0)


def test_iwe_logistic_discrimination():
    """Test for estimating through logistic classifier."""
    X = rnd.randn(10, 2)
    Z = rnd.randn(10, 2) + 1
    clf = ImportanceWeightedClassifier()
    iw = clf.iwe_logistic_discrimination(X, Z)
    assert np.all(iw >= 0)
