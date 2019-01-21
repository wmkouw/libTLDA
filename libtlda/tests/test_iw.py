import numpy as np
import numpy.random as rnd

from libtlda.iw import ImportanceWeightedClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


def test_init():
    """Test for object type."""
    clf = ImportanceWeightedClassifier()
    assert type(clf) == ImportanceWeightedClassifier
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


def test_iwe_kernel_densities():
    """Test for estimating through kernel density estimation."""
    X = rnd.randn(10, 2)
    Z = rnd.randn(10, 2) + 1
    clf = ImportanceWeightedClassifier()
    iw = clf.iwe_kernel_densities(X, Z)
    assert np.all(iw >= 0)


def test_iwe_kernel_mean_matching():
    """Test for estimating through kernel mean matching."""
    X = rnd.randn(10, 2)
    Z = rnd.randn(10, 2) + 1
    clf = ImportanceWeightedClassifier()
    iw = clf.iwe_kernel_mean_matching(X, Z)
    assert np.all(iw >= 0)


def test_iwe_nearest_neighbours():
    """Test for estimating through nearest neighbours."""
    X = rnd.randn(10, 2)
    Z = rnd.randn(10, 2) + 1
    clf = ImportanceWeightedClassifier()
    iw = clf.iwe_nearest_neighbours(X, Z)
    assert np.all(iw >= 0)


def test_regularization():
    """Test for fitting the model."""
    X = rnd.randn(10, 2)
    y = np.hstack((-np.ones((5,)), np.ones((5,))))
    Z = rnd.randn(10, 2) + 1
    clf = ImportanceWeightedClassifier(loss_function='lr',
                                       l2_regularization=None)
    assert isinstance(clf.clf, LogisticRegressionCV)
    clf = ImportanceWeightedClassifier(loss_function='lr',
                                       l2_regularization=1.0)
    assert isinstance(clf.clf, LogisticRegression)


def test_fit():
    """Test for fitting the model."""
    X = rnd.randn(10, 2)
    y = np.hstack((-np.ones((5,)), np.ones((5,))))
    Z = rnd.randn(10, 2) + 1
    clf = ImportanceWeightedClassifier(loss_function='lr')
    clf.fit(X, y, Z)
    assert clf.is_trained
    clf = ImportanceWeightedClassifier(loss_function='qd')
    clf.fit(X, y, Z)
    assert clf.is_trained
    clf = ImportanceWeightedClassifier(loss_function='hinge')
    clf.fit(X, y, Z)
    assert clf.is_trained


def test_predict():
    """Test for making predictions."""
    X = rnd.randn(10, 2)
    y = np.hstack((-np.ones((5,)), np.ones((5,))))
    Z = rnd.randn(10, 2) + 1
    clf = ImportanceWeightedClassifier()
    clf.fit(X, y, Z)
    u_pred = clf.predict(Z)
    labels = np.unique(y)
    assert len(np.setdiff1d(np.unique(u_pred), labels)) == 0
