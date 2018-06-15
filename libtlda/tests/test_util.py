import numpy as np
import numpy.random as rnd
import pytest
from libtlda.util import is_pos_def, one_hot, regularize_matrix


def test_is_pos_def():
    """Check if function returns boolean positive-definiteness."""
    # Positive-definite matrix
    A = np.array([[1, 0], [0, 1]])

    # Not positive-definite matrix
    B = np.array([[-1, 0], [0, 1]])

    # Assert correct positive-definiteness
    assert is_pos_def(A)
    assert not is_pos_def(B)


def test_one_hot():
    """Check if one_hot returns correct label matrices."""
    # Generate label vector
    y = np.hstack((np.ones((10,))*0,
                   np.ones((10,))*1,
                   np.ones((10,))*2))

    # Map to matrix
    Y, labels = one_hot(y)

    # Check for only 0's and 1's
    assert len(np.setdiff1d(np.unique(Y), [0, 1])) == 0

    # Check for correct labels
    assert np.all(labels == np.unique(y))

    # Check correct shape of matrix
    assert Y.shape[0] == y.shape[0]
    assert Y.shape[1] == len(labels)


def test_regularize_matrix():
    """Test whether function regularizes matrix correctly."""
    # Generate test matrix
    A = rnd.randn(3)

    # Check for inappropriate input argument
    with pytest.raises(ValueError):
        regularize_matrix(A, a=-1.0)
