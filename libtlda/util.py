"""
Set of utility functions necessary for different classifiers.
"""

import numpy as np
import scipy.stats as st
from numpy.linalg import svd


def is_pos_def(X):
    """Check for positive definiteness."""
    return np.all(np.linalg.eigvals(X) > 0)


def one_not(y):
    """Map to one-hot encoding with -1 as negatives."""
    # Number of samples
    N = y.shape[0]

    # Number of classes
    K = len(np.unique(y))

    # Preallocate array
    Y = -np.ones((N, K))

    # Set k-th column to 1 for n-th sample
    for n in range(N):
        Y[n, y[n]] = 1

    return Y


def nullspace(A, atol=1e-13, rtol=0):
    """
    Compute an approximate basis for the nullspace of A.

    INPUT   (1) array 'A': 1-D array with length k will be treated
                as a 2-D with shape (1, k).
            (2) float 'atol': the absolute tolerance for a zero singular value.
                Singular values smaller than `atol` are considered to be zero.
            (3) float 'rtol': relative tolerance. Singular values less than
                rtol*smax are considered to be zero, where smax is the largest
                singular value.

                If both `atol` and `rtol` are positive, the combined tolerance
                is the maximum of the two; tol = max(atol, rtol * smax)
                Singular values smaller than `tol` are considered to be zero.
    OUTPUT  (1) array 'B': if A is an array with shape (m, k), then B will be
                an array with shape (k, n), where n is the estimated dimension
                of the nullspace of A.  The columns of B are a basis for the
                nullspace; each element in np.dot(A, B) will be
                approximately zero.
    """
    # Expand A to a matrix
    A = np.atleast_2d(A)

    # Singular value decomposition
    u, s, vh = svd(A)

    # Set tolerance
    tol = max(atol, rtol * s[0])

    # Compute the number of non-zero entries
    nnz = (s >= tol).sum()

    # Conjugate and transpose to ensure real numbers
    ns = vh[nnz:].conj().T

    return ns
