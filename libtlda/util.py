"""
Utility functions necessary for different classifiers.

Contains algebraic operations, label encodings and visualizations.
"""

import numpy as np
import numpy.linalg as al
import scipy.stats as st
import matplotlib.pyplot as plt


def one_hot(y, fill_k=False, one_not=False):
    """Map to one-hot encoding."""
    # Check for negative labels
    assert np.all(y >= 0)

    # Number of samples
    N = y.shape[0]

    # Number of classes
    K = len(np.unique(y))

    # Preallocate array
    if one_not:
        Y = -np.ones((N, K))
    else:
        Y = np.zeros((N, K))

    # Set k-th column to 1 for n-th sample
    for n in range(N):

        if fill_k:
            Y[n, y[n]] = y[n]
        else:
            Y[n, y[n]] = 1

    return Y


def regularize_matrix(A, a=0.0):
    """
    Regularize matrix by ensuring minimum eigenvalues.

    INPUT   (1) array 'A': square matrix
            (2) float 'a': constraint on minimum eigenvalue
    OUTPUT  (1) array 'B': constrained matrix
    """
    # Check for square matrix
    N, M = A.shape
    assert N == M

    # Check for valid matrix entries
    assert not np.any(np.isnan(A)) or np.any(np.isinf(A))

    # Check for non-negative minimum eigenvalue
    if a < 0:
        raise ValueError('minimum eigenvalue cannot be negative.')

    elif a == 0:
        return A

    else:
        # Ensure symmetric matrix
        A = (A + A.T) / 2

        # Eigenvalue decomposition
        E, V = al.eig(A)

        # Regularization matrix
        aI = a * np.eye(N)

        # Subtract regularization
        E = np.diag(E) + aI

        # Cap negative eigenvalues at zero
        E = np.maximum(0, E)

        # Reconstruct matrix
        B = np.dot(np.dot(V, E), V.T)

        # Add back subtracted regularization
        return B + aI


def is_pos_def(X):
    """Check for positive definiteness."""
    return np.all(np.linalg.eigvals(X) > 0)


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
    u, s, vh = al.svd(A)

    # Set tolerance
    tol = max(atol, rtol * s[0])

    # Compute the number of non-zero entries
    nnz = (s >= tol).sum()

    # Conjugate and transpose to ensure real numbers
    ns = vh[nnz:].conj().T

    return ns


def plotc(parameters, ax=[], color='k', gridsize=(101, 101)):
    """
    Plot a linear classifier in a 2D scatterplot.

    INPUT   (1) tuple 'parameters': consists of a list of class proportions
                (1 by K classes), an array of class means (K classes by
                D features), an array of class-covariance matrices (D features
                by D features by K classes)
            (2) object 'ax': axes of a pyplot figure or subject (def: empty)
            (3) str 'colors': colors of the contours in the plot (def: 'k')
            (4) tuple 'gridsize': number of points in the grid
                (def: (101, 101))
    OUTPUT  None
    """
    # Check for figure object
    if fig:
        ax = fig.gca()
    else:
        fig, ax = plt.subplots()

    # Get axes limits
    xl = ax.get_xlim()
    yl = ax.get_ylim()

    # Define grid
    gx = np.linspace(xl[0], xl[1], gridsize[0])
    gy = np.linspace(yl[0], yl[1], gridsize[1])
    x, y = np.meshgrid(gx, gy)
    xy = np.vstack((x.ravel(), y.ravel())).T

    # Values of grid
    z = np.dot(xy, parameters[:-1, :]) + parameters[-1, :]
    z = np.reshape(z[:, 0] - z[:, 1], gridsize)

    # Plot grid
    ax.contour(x, y, z, levels=0, colors=colors)


def plotlda(parameters, ax=[], colors='k', gridsize=(101, 101)):
    """
    Plot a linear discriminant analysis classifier in a 2D scatterplot.

    INPUT   (1) tuple 'parameters': consists of a list of class proportions
                (1 by K classes), an array of class means (K classes by
                D features), an array of class-covariance matrices (D features
                by D features by K classes)
            (2) object 'ax': axes of a pyplot figure or subject (def: empty)
            (3) str 'colors': colors of the contours in the plot (def: 'k')
            (4) tuple 'gridsize': number of points in the grid
                (def: (101, 101))
    OUTPUT  None
    """
    # Unpack parameters
    pi, mu, Si = parameters

    # Number of classes
    K = mu.shape[0]

    # Sum class-covariance matrices
    if len(Si) == 3:
        Si = np.sum(Si, axis=2)

    # Check for figure object
    if not ax:
        fig, ax = plt.subplots()

    # Get axes limits
    xl = ax.get_xlim()
    yl = ax.get_ylim()

    # Define grid
    gx = np.linspace(xl[0], xl[1], gridsize[0])
    gy = np.linspace(yl[0], yl[1], gridsize[1])
    x, y = np.meshgrid(gx, gy)
    xy = np.stack((x, y), axis=2)

    # Generate pdf's
    z = np.zeros((*gridsize, K))
    for k in range(K):
        z[:, :, k] = st.multivariate_normal(mean=mu[k, :], cov=Si).pdf(xy)

    # Difference of Gaussians
    dz = 10*(z[:, :, 1] - z[:, :, 0])

    # Plot grid
    ax.contour(x, y, dz, colors=colors)
