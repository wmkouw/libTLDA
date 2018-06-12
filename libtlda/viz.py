"""
Visualization functions for different classifiers.

Contains plots for decision boundaries.
"""

import numpy as np
import scipy.stats as st

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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
    z = np.zeros((gridsize[0], gridsize[1], K))
    for k in range(K):
        z[:, :, k] = st.multivariate_normal(mean=mu[k, :], cov=Si).pdf(xy)

    # Difference of Gaussians
    dz = 10*(z[:, :, 1] - z[:, :, 0])

    # Plot grid
    ax.contour(x, y, dz, colors=colors)
