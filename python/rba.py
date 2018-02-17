#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as st
from scipy.sparse.linalg import eigs
from scipy.spatial.distance import cdist
import sklearn as sk
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_predict
from os.path import basename

from util import is_pos_def


class RobustBiasAwareClassifier(object):
    """
    Class of robust bias-aware classifiers.

    Reference: Liu & Ziebart (20140. Robust Classification under Sample
    Selection Bias. NIPS.

    Methods contain training and prediction functions.
    """

    def __init__(self, l2=1.0, order='first', gamma=1.0, tau=1e-5,
                 max_iter=100, clip=1000):
        """
        Set classifier instance parameters.

        INPUT   (1) float 'l2': l2-regularization parameter value (def:0.01)
                (2) str 'order': order of feature statistics to employ; options
                    are 'first', or 'second' (def: 'first')
                (3) float 'gamma': decaying learning rate (def: 1.0)
                (4) float 'tau': convergence threshold (def: 1e-5)
                (5) int 'max_iter': maximum number of iterations (def: 100)
                (6) int 'clip': upper bound on importance weights (def: 1000)
        OUTPUT  None
        """
        self.l2 = l2
        self.order = order
        self.gamma = gamma
        self.tau = tau
        self.max_iter = max_iter
        self.clip = clip

        # Whether model has been trained
        self.is_trained = False

        # Dimensionality of training data
        self.train_data_dim = ''

        # Classifier parameters
        self.theta = 0

    def feature_stats(self, X, y, order='first'):
        """
        Compute first-order moment feature statistics.

        INPUT   (1) array 'X': dataset (N samples by D features)
                (2) array 'y': label vector (N samples by 1)
        OUTPUT  (1) array
        """
        # Data shape
        N, D = X.shape

        # Expand label vector
        if len(y.shape) < 2:
            y = np.atleast_2d(y).T

        if (order == 'first'):

            # First-order consists of data times label
            mom = y * X

        elif (order == 'second'):

            # First-order consists of data times label
            yX = y * X

            # Second-order is label times Kronecker delta product of data
            yXX = y*np.kron(X, X)

            # Concatenate moments
            mom = np.concatenate((yX, yXX), axis=1)

        # Concatenate label vector, moments, and ones-augmentation
        return np.concatenate((y, mom, np.ones((N, 1))), axis=1)

    def iwe_kernel_densities(self, X, Z):
        """
        Estimate importance weights based on kernel density estimation.

        INPUT   (1) array 'X': source data (N samples by D features)
                (2) array 'Z': target data (M samples by D features)
        OUTPUT  (1) array: importance weights (N samples by 1)
        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Assert equivalent dimensionalities
        assert DX == DZ

        # Compute probabilities based on source kernel densities
        pT = st.gaussian_kde(Z.T).pdf(X.T)
        pS = st.gaussian_kde(X.T).pdf(X.T)

        # Check for numerics
        assert not np.any(np.isnan(pT)) or np.any(pT == 0)
        assert not np.any(np.isnan(pS)) or np.any(pS == 0)

        # Return the ratio of probabilities
        return pT / pS

    def psi(self, X, theta, w, K=2):
        """
        Compute psi function.

        INPUT   (1)
        OUTPUT  (1) array 'psi'
        """
        # Number of samples
        N = X.shape[0]

        # Preallocate psi array
        psi = np.zeros((N, K))

        # Loop over classes
        for k in range(K):
            # Compute feature statistics
            Xk = self.feature_stats(X, k*np.ones((N, 1)))

            # Compute psi function
            psi[:, k] = w*np.dot(Xk, theta)[:, 0]

        return psi

    def posterior(self, psi):
        """
        Class-posterior estimation.

        INPUT   (1)
        OUTPUT  (1)
        """
        # Data shape
        N, K = psi.shape

        # Preallocate array
        pyx = np.zeros((N, K))

        # Subtract maximum value for numerical stability
        psi = (psi.T - np.max(psi, axis=1).T).T

        # Loop over classes
        for k in range(K):

            # Estimate posterior p^(Y=y | x_i)
            pyx[:, k] = np.exp(psi[:, k]) / np.sum(np.exp(psi), axis=1)

        return pyx

    def fit(self, X, y, Z):
        """
        Fit/train a robust bias-aware classifier.

        INPUT   (1) array 'X': source data (N samples by D features)
                (2) array 'y': source labels (N samples by 1)
                (3) array 'Z': target data (M samples by D features)
        OUTPUT  None
        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Number of classes
        self.K = len(np.unique(y))

        # Assert equivalent dimensionalities
        assert DX == DZ

        # Dimenionsality of expanded feature space
        if (self.order == 'first'):
            D = 1 + DX + 1
        elif (self.order == 'second'):
            D = 1 + DX + DX**2 + 1
        else:
            raise ValueError

        # Compute moment-matching constraint
        c = np.mean(self.feature_stats(X, y, order=self.order), axis=0)

        # Estimate importance-weights
        w = self.iwe_kernel_densities(X, Z)

        # Inverse weights to achieve p_S(x)/p_T(x)
        w = 1./w

        # Clip weights if necessary
        w = np.clip(w, 0, self.clip)

        # Initialize classifier parameters
        theta = np.random.randn(1, D)*0.01

        # Start gradient descent
        for t in range(self.max_iter):

            # Calculate psi function
            psi = self.psi(X, theta.T, w, K=self.K)

            # Compute posterior
            pyx = self.posterior(psi)

            # Compute product of estimated posterior and feature statistics
            pfs = 0
            for k in range(self.K):
                pfs += (pyx[:, k].T * self.feature_stats(X, k*np.ones((N, 1))).T).T

            # Gradient computation and regularization
            dL = c - np.mean(pfs, axis=0) + self.l2*2*theta

            # Apply learning rate to gradient
            dT = dL / (t * self.gamma)

            # Update classifier parameters
            theta += dT

            # Report progress
            if (t % self.max_iter / 10) == 1:
                print('Iteration %4i/%4i - Norm gradient: %3.5f\n', t,
                      self.max_iter, np.linalg.norm(dL))

            # Check for convergence
            if (np.linalg.norm(dL) <= self.tau):
                print('Broke at ' + str(t))
                break

        # Store resultant classifier parameters
        self.theta = theta

        # Mark classifier as trained
        self.is_trained = True

        # Store training data dimensionality
        self.train_data_dim = DX

    def predict(self, Z_):
        """
        Make predictions on new dataset.

        INPUT   (1) array 'Z_': new data set (M samples by D features)
        OUTPUT  (1) array 'preds': label predictions (M samples by 1)
        """
        # Data shape
        M, D = Z_.shape

        # If classifier is trained, check for same dimensionality
        if self.is_trained:
            assert self.train_data_dim == D
        else:
            raise UserWarning('Classifier is not trained yet.')

        # Calculate psi function for target samples
        psi = self.psi(Z_, self.theta.T, np.ones((M, 1)), K=self.K)

        # Compute posteriors for target samples
        pyz = self.posterior(psi)

        # Predictions through max-posteriors
        preds = np.argmax(pyz, axis=1)

        # Return predictions array
        return preds

    def get_params(self):
        """Get classifier parameters."""
        return self.clf.get_params()

    def is_trained(self):
        """Check whether classifier is trained."""
        return self.is_trained
