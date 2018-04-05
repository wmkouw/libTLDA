#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as st
from scipy.optimize import minimize
from scipy.sparse.linalg import eigs
from scipy.spatial.distance import cdist
import sklearn as sk
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_predict
from os.path import basename

from .util import is_pos_def, one_hot


class FeatureLevelDomainAdaptiveClassifier(object):
    """
    Class of feature-level domain-adaptive classifiers.

    Reference: Kouw, Krijthe, Loog & Van der Maaten (2016). Feature-level
    domain adaptation. JMLR.

    Methods contain training and prediction functions.
    """

    def __init__(self, l2=0.0, loss='logistic', transfer_model='blankout',
                 max_iter=100, tolerance=1e-5, verbose=True):
        """
        Set classifier instance parameters.

        INPUT   (1) float 'l2': l2-regularization parameter value (def:0.01)
                (2) str 'loss': loss function for classifier, options are
                    'logistic' or 'quadratic' (def: 'logistic')
                (3) str 'transfer_model': distribution to use for transfer
                    model, options are 'dropout' and 'blankout'
                    (def: 'blankout')
                (4) int 'max_iter': maximum number of iterations (def: 100)
                (5) float 'tolerance': convergence criterion threshold on x
                    (def: 1e-5)
                (7) boolean 'verbose': report training progress (def: True)
        OUTPUT  None
        """
        # Classifier choices
        self.l2 = l2
        self.loss = 'logistic'
        self.transfer_model = transfer_model

        # Optimization parameters
        self.max_iter = max_iter
        self.tolerance = tolerance

        # Whether model has been trained
        self.is_trained = False

        # Dimensionality of training data
        self.train_data_dim = 0

        # Classifier parameters
        self.theta = 0

        # Verbosity
        self.verbose = verbose

    def mle_transfer_dist(self, X, Z, dist='blankout'):
        """
        Maximum likelihood estimation of transfer model parameters.

        INPUT   (1) array 'X': source data set (N samples by D features)
                (2) array 'Z': target data set (M samples by D features)
                (3) str 'dist': distribution of transfer model, options are
                    'blankout' or 'dropout' (def: 'blankout')
        OUTPUT  (1) array 'iota': estimated transfer model parameters
                    (D features by 1)
        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Assert equivalent dimensionalities
        assert DX == DZ

        # Blankout and dropout have same maximum likelihood estimator
        if (dist == 'blankout') or (dist == 'dropout'):

            # Rate parameters
            eta = np.mean(X > 0, axis=0)
            zeta = np.mean(Z > 0, axis=0)

            # Ratio of rate parameters
            iota = np.clip(1 - zeta / eta, 0, None)

        else:
            raise ValueError('Distribution unknown.')

        return iota

    def moments_transfer_model(self, X, iota, dist='blankout'):
        """
        Moments of the transfer model.

        INPUT   (1) array 'X': data set (N samples by D features)
                (2) array 'iota': transfer model parameters (D samples by 1)
                (3) str 'dist': transfer model, options are 'dropout' and
                    'blankout' (def: 'blankout')
        OUTPUT  (1) array 'E': expected value of transfer model (N samples by
                    D feautures)
                (2) array 'V': variance of transfer model (D features by D
                    features by N samples)
        """
        # Data shape
        N, D = X.shape

        if (dist == 'dropout'):

            # First moment of transfer distribution
            E = (1-iota) * X

            # Second moment of transfer distribution
            V = np.zeros((D, D, N))
            for i in range(N):
                V[:, :, i] = np.diag(iota * (1-iota)) * (X[i, :].T*X[i, :])

        elif (dist == 'blankout'):

            # First moment of transfer distribution
            E = X

            # Second moment of transfer distribution
            V = np.zeros((D, D, N))
            for i in range(N):
                V[:, :, i] = np.diag(iota * (1-iota)) * (X[i, :].T * X[i, :])

        else:
            raise ValueError('Transfer distribution not implemented')

        return E, V

    def flda_log_loss(self, theta, X, y, E, V, l2=0.0):
        """
        Compute average loss for flda-log.

        INPUT   (1) array 'theta': classifier parameters (D features by 1)
                (2) array 'X': source data set ()
                (3) array 'y': label vector (N samples by 1)
                (4) array 'E': expected value with respect to transfer model
                    (N samples by D features)
                (5) array 'V': variance with respect to transfer model
                    (D features by D features by N samples)
                (6) float 'l2': regularization parameter (def: 0.0)
        OUTPUT  (1) float 'L': loss function value
        """
        # Data shape
        N, D = X.shape

        # Assert y in {-1,+1}
        assert np.all(np.sort(np.unique(y)) == (-1, 1))

        # Precompute terms
        Xt = np.dot(X, theta)
        Et = np.dot(E, theta)
        alpha = np.exp(Xt) + np.exp(-Xt)
        beta = np.exp(Xt) - np.exp(-Xt)
        gamma = (np.exp(Xt).T * X.T).T + (np.exp(-Xt).T * X.T).T
        delta = (np.exp(Xt).T * X.T).T - (np.exp(-Xt).T * X.T).T

        # Log-partition function
        A = np.log(alpha)

        # First-order partial derivative of log-partition w.r.t. Xt
        dA = beta / alpha

        # Second-order partial derivative of log-partition w.r.t. Xt
        d2A = 1 - beta**2 / alpha**2

        # Compute pointwise loss (negative log-likelihood)
        L = np.zeros((N, 1))
        for i in range(N):
            L[i] = -y[i] * Et[i] + A[i] + dA[i] * (Et[i] - Xt[i]) + \
                   1./2*d2A[i]*np.dot(np.dot(theta.T, V[:, :, i]), theta)

        # Compute risk (average loss)
        R = np.mean(L, axis=0)

        # Add regularization
        return R + l2*np.sum(theta**2, axis=0)

    def flda_log_grad(self, theta, X, y, E, V, l2=0.0):
        """
        Compute gradient with respect to theta for flda-log.

        INPUT   (1) array 'theta': classifier parameters (D features by 1)
                (2) array 'X': source data set ()
                (3) array 'y': label vector (N samples by 1)
                (4) array 'E': expected value with respect to transfer model
                    (N samples by D features)
                (5) array 'V': variance with respect to transfer model
                    (D features by D features by N samples)
                (6) float 'l2': regularization parameter (def: 0.0)
        OUTPUT  (1) float
        """
        # Data shape
        N, D = X.shape

        # Assert y in {-1,+1}
        assert np.all(np.sort(np.unique(y)) == (-1, 1))

        # Precompute common terms
        Xt = np.dot(X, theta)
        Et = np.dot(E, theta)
        alpha = np.exp(Xt) + np.exp(-Xt)
        beta = np.exp(Xt) - np.exp(-Xt)
        gamma = (np.exp(Xt).T * X.T).T + (np.exp(-Xt).T * X.T).T
        delta = (np.exp(Xt).T * X.T).T - (np.exp(-Xt).T * X.T).T

        # Log-partition function
        A = np.log(alpha)

        # First-order partial derivative of log-partition w.r.t. Xt
        dA = beta / alpha

        # Second-order partial derivative of log-partition w.r.t. Xt
        d2A = 1 - beta**2 / alpha**2

        dR = 0
        for i in range(N):

            # Compute gradient terms
            t1 = -y[i]*E[i, :].T

            t2 = beta[i] / alpha[i] * X[i, :].T

            t3 = (gamma[i, :] / alpha[i] - beta[i]*delta[i, :] /
                  alpha[i]**2).T * (Et[i] - Xt[i])

            t4 = beta[i] / alpha[i] * (E[i, :] - X[i, :]).T

            t5 = (1 - beta[i]**2 / alpha[i]**2) * np.dot(V[:, :, i], theta)

            t6 = -(beta[i] * gamma[i, :] / alpha[i]**2 - beta[i]**2 *
                   delta[i, :] / alpha[i]**3).T * np.dot(np.dot(theta.T,
                                                         V[:, :, i]), theta)

            dR += t1 + t2 + t3 + t4 + t5 + t6

        # Add regularization
        dR += l2*2*theta

        return dR

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

        # Assert equivalent dimensionalities
        assert DX == DZ

        # Number of classes
        K = len(np.unique(y))

        # Map to one-not-encoding
        Y = one_hot(y, one_not=True)

        # Compute transfer distribution parameters
        iota = self.mle_transfer_dist(X, Z)

        # Compute moments of transfer distribution
        E, V = self.moments_transfer_model(X, iota)

        # Select loss function
        if (self.loss == 'logistic'):

            # Preallocate parameter array
            theta = np.random.randn(DX, K)

            # Train a classifier for each class
            for k in range(K):

                # Shorthand for loss computation
                def L(theta): return self.flda_log_loss(theta, X, Y[:, k],
                                                        E, V, l2=self.l2)

                # Shorthand for gradient computation
                def J(theta): return self.flda_log_grad(theta, X, Y[:, k],
                                                        E, V, l2=self.l2)

                # Call scipy's minimizer
                results = minimize(L, theta[:, k], jac=J, method='BFGS',
                                   options={'gtol': self.tolerance,
                                            'disp': self.verbose})

                # Store resultant classifier parameters
                theta[:, k] = results.x

        elif (self.loss == 'quadratic'):

            # Compute closed-form least-squares solution
            theta = np.inv(E.T*E + np.sum(V, axis=2) + l2*np.eye(D))\
                         * (E.T * Y)

        # Store trained classifier parameters
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

        # Predict target labels
        preds = np.argmax(np.dot(Z_, self.theta), axis=1)

        # Return predictions array
        return preds

    def get_params(self):
        """Get classifier parameters."""
        return self.clf.get_params()

    def is_trained(self):
        """Check whether classifier is trained."""
        return self.is_trained
