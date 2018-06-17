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

        Parameters
        ----------
        l2 : float
            l2-regularization parameter value (def:0.01)
        loss : str
            loss function for classifier, options are 'logistic' or 'quadratic'
            (def: 'logistic')
        transfer_model : str
            distribution to use for transfer model, options are 'dropout' and
            'blankout' (def: 'blankout')
        max_iter : int
            maximum number of iterations (def: 100)
        tolerance : float
            convergence criterion threshold on x (def: 1e-5)
        verbose : bool
            report training progress (def: True)

        Returns
        -------
        None

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

        Parameters
        ----------
        X : array
            source data set (N samples by D features)
        Z : array
            target data set (M samples by D features)
        dist : str
            distribution of transfer model, options are 'blankout' or 'dropout'
            (def: 'blankout')

        Returns
        -------
        iota : array
            estimated transfer model parameters (D features by 1)

        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Assert equivalent dimensionalities
        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

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

        Parameters
        ----------
        X : array
            data set (N samples by D features)
        iota : array
            transfer model parameters (D samples by 1)
        dist : str
            transfer model, options are 'dropout' and 'blankout'
            (def: 'blankout')

        Returns
        -------
        E : array
            expected value of transfer model (N samples by D feautures)
        V : array
            variance of transfer model (D features by D features by N samples)

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
            raise NotImplementedError('Transfer distribution not implemented')

        return E, V

    def flda_log_loss(self, theta, X, y, E, V, l2=0.0):
        """
        Compute average loss for flda-log.

        Parameters
        ----------
        theta : array
            classifier parameters (D features by 1)
        X : array
            source data set (N samples by D features)
        y : array
            label vector (N samples by 1)
        E : array
            expected value with respect to transfer model (N samples by
            D features)
        V : array
            variance with respect to transfer model (D features by D features
            by N samples)
        l2 : float
            regularization parameter (def: 0.0)

        Returns
        -------
        dL : array
            Value of loss function.

        """
        # Data shape
        N, D = X.shape

        # Assert y in {-1,+1}
        if not np.all(np.sort(np.unique(y)) == (-1, 1)):
            raise NotImplementedError('Labels can only be {-1, +1} for now.')

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

        Parameters
        ----------
        theta : array
            classifier parameters (D features by 1)
        X : array
            source data set (N samples by D features)
        y : array
            label vector (N samples by 1)
        E : array
            expected value with respect to transfer model (N samples by
            D features)
        V : array
            variance with respect to transfer model (D features by D features
            by N samples)
        l2 : float
            regularization parameter (def: 0.0)

        Returns
        -------
        dR : array
            Value of gradient.

        """
        # Data shape
        N, D = X.shape

        # Assert y in {-1,+1}
        if not np.all(np.sort(np.unique(y)) == (-1, 1)):
            raise NotImplementedError('Labels can only be {-1, +1} for now.')

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

        Parameters
        ----------
        X : array
            source data (N samples by D features)
        y : array
            source labels (N samples by 1)
        Z : array
            target data (M samples by D features)

        Returns
        -------
        None

        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Assert equivalent dimensionalities
        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        # Map to one-not-encoding
        Y, labels = one_hot(y, one_not=True)

        # Number of classes
        K = len(labels)

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

        # Store classes
        self.classes = labels

        # Mark classifier as trained
        self.is_trained = True

        # Store training data dimensionality
        self.train_data_dim = DX

    def predict(self, Z_):
        """
        Make predictions on new dataset.

        Parameters
        ----------
        Z : array
            new data set (M samples by D features)

        Returns
        -------
        preds : array
            label predictions (M samples by 1)

        """
        # Data shape
        M, D = Z_.shape

        # If classifier is trained, check for same dimensionality
        if self.is_trained:
            if not self.train_data_dim == D:
                raise ValueError('''Test data is of different dimensionality
                                 than training data.''')

        # Predict target labels
        preds = np.argmax(np.dot(Z_, self.theta), axis=1)

        # Map predictions back to labels
        preds = self.classes[preds]

        # Return predictions array
        return preds

    def get_params(self):
        """Get classifier parameters."""
        return self.clf.get_params()

    def is_trained(self):
        """Check whether classifier is trained."""
        return self.is_trained
