#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as st
from scipy.spatial.distance import cdist
import sklearn as sk
from sklearn.SVM import LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_predict
from os.path import basename

from util import is_pos_def


class ImportanceWeightedClassifier(object):
    """
    Class of importance-weighted classifiers.

    Methods contain different importance-weight estimators and different loss
    functions.
    """

    def __init__(self, iwe='lr', l2=1.0, smoothing=True, clip=-1,
                 loss='logistic'):
        """
        Select a particular type of importance-weighted classifier.

        INPUT   (1) str 'iwe': importance-weight estimator (def:'lr')
                (2) float 'l2': l2-regularization parameter value (def:0.01)
                (3) boolean 'smoothing': whether to apply Laplace smoothing to
                    the nearest-neighbour importance-weight estimator
                    (def: True)
                (4) float 'clip': maximum allowable importance-weight value; if
                    set to -1, then the weights are not clipped (def:-1)
                (5) str 'loss': loss function for weighted classifier, options:
                    'logistic', 'quadratic', 'hinge' (def: 'logistic')
        OUTPUT  (1) array 'preds': predictions on given target data (M samples
                    by 1)
                (2) array 'theta': linear classifier parameters (D features by
                    K classes)
        """
        self.iwe = iwe
        self.l2 = l2
        self.smoothing = smoothing
        self.clip = clip
        self.loss = loss

    def iwe_gauss(self, X, Z):
        """
        Estimate importance weights based on a ratio of Gaussian distributions.

        INPUT   (1) array 'X': source data (N samples by D features)
                (2) array 'Z': target data (M samples by D features)
        OUTPUT  (1) array: importance weights (N samples by 1)
        """
        # Sample means in each domain
        mu_X = np.mean(X, axis=0)
        mu_Z = np.mean(Z, axis=0)

        # Sample covariances
        Si_X = np.cov(X)
        Si_Z = np.cov(Z)

        # Check for positive-definiteness of covariance matrices
        if not (is_pos_def(Si_X) or is_pos_def(Si_Z)):
            print('Warning: covariate matrices not PSD.')

            regct = -6
            while not (is_pos_def(Si_X) or is_pos_def(Si_Z)):
                print('Adding regularization: ' + str(1**regct))

                # Add regularization
                Si_X += 1**regct
                Si_Z += 1**regct

                # Increment regularization counter
                regct += 1

        # Compute probability of X under each domain
        pT = st.multivariate_normal(X, mu_Z, Si_Z)
        pS = st.multivariate_normal(X, mu_X, Si_X)

        # Return the ratio of Gaussians
        return pT / pX

    def iwe_lr(self, X, Z):
        """
        Estimate importance weights based on logistic regression.

        INPUT   (1) array 'X': source data (N samples by D features)
                (2) array 'Z': target data (M samples by D features)
        OUTPUT  (1) array: importance weights (N samples by 1)
        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Assert equivalent dimensionalities
        if DX != DZ:
            raise AssertionError

        # Make domain-label variable
        y = np.concatenate((np.zeros((N, 1)),
                            np.ones((M, 1))), axis=0)

        # Concatenate data
        XZ = np.concatenate((X, Z), axis=0)

        # Call a logistic regressor
        lr = LogisticRegression(C=self.l2)

        # Predict probability of belonging to target using cross-validation
        preds = cross_val_predict(lr, XZ, y[:, 0])

        # Return predictions for source samples
        return preds[:N]

    def iwe_nn(self, X, Z):
        """
        Estimate importance weights based on nearest-neighbours.

        INPUT   (1) array 'X': source data (N samples by D features)
                (2) array 'Z': target data (M samples by D features)
        OUTPUT  (1) array: importance weights (N samples by 1)
        """
        # Number of source samples
        N = X.shape[0]

        # Compute Euclidean distance between samples
        d = cdist(X, Z, metric='euclidean')

        # Count target samples within each source Voronoi cell
        ix = np.argmin(d, axis=1)
        iw = hist(ix, np.arange(N))

        # Laplace smoothing
        if self.smoothing:
            iw = (iw + 1) / (N + 1)

        # Weight clipping
        if self.clip > 0:
            iw = min(self.clip, max(0, iw))

        # Return weights
        return iw

    def fit(self, X, y, Z):
        """
        Fit/train an importance-weighted classifier.

        INPUT   (1) array 'X': source data (N samples by D features)
                (2) array 'y': source labels (N samples by 1)
                (3) array 'Z': target data (M samples by D features)
                (4) str 'iwe': importance weight estimator,
                    options: 'lr', 'nn', 'rg', 'kmm', 'kde' (def: 'lr')
        OUTPUT  (1) array 'theta': trained classifier parameters
                (2) array 'preds': predictions of trained classifier on given
                    target samples
        """
        # Find importance-weights
        if self.iwe == 'lr':
            w = self.iwe_lr(X, Z)
        elif self.iwe == 'gauss':
            w = self.iwe_lr(X, Z)
        elif self.iwe == 'nn':
            w = self.iwe_nn(X, Z)
        elif self.iwe == 'kde':
            w = self.iwe_kde(X, Z)
        elif self.iwe == 'kmm':
            w = self.iwe_kmm(X, Z)

        # Train a weighted classifier
        if self.loss == 'logistic':
            # Logistic regression model with sample weights
            iwclf = LogisticRegression().fit(X, y, w)

        elif self.loss == 'quadratic':
            # Least-squares model with sample weights
            iwclf = LinearRegression().fit(X, y, w)

        elif self.loss == 'hinge':
            # Linear support vector machine with sample weights
            iwclf = LinearSVC().fit(X, y, w)

        # Get trained classifier parameters
        theta = iwclf.get_params()

        # Make predictions on given target data
        preds = iwclf.predict(Z)

        # Return classifier parameters and predictions
        return theta, preds
