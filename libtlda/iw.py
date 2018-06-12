#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as st
from scipy.spatial.distance import cdist
import sklearn as sk
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_predict
from os.path import basename
from cvxopt import matrix, solvers

from .util import is_pos_def


class ImportanceWeightedClassifier(object):
    """
    Class of importance-weighted classifiers.

    Methods contain different importance-weight estimators and different loss
    functions.
    """

    def __init__(self, loss='logistic', l2=1.0, iwe='lr', smoothing=True,
                 clip=-1, kernel_type='rbf', bandwidth=1):
        """
        Select a particular type of importance-weighted classifier.

        Parameters
        ----------
        loss : str
            loss function for weighted classifier, options: 'logistic',
            'quadratic', 'hinge' (def: 'logistic')
        l2 : float
            l2-regularization parameter value (def:0.01)
        iwe : str
            importance weight estimator, options: 'lr', 'nn', 'rg', 'kmm',
            'kde' (def: 'lr')
        smoothing : bool
            whether to apply Laplace smoothing to the nearest-neighbour
            importance-weight estimator (def: True)
        clip : float
            maximum allowable importance-weight value; if set to -1, then the
            weights are not clipped (def:-1)
        kernel_type : str
            what type of kernel to use for kernel density estimation or kernel
            mean matching, options: 'diste', 'rbf' (def: 'rbf')
        bandwidth : float
            kernel bandwidth parameter value for kernel-based weight
            estimators (def: 1)

        Returns
        -------
        None

        Examples
        --------
        >>>> clf = ImportanceWeightedClassifier()

        """
        self.loss = loss
        self.l2 = l2
        self.iwe = iwe
        self.smoothing = smoothing
        self.clip = clip
        self.kernel_type = kernel_type
        self.bandwidth = bandwidth

        # Initialize untrained classifiers based on choice of loss function
        if self.loss == 'logistic':
            # Logistic regression model
            self.clf = LogisticRegression()
        elif self.loss == 'quadratic':
            # Least-squares model
            self.clf = LinearRegression()
        elif self.loss == 'hinge':
            # Linear support vector machine
            self.clf = LinearSVC()
        else:
            # Other loss functions are not implemented
            raise NotImplementedError('Loss function not implemented.')

        # Whether model has been trained
        self.is_trained = False

        # Dimensionality of training data
        self.train_data_dim = ''

    def iwe_ratio_gaussians(self, X, Z):
        """
        Estimate importance weights based on a ratio of Gaussian distributions.

        Parameters
        ----------
        X : array
            source data (N samples by D features)
        Z : array
            target data (M samples by D features)

        Returns
        -------
        iw : array
            importance weights (N samples by 1)

        Examples
        --------
        X = np.random.randn(10, 2)
        Z = np.random.randn(10, 2)
        clf = ImportanceWeightedClassifier()
        iw = clf.iwe_ratio_gaussians(X, Z)

        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Assert equivalent dimensionalities
        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        # Sample means in each domain
        mu_X = np.mean(X, axis=0)
        mu_Z = np.mean(Z, axis=0)

        # Sample covariances
        Si_X = np.cov(X.T)
        Si_Z = np.cov(Z.T)

        # Check for positive-definiteness of covariance matrices
        if not (is_pos_def(Si_X) or is_pos_def(Si_Z)):
            print('Warning: covariate matrices not PSD.')

            regct = -6
            while not (is_pos_def(Si_X) or is_pos_def(Si_Z)):
                print('Adding regularization: ' + str(1**regct))

                # Add regularization
                Si_X += np.eye(DX)*10.**regct
                Si_Z += np.eye(DZ)*10.**regct

                # Increment regularization counter
                regct += 1

        # Compute probability of X under each domain
        pT = st.multivariate_normal.pdf(X, mu_Z, Si_Z)
        pS = st.multivariate_normal.pdf(X, mu_X, Si_X)

        # Check for numerical problems
        if np.any(np.isnan(pT)) or np.any(pT == 0):
            raise ValueError('Source probabilities are NaN or 0.')
        if np.any(np.isnan(pS)) or np.any(pS == 0):
            raise ValueError('Target probabilities are NaN or 0.')

        # Return the ratio of probabilities
        return pT / pS

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
        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        # Compute probabilities based on source kernel densities
        pT = st.gaussian_kde(Z.T).pdf(X.T)
        pS = st.gaussian_kde(X.T).pdf(X.T)

        # Check for numerical problems
        if np.any(np.isnan(pT)) or np.any(pT == 0):
            raise ValueError('Source probabilities are NaN or 0.')
        if np.any(np.isnan(pS)) or np.any(pS == 0):
            raise ValueError('Target probabilities are NaN or 0.')

        # Return the ratio of probabilities
        return pT / pS

    def iwe_logistic_discrimination(self, X, Z):
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
        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

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

    def iwe_nearest_neighbours(self, X, Z):
        """
        Estimate importance weights based on nearest-neighbours.

        INPUT   (1) array 'X': source data (N samples by D features)
                (2) array 'Z': target data (M samples by D features)
        OUTPUT  (1) array: importance weights (N samples by 1)
        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Assert equivalent dimensionalities
        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        # Compute Euclidean distance between samples
        d = cdist(X, Z, metric='euclidean')

        # Count target samples within each source Voronoi cell
        ix = np.argmin(d, axis=1)
        iw, _ = np.array(np.histogram(ix, np.arange(N+1)))

        # Laplace smoothing
        if self.smoothing:
            iw = (iw + 1.) / (N + 1)

        # Weight clipping
        if self.clip > 0:
            iw = np.minimum(self.clip, np.maximum(0, iw))

        # Return weights
        return iw

    def iwe_kernel_mean_matching(self, X, Z):
        """
        Estimate importance weights based on kernel mean matching.

        INPUT   (1) array 'X': source data (N samples by D features)
                (2) array 'Z': target data (M samples by D features)
        OUTPUT  (1) array: importance weights (N samples by 1)
        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Assert equivalent dimensionalities
        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        # Compute sample pairwise distances
        KXX = cdist(X, X, metric='euclidean')
        KXZ = cdist(X, Z, metric='euclidean')

        # Check non-negative distances
        if not np.all(KXX >= 0):
            raise ValueError('Non-positive distance in source kernel.')
        if not np.all(KXZ >= 0):
            raise ValueError('Non-positive distance in source-target kernel.')

        # Compute kernels
        if self.kernel_type == 'rbf':
            # Radial basis functions
            KXX = np.exp(-KXX / (2*self.bandwidth**2))
            KXZ = np.exp(-KXZ / (2*self.bandwidth**2))

        # Collapse second kernel and normalize
        KXZ = N/M * np.sum(KXZ, axis=1)

        # Prepare for CVXOPT
        Q = matrix(KXX, tc='d')
        p = matrix(KXZ, tc='d')
        G = matrix(np.concatenate((np.ones((1, N)), -1*np.ones((1, N)),
                                   -1.*np.eye(N)), axis=0), tc='d')
        h = matrix(np.concatenate((np.array([N/np.sqrt(N) + N], ndmin=2),
                                   np.array([N/np.sqrt(N) - N], ndmin=2),
                                   np.zeros((N, 1))), axis=0), tc='d')

        # Call quadratic program solver
        sol = solvers.qp(Q, p, G, h)

        # Return optimal coefficients as importance weights
        return np.array(sol['x'])[:, 0]

    def fit(self, X, y, Z):
        """
        Fit/train an importance-weighted classifier.

        INPUT   (1) array 'X': source data (N samples by D features)
                (2) array 'y': source labels (N samples by 1)
                (3) array 'Z': target data (M samples by D features)
        OUTPUT  None
        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Assert equivalent dimensionalities
        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        # Find importance-weights
        if self.iwe == 'lr':
            w = self.iwe_logistic_discrimination(X, Z)
        elif self.iwe == 'rg':
            w = self.iwe_ratio_gaussians(X, Z)
        elif self.iwe == 'nn':
            w = self.iwe_nearest_neighbours(X, Z)
        elif self.iwe == 'kde':
            w = self.iwe_kernel_densities(X, Z)
        elif self.iwe == 'kmm':
            w = self.iwe_kernel_mean_matching(X, Z)
        else:
            raise NotImplementedError('Estimator not implemented.')

        # Train a weighted classifier
        if self.loss == 'logistic':
            # Logistic regression model with sample weights
            self.clf.fit(X, y, w)
        elif self.loss == 'quadratic':
            # Least-squares model with sample weights
            self.clf.fit(X, y, w)
        elif self.loss == 'hinge':
            # Linear support vector machine with sample weights
            self.clf.fit(X, y, w)
        else:
            # Other loss functions are not implemented
            raise NotImplementedError('Loss function not implemented.')

        # Mark classifier as trained
        self.is_trained = True

        # Store training data dimensionality
        self.train_data_dim = DX

    def predict(self, Z_):
        """
        Make predictions on new dataset.

        INPUT   (1) array 'Z_': new data set (M samples by D features)
        OUTPUT  (2) array 'preds': label predictions (M samples by 1)
        """
        # Data shape
        M, D = Z_.shape

        # If classifier is trained, check for same dimensionality
        if self.is_trained:
            if not self.train_data_dim == D:
                raise ValueError('''Test data is of different dimensionality 
                                 than training data.''')

        # Call scikit's predict function
        preds = self.clf.predict(Z_)

        # For quadratic loss function, correct predictions
        if self.loss == 'quadratic':
            preds = (np.sign(preds)+1)/2.

        # Return predictions array
        return preds

    def get_params(self):
        """Get classifier parameters."""
        return self.clf.get_params()

    def is_trained(self):
        """Check whether classifier is trained."""
        return self.is_trained
