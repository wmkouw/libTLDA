#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as st
from scipy.spatial.distance import cdist
import sklearn as sk
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, \
    RidgeClassifier, RidgeClassifierCV
from sklearn.model_selection import cross_val_predict
from os.path import basename
from cvxopt import matrix, solvers

from .util import is_pos_def


class ImportanceWeightedClassifier(object):
    """
    Class of importance-weighted classifiers.

    Methods contain different importance-weight estimators and different loss
    functions.

    Examples
    --------
    | >>>> X = np.random.randn(10, 2)
    | >>>> y = np.vstack((-np.ones((5,)), np.ones((5,))))
    | >>>> Z = np.random.randn(10, 2)
    | >>>> clf = ImportanceWeightedClassifier()
    | >>>> clf.fit(X, y, Z)
    | >>>> u_pred = clf.predict(Z)

    """

    def __init__(self, loss_function='logistic', l2_regularization=None,
                 weight_estimator='lr', smoothing=True, clip_max_value=-1,
                 kernel_type='rbf', bandwidth=1):
        """
        Select a particular type of importance-weighted classifier.

        Parameters
        ----------
        loss : str
            loss function for weighted classifier, options: 'logistic',
            'quadratic', 'hinge' (def: 'logistic')
        l2_regularization : float
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

        """
        self.loss = loss_function
        self.l2 = l2_regularization
        self.iwe = weight_estimator
        self.smoothing = smoothing
        self.clip = clip_max_value
        self.kernel_type = kernel_type
        self.bandwidth = bandwidth

        # Initialize untrained classifiers based on choice of loss function
        if self.loss in ('lr', 'logr', 'logistic'):

            if l2_regularization:

                # Logistic regression model
                self.clf = LogisticRegression(C=self.l2, solver='lbfgs')

            else:
                # Logistic regression model
                self.clf = LogisticRegressionCV(cv=5, solver='lbfgs')

        elif self.loss in ('squared', 'qd', 'quadratic'):

            if l2_regularization:

                # Least-squares model with fixed regularization
                self.clf = RidgeClassifier(alpha=self.l2)

            else:
                # Least-squares model, cross-validated for regularization
                self.clf = RidgeClassifierCV(cv=5)

        elif self.loss in ('hinge', 'linsvm', 'linsvc'):

            # Linear support vector machine
            self.clf = LinearSVC()

        else:
            # Other loss functions are not implemented
            raise NotImplementedError('Loss function not implemented.')

        # Whether model has been trained
        self.is_trained = False

        # Initalize empty weight attribute
        self.iw = []

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

        Parameters
        ----------
        X : array
            source data (N samples by D features)
        Z : array
            target data (M samples by D features)

        Returns
        -------
        array
            importance weights (N samples by 1)

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

        Parameters
        ----------
        X : array
            source data (N samples by D features)
        Z : array
            target data (M samples by D features)

        Returns
        -------
        array
            importance weights (N samples by 1)

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
        if self.l2:

            lr = LogisticRegression(C=self.l2, solver='lbfgs')

        else:
            lr = LogisticRegressionCV(cv=5, solver='lbfgs')

        # Predict probability of belonging to target using cross-validation
        preds = cross_val_predict(lr, XZ, y[:, 0], cv=5)

        # Return predictions for source samples
        return preds[:N]

    def iwe_nearest_neighbours(self, X, Z):
        """
        Estimate importance weights based on nearest-neighbours.

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

        # Find importance-weights
        if self.iwe == 'lr':
            self.iw = self.iwe_logistic_discrimination(X, Z)
        elif self.iwe == 'rg':
            self.iw = self.iwe_ratio_gaussians(X, Z)
        elif self.iwe == 'nn':
            self.iw = self.iwe_nearest_neighbours(X, Z)
        elif self.iwe == 'kde':
            self.iw = self.iwe_kernel_densities(X, Z)
        elif self.iwe == 'kmm':
            self.iw = self.iwe_kernel_mean_matching(X, Z)
        else:
            raise NotImplementedError('Estimator not implemented.')

        # Train a weighted classifier
        self.clf.fit(X, y, self.iw)

        # Mark classifier as trained
        self.is_trained = True

        # Store training data dimensionality
        self.train_data_dim = DX

    def predict(self, Z):
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
        M, D = Z.shape

        # If classifier is trained, check for same dimensionality
        if self.is_trained:
            if not self.train_data_dim == D:
                raise ValueError('''Test data is of different dimensionality
                                 than training data.''')

        # Call scikit's predict function
        preds = self.clf.predict(Z)

        # For quadratic loss function, correct predictions
        if self.loss == 'quadratic':
            preds = (np.sign(preds)+1)/2.

        # Return predictions array
        return preds

    def predict_proba(self, Z):
        """
        Compute posterior probabilities on new dataset.

        Parameters
        ----------
        Z : array
            new data set (M samples by D features)

        Returns
        -------
        probs : array
            label predictions (M samples by K)

        """
        # Data shape
        M, D = Z.shape

        # If classifier is trained, check for same dimensionality
        if self.is_trained:
            if not self.train_data_dim == D:
                raise ValueError('''Test data is of different dimensionality
                                 than training data.''')

        # Call scikit's predict function
        if self.loss in ['logistic']:

            # Use scikit's predict_proba for posterior probabilities
            probs = self.clf.predict_proba(Z)

        else:
            raise NotImplementedError('''Posterior probabilities for quadratic
                                      and hinge losses not implemented yet.''')

        # Return posterior probabilities array
        return probs

    def get_params(self):
        """Get classifier parameters."""
        if self.is_trained:
            return self.clf.get_params()
        else:
            raise ValueError('Classifier is not yet trained.')

    def get_weights(self):
        """Get estimated importance weights."""
        if self.is_trained:
            return self.iw
        else:
            raise ValueError('Classifier is not yet trained.')

    def is_trained(self):
        """Check whether classifier is trained."""
        return self.is_trained
