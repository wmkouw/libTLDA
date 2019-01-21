#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as st
from scipy.sparse.linalg import eigs
from scipy.spatial.distance import cdist
import sklearn as sk
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, \
    RidgeClassifier, RidgeClassifierCV
from sklearn.model_selection import cross_val_predict
from os.path import basename

from .util import is_pos_def


class TransferComponentClassifier(object):
    """
    Class of classifiers based on Transfer Component Analysis.

    Methods contain component analysis and general utilities.
    """

    def __init__(self,
                 loss_function='logistic',
                 l2_regularization=1.0,
                 mu=1.0,
                 num_components=1,
                 kernel_type='rbf',
                 bandwidth=1.0,
                 order=2.0):
        """
        Select a particular type of transfer component classifier.

        Parameters
        ----------
        loss_function : str
            loss function for weighted classifier, options: 'logistic',
            'quadratic', 'hinge' (def: 'logistic')
        l2 : float
            l2-regularization parameter value (def:0.01)
        mu : float
            trade-off parameter (def: 1.0)
        num_components : int
            number of transfer components to maintain (def: 1)
        kernel_type : str
            type of kernel to use, options: 'rbf' (def: 'rbf')
        bandwidth : float
            kernel bandwidth for transfer component analysis (def: 1.0)
        order : float
            order of polynomial for kernel (def: 2.0)

        Returns
        -------
        None

        Attributes
        ----------
        loss
            which loss function to use
        is_trained
            whether the classifier has been trained on data already

        """
        self.loss = loss_function
        self.l2 = l2_regularization
        self.mu = mu
        self.num_components = num_components

        self.kernel_type = kernel_type
        self.bandwidth = bandwidth
        self.order = order

        # Initialize untrained classifiers
        if self.loss in ('lr', 'logr', 'logistic'):

            if l2_regularization:

                # Logistic regression model with fixed regularization
                self.clf = LogisticRegression(C=self.l2, solver='lbfgs')

            else:
                # Logistic regression model, cross-validated for regularization
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
            self.clf = LinearSVC(C=self.l2)

        elif self.loss in ('rbfsvc', 'rbfsvm'):

            # Radial basis function support vector machine
            self.clf = SVC(C=self.l2)
        else:
            # Other loss functions are not implemented
            raise NotImplementedError

        # Maintain source and transfer data for computing kernels
        self.XZ = ''

        # Maintain transfer components
        self.C = ''

        # Whether model has been trained
        self.is_trained = False

        # Dimensionality of training data
        self.train_data_dim = ''

    def kernel(self, X, Z, type='rbf', order=2, bandwidth=1.0):
        """
        Compute kernel for given data set.

        Parameters
        ----------
        X : array
            data set (N samples by D features)
        Z : array
            data set (M samples by D features)
        type : str
            type of kernel, options: 'linear', 'polynomial', 'rbf',
            'sigmoid' (def: 'linear')
        order : float
            degree for the polynomial kernel (def: 2.0)
        bandwidth : float
            kernel bandwidth (def: 1.0)

        Returns
        -------
        array
            kernel matrix (N+M by N+M)

        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Assert equivalent dimensionalities
        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        # Select type of kernel to compute
        if type == 'linear':
            # Linear kernel is data outer product
            return np.dot(X, Z.T)
        elif type == 'polynomial':
            # Polynomial kernel is an exponentiated data outer product
            return (np.dot(X, Z.T) + 1)**p
        elif type == 'rbf':
            # Radial basis function kernel
            return np.exp(-cdist(X, Z) / (2.*bandwidth**2))
        elif type == 'sigmoid':
            # Sigmoidal kernel
            return 1./(1 + np.exp(np.dot(X, Z.T)))
        else:
            raise NotImplementedError('Loss not implemented yet.')

    def transfer_component_analysis(self, X, Z):
        """
        Transfer Component Analysis.

        Parameters
        ----------
        X : array
            source data set (N samples by D features)
        Z : array
            target data set (M samples by D features)

        Returns
        -------
        C : array
            transfer components (D features by num_components)
        K : array
            source and target data kernel distances

        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Assert equivalent dimensionalities
        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        # Compute kernel matrix
        XZ = np.concatenate((X, Z), axis=0)
        K = self.kernel(XZ, XZ, type=self.kernel_type,
                        bandwidth=self.bandwidth)

        # Ensure positive-definiteness
        if not is_pos_def(K):
            print('Warning: covariate matrices not PSD.')

            regct = -6
            while not is_pos_def(K):
                print('Adding regularization: ' + str(10**regct))

                # Add regularization
                K += np.eye(N + M)*10.**regct

                # Increment regularization counter
                regct += 1

        # Normalization matrix
        L = np.vstack((np.hstack((np.ones((N, N))/N**2,
                                  -1*np.ones((N, M))/(N*M))),
                       np.hstack((-1*np.ones((M, N))/(N*M),
                                  np.ones((M, M))/M**2))))

        # Centering matrix
        H = np.eye(N + M) - np.ones((N + M, N + M)) / float(N + M)

        # Matrix Lagrangian objective function: (I + mu*K*L*K)^{-1}*K*H*K
        J = np.dot(np.linalg.inv(np.eye(N + M) +
                   self.mu*np.dot(np.dot(K, L), K)),
                   np.dot(np.dot(K, H), K))

        # Eigenvector decomposition as solution to trace minimization
        _, C = eigs(J, k=self.num_components)

        # Discard imaginary numbers (possible computation issue)
        return np.real(C), K

    def fit(self, X, y, Z):
        """
        Fit/train a classifier on data mapped onto transfer components.

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

        # Assert correct number of components for given dataset
        if not self.num_components <= N + M - 1:
            raise ValueError('''Number of components must be smaller than or
                             equal to the source sample size plus target sample
                             size plus 1.''')

        # Maintain source and target data for later kernel computations
        self.XZ = np.concatenate((X, Z), axis=0)

        # Transfer component analysis
        self.C, K = self.transfer_component_analysis(X, Z)

        # Map source data onto transfer components
        X = np.dot(K[:N, :], self.C)

        # Train a weighted classifier
        if self.loss in ('lr', 'logr', 'logistic'):

            # Logistic regression model with sample weights
            self.clf.fit(X, y)

        elif self.loss in ('squared', 'qd', 'quadratic'):

            # Least-squares model with sample weights
            self.clf.fit(X, y)

        elif self.loss in ('hinge', 'linsvm', 'linsvc'):

            # Linear support vector machine with sample weights
            self.clf.fit(X, y)
        else:
            # Other loss functions are not implemented
            raise NotImplementedError('Loss not implemented yet.')

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

        # Compute kernel for new data
        K = self.kernel(Z, self.XZ, type=self.kernel_type,
                        bandwidth=self.bandwidth, order=self.order)

        # Map new data onto transfer components
        Z = np.dot(K, self.C)

        # Call scikit's predict function
        preds = self.clf.predict(Z)

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
