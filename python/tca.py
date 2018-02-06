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


class TransferComponentClassifier(object):
    """
    Class of classifiers based on Transfer Component Analysis.

    Methods contain component analysis and general utilities.
    """

    def __init__(self, loss='logistic', l2=1.0, mu=1.0, num_components=1,
                 kernel_type='rbf', bandwidth=1.0, order=2.0):
        """
        Select a particular type of importance-weighted classifier.

        INPUT   (1) str 'loss': loss function for weighted classifier, options:
                    'logistic', 'quadratic', 'hinge' (def: 'logistic')
                (2) float 'l2': l2-regularization parameter value (def:0.01)
                (3) float 'mu': trade-off parameter (def: 1.0)
                (4) int 'num_components': number of transfer components to
                    maintain (def: 1)
                (5) str 'kernel_type': type of kernel to use, options: 'rbf'
                    (def: 'rbf')
                (6) float 'bandwidth': kernel bandwidth for transfer component
                    analysis (def: 1.0)
                (7) float 'order': order of polynomial for kernel (def: 2.0)
        """
        self.loss = loss
        self.l2 = l2
        self.mu = mu
        self.num_components = num_components

        self.kernel_type = kernel_type
        self.bandwidth = bandwidth
        self.order = order


        # Initialize untrained classifiers
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

        INPUT   (1) array 'X': data set (N samples by D features)
                (2) array 'Z': data set (M samples by D features)
                (3) str 'type': type of kernel, options: 'linear',
                    'polynomial', 'rbf', 'sigmoid' (def: 'linear')
                (4) float 'order': order of polynomial to use for the
                    polynomial kernel (def: 2.0)
                (5) float 'bandwidth': kernel bandwidth (def: 1.0)
        OUTPUT  (1) array: kernel matrix (N+M by N+M)
        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Assert equivalent dimensionalities
        assert DX == DZ

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
            raise NotImplementedError

    def transfer_component_analysis(self, X, Z):
        """
        Transfer Component Analysis.

        INPUT   (1) array 'X': source data set (N samples by D features)
                (2) array 'Z': target data set (M samples by D features)
        OUTPUT  (1) array 'C': transfer components (D features
                    by num_components)
                (2) array 'K': source and target data kernel distances
        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Assert equivalent dimensionalities
        assert DX == DZ

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

        INPUT   (1) array 'X': source data (N samples by D features)
                (2) array 'y': source labels (N samples by 1)
                (3) array 'Z': target data (M samples by D features)
        OUTPUT
        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Assert equivalent dimensionalities
        assert DX == DZ

        # Assert correct number of components for given dataset
        assert self.num_components <= N + M - 1

        # Maintain source and target data for later kernel computations
        self.XZ = np.concatenate((X, Z), axis=0)

        # Transfer component analysis
        self.C, K = self.transfer_component_analysis(X, Z)

        # Map source data onto transfer components
        X = np.dot(K[:N, :], self.C)

        # Train a weighted classifier
        if self.loss == 'logistic':
            # Logistic regression model with sample weights
            self.clf.fit(X, y)
        elif self.loss == 'quadratic':
            # Least-squares model with sample weights
            self.clf.fit(X, y)
        elif self.loss == 'hinge':
            # Linear support vector machine with sample weights
            self.clf.fit(X, y)
        else:
            # Other loss functions are not implemented
            raise NotImplementedError

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
            assert self.train_data_dim == D

        # Compute kernel for new data
        K = self.kernel(Z_, self.XZ, type=self.kernel_type,
                        bandwidth=self.bandwidth, order=self.order)

        # Map new data onto transfer components
        Z_ = np.dot(K, self.C)

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
