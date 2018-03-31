#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg.lapack as lap
import scipy.stats as st
from scipy.sparse.linalg import eigs
from scipy.spatial.distance import cdist
import sklearn as sk
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from os.path import basename

from .util import is_pos_def, nullspace


class GeodesicFlowClassifier(object):
    """
    Class of classifiers based on Geodesic Flow Kernel.

    Methods contain flow kernel computation, training and general utilities.
    """

    def __init__(self, loss='logistic', l2=1.0, num_neighbours=1):
        """
        Select a particular type of geodesic flow classifier.

        INPUT   (1) str 'loss': loss function for weighted classifier, options:
                    'logistic', 'quadratic', 'hinge' (def: 'logistic')
                (2) float 'l2': l2-regularization parameter value (def:0.01)
                (3) int 'num_neighbours': number of neighbours for knn (def: 1)
        """
        self.loss = loss
        self.l2 = l2

        # Initialize untrained classifiers
        if self.loss == 'logistic':
            # Logistic regression model
            self.clf = LogisticRegression(C=l2)
        elif self.loss == 'quadratic':
            # Least-squares model
            self.clf = LinearRegression(C=l2)
        elif self.loss == 'hinge':
            # Linear support vector machine
            self.clf = LinearSVC(C=l2)
        elif self.loss = 'knn':
            # k-nearest neighbours
            self.clf = KNeighborsClassifier(n_neighbors=num_neighbours)
        else:
            # Other loss functions are not implemented
            raise NotImplementedError

        # Whether model has been trained
        self.is_trained = False

        # Dimensionality of training data
        self.train_data_dim = ''

    def geodesic_flow_kernel(self, X, Z, subspace_dim=1):
        """
        Compute kernel for given data set.

        Reference: Geodesic Flow Kernel for Unsupervised Domain Adaptation.
        Gong, et al. (2008). CVPR

        INPUT   (1) array 'X': data set (N samples by D features)
                (2) array 'Z': data set (M samples by D features)
                (5) int 'subspace_dim': dimensionality of the subspace (def: 1)
        OUTPUT  (1) array: kernel matrix (N+M by N+M)
        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Assert equivalent dimensionalities
        assert DX == DZ

        # Find principal components
        CX = PCA(n_components=DX, whiten=True).fit(X).components_.T
        CZ = PCA(n_components=subspace_dim, whiten=True).fit(Z).components_.T

        # Compute component product
        CC = np.dot(CX.T, CZ)

        '''
        #TODO: Generalized singular value decomposition
        #U,S,Vt,info = lap.dgesvd(CC[:subspace_dim,:], CC[subspace_dim:, :])
        #[U,V,~,Gamma,~] = gsvd(QPZ(1:d,:), QPZ(d+1:end,:));

        # Find principal angles
        theta = real(acos(diag(Gamma)))

        # Ensure non-zero angles for computational stability
        theta = max(theta, 1e-20)

        # Filler zero matrices
        A1 = np.zeros(subspace_dim, D - subspace_dim)
        A2 = np.zeros(subspacec_dim, D - 2*subspace_dim)
        A3 = np.zeros(D, D - 2*subspace_dim)

        # Angle matrices
        L1 = 0.5.*diag(1 + sin(2*theta)./ (2.*theta));
        L2 = 0.5.*diag((-1 + cos(2*theta))./ (2.*theta));
        L3 = 0.5.*diag(1 - sin(2*theta)./ (2.*theta));

        # Constituent matrices
        C1 = [U, A1; A1', -V]
        C2 = [L1, L2, A2; L2, L3, A2; A3']
        C3 = [U, A1; A1', -V]'

        # Geodesic flow kernel
        G = Q * C1 * C2 * C3 * Q'
        '''

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

        # Compute geodesic flow kernel
        G = self.geodesic_flow_kernel(X, Z)

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
        elif self.loss == 'knn':
            # k-nearest-neighbour classifier
            self.clf = KNeighborsClassifier(metric='Mahalanobis',
                                            metric_params=G)
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

        '''GFK'''

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
