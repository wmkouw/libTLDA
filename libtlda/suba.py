#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as st
from scipy.sparse.linalg import eigs
from scipy.spatial.distance import cdist
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_predict
from os.path import basename

from .util import is_pos_def


class SubspaceAlignedClassifier(object):
    """
    Class of classifiers based on Subspace Alignment.

    Methods contain the alignment itself, classifiers and general utilities.
    """

    def __init__(self, loss='logistic', l2=1.0, num_components=1):
        """
        Select a particular type of subspace aligned classifier.

        INPUT   (1) str 'loss': loss function for weighted classifier, options:
                    'logistic', 'quadratic', 'hinge' (def: 'logistic')
                (2) float 'l2': l2-regularization parameter value (def:0.01)
                (3) int 'num_components': number of transfer components to
                    maintain (def: 1)
        """
        self.loss = loss
        self.l2 = l2
        self.num_components = num_components

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

        # Maintain target principal component coefficients
        self.CZ = ''

        # Whether model has been trained
        self.is_trained = False

        # Dimensionality of training data
        self.train_data_dim = ''

    def subspace_alignment(self, X, Z, num_components=1):
        """
        Compute subspace and alignment matrix.

        INPUT   (1) array 'X': source data set (N samples by D features)
                (2) array 'Z': target data set (M samples by D features)
                (3) int 'num_components': number of components (def: 1)
        OUTPUT  (1) array 'V': transformation matrix (D features by D features)
                (2) array 'CX': source principal component coefficients
                (3) array 'CZ': target principal component coefficients
        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Assert equivalent dimensionalities
        assert DX == DZ

        # Compute principal components
        CX = PCA(n_components=num_components, whiten=True).fit(X).components_.T
        CZ = PCA(n_components=num_components, whiten=True).fit(Z).components_.T

        # Aligned source components
        V = np.dot(CX.T, CZ)

        # Return transformation matrix and principal component coefficients
        return V, CX, CZ

    def fit(self, X, y, Z):
        """
        Fit/train a classifier on data mapped onto transfer components.

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

        # Transfer component analysis (store target subspace)
        V, CX, self.CZ = self.subspace_alignment(X, Z, num_components=self.
                                                 num_components)

        # Map source data onto source principal components
        X = np.dot(X, CX)

        # Align source data to target subspace
        X = np.dot(X, V)

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

    def predict(self, Z_, whiten=False):
        """
        Make predictions on new dataset.

        INPUT   (1) array 'Z_': new data set (M samples by D features)
                (2) boolean 'whiten': whether to whiten new data (def: false)
        OUTPUT  (1) array 'preds': label predictions (M samples by 1)
        """
        # Data shape
        M, D = Z_.shape

        # If classifier is trained, check for same dimensionality
        if self.is_trained:
            assert self.train_data_dim == D

        # Check for need to whiten data beforehand
        if whiten:
            Z_ = st.zscore(Z_)

        # Map new target data onto target subspace
        Z_ = np.dot(Z_, self.CZ)

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
