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

        Arguments
        ---------
        loss : str
            loss function for weighted classifier, options: 'logistic',
            'quadratic', 'hinge' (def: 'logistic')
        l2 : float
            l2-regularization parameter value (def:0.01)
        num_components : int
            number of transfer components to maintain (def: 1)

        Returns
        -------
        None

        Examples
        --------
        clf = SubspaceAlignedClassifier(loss='hinge', l2=0.1)

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
            raise NotImplementedError('Loss function not implemented.')

        # Whether model has been trained
        self.is_trained = False

        # Dimensionality of training data
        self.train_data_dim = ''

    def subspace_alignment(self, X, Z, num_components=1):
        """
        Compute subspace and alignment matrix.

        Arguments
        ---------
        X : array
            source data set (N samples by D features)
        Z : array
            target data set (M samples by D features)
        num_components : int
            number of components (def: 1)

        Returns
        -------
        V : array
            transformation matrix (D features by D features)
        CX : array
            source principal component coefficients
        CZ : array
            target principal component coefficients

        Examples
        --------
        X = np.random.randn(100, 10)
        Z = np.random.randn(100, 10)*2 + 1
        clf = SubspaceAlignedClassifier()
        V, CX, CZ = clf.subspace_alignment(X, Z, num_components=2)

        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Assert equivalent dimensionalities
        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

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

        Arguments
        X : array
            source data (N samples by D features)
        y : array
            source labels (N samples by 1)
        Z : array
            target data (M samples by D features)

        Returns
        -------
        None

        Examples
        --------
        X = np.random.randn(10, 2)
        y = np.vstack((-np.ones((5,)), np.ones((5,))))
        Z = np.random.randn(10, 2)
        clf = SubspaceAlignedClassifier()
        clf.fit(X, y, Z)

        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Assert equivalent dimensionalities
        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        # Transfer component analysis
        V, CX, CZ = self.subspace_alignment(X, Z,
                                            num_components=self.num_components)

        # Store target subspace
        self.target_subspace = CZ

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

        Arguments
        ---------
        Z_ : array
            new data set (M samples by D features)
        whiten : boolean
            whether to whiten new data (def: false)

        Returns
        -------
        preds : array
            label predictions (M samples by 1)

        Examples
        --------
        X = np.random.randn(10, 2)
        y = np.vstack((-np.ones((5,)), np.ones((5,))))
        Z = np.random.randn(10, 2)
        clf = SubspaceAlignedClassifier()
        clf.fit(X, y, Z)
        preds = clf.predict(Z)

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
        Z_ = np.dot(Z_, self.target_subspace)

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
