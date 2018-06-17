#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as st
from scipy.sparse import linalg
from scipy.optimize import minimize
import sklearn as sk
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_predict
from os.path import basename

from .util import is_pos_def


class StructuralCorrespondenceClassifier(object):
    """
    Class of classifiers based on structural correspondence learning.

    Methods contain different importance-weight estimators and different loss
    functions.
    """

    def __init__(self, loss='logistic', l2=1.0, num_pivots=1,
                 num_components=1):
        """
        Select a particular type of importance-weighted classifier.

        Parameters
        ----------
        loss : str
            loss function for weighted classifier, options: 'logistic',
                'quadratic', 'hinge' (def: 'logistic')
        l2 : float
            l2-regularization parameter value (def:0.01)
        num_pivots : int
            number of pivot features to use (def: 1)
        num_components : int
            number of components to use after extracting pivot features
            (def: 1)

        Returns
        -------
        None

        """
        self.loss = loss
        self.l2 = l2
        self.num_pivots = num_pivots
        self.num_components = num_components

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
            raise NotImplementedError('Loss not implemented yet.')

        # Whether model has been trained
        self.is_trained = False

        # Maintain pivot component matrix
        self.C = 0

        # Dimensionality of training data
        self.train_data_dim = ''

    def augment_features(self, X, Z, l2=0.0):
        """
        Find a set of pivot features, train predictors and extract bases.

        Parameters
        X : array
            source data array (N samples by D features)
        Z : array
            target data array (M samples by D features)
        l2 : float
            regularization parameter value (def: 0.0)

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

        # Concatenate source and target data
        XZ = np.concatenate((X, Z), axis=0)

        # Sort indices based on frequency of features (assumes BoW encoding)
        ix = np.argsort(np.sum(XZ, axis=0))

        # Keep most frequent features
        ix = ix[::-1][:self.num_pivots]

        # Slice out pivot features and relabel them as present(=1)/absent(=0)
        pivot = (XZ[:, ix] > 0).astype('float')

        # Solve prediction tasks with a Huber loss function
        P = np.zeros((DX, self.num_pivots))

        # Loop over pivot features
        for l in range(self.num_pivots):

            # Setup loss function for single pivot
            def L(theta): return self.Huber_loss(theta, XZ, pivot[:, l])

            # Setup gradient function for single pivot
            def J(theta): return self.Huber_grad(theta, XZ, pivot[:, l])

            # Make pivot predictor with a Huber loss function
            results = minimize(L, np.random.randn(DX, 1), jac=J, method='BFGS',
                               options={'gtol': 1e-6, 'disp': True})

            # Store optimal parameters
            P[:, l] = results.x

        # Compute covariance matrix of predictors
        SP = np.cov(P)

        # Add regularization to ensure positive-definiteness
        SP += l2*np.eye(self.num_pivots)

        # Eigenvalue decomposition of pivot predictor matrix
        V, C = np.linalg.eig(SP)

        # Reduce number of components
        C = C[:, :self.num_components]

        # Augment features
        Xa = np.concatenate((np.dot(X, C), X), axis=1)
        Za = np.concatenate((np.dot(Z, C), Z), axis=1)

        return Xa, Za, C

    def Huber_loss(self, theta, X, y, l2=0.0):
        """
        Huber loss function.

        Reference: Ando & Zhang (2005a). A framework for learning predictive
        structures from multiple tasks and unlabeled data. JMLR.

        Parameters
        ----------
        theta : array
            classifier parameters (D features by 1)
        X : array
            data (N samples by D features)
        y : array
            label vector (N samples by 1)
        l2 : float
            l2-regularization parameter (def= 0.0)

        Returns
        -------
        array
            Objective function value.

        """
        # Precompute terms
        Xy = (X.T*y.T).T
        Xyt = np.dot(Xy, theta)

        # Indices of discontinuity
        ix = (Xyt >= -1)

        # Loss function
        return np.sum(np.clip(1 - Xyt[ix], 0, None)**2, axis=0) \
            + np.sum(-4*Xyt[~ix], axis=0) + l2*np.sum(theta**2, axis=0)

    def Huber_grad(self, theta, X, y, l2=0.0):
        """
        Huber gradient computation.

        Reference: Ando & Zhang (2005a). A framework for learning predictive
        structures from multiple tasks and unlabeled data. JMLR.

        Parameters
        ----------
        theta : array
            classifier parameters (D features by 1)
        X : array
            data (N samples by D features)
        y : array
            label vector (N samples by 1)
        l2 : float
            l2-regularization parameter (def= 0.0)

        Returns
        -------
        array
            Gradient with respect to classifier parameters

        """
        # Precompute terms
        Xy = (X.T*y.T).T
        Xyt = np.dot(Xy, theta)

        # Indices of discontinuity
        ix = (Xyt >= -1)

        # Gradient
        return np.sum(2*np.clip(1-Xyt[ix], 0, None).T * -Xy[ix, :].T,
                      axis=1).T + np.sum(-4*Xy[~ix, :], axis=0) + 2*l2*theta

    def fit(self, X, y, Z):
        """
        Fit/train an structural correpondence classifier.

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

        # Augment features
        X, _, self.C = self.augment_features(X, Z, l2=self.l2)

        # Train a classifier
        if self.loss == 'logistic':
            # Logistic regression model
            self.clf.fit(X, y)
        elif self.loss == 'quadratic':
            # Least-squares model
            self.clf.fit(X, y)
        elif self.loss == 'hinge':
            # Linear support vector machine
            self.clf.fit(X, y)
        else:
            # Other loss functions are not implemented
            raise NotImplementedError('Loss not implemented.')

        # Mark classifier as trained
        self.is_trained = True

        # Store training data dimensionality
        self.train_data_dim = DX + self.num_components

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

        # Check for augmentation
        if not self.train_data_dim == D:
            Z = np.concatenate((np.dot(Z, self.C), Z), axis=1)

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
