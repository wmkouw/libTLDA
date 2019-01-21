#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as st
from scipy.linalg import eig, eigh, svd
from scipy.spatial.distance import pdist, squareform

from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, \
    RidgeClassifier, RidgeClassifierCV
from sklearn.model_selection import cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from os.path import basename


class SubspaceAlignedClassifier(object):
    """
    Class of classifiers based on Subspace Alignment.

    Methods contain the alignment itself, classifiers and general utilities.

    Examples
    --------
    | >>>> X = np.random.randn(10, 2)
    | >>>> y = np.vstack((-np.ones((5,)), np.ones((5,))))
    | >>>> Z = np.random.randn(10, 2)
    | >>>> clf = SubspaceAlignedClassifier()
    | >>>> clf.fit(X, y, Z)
    | >>>> preds = clf.predict(Z)

    """

    def __init__(self,
                 loss_function='logistic',
                 l2_regularization=None,
                 subspace_dim=1):
        """
        Select a particular type of subspace aligned classifier.

        Parameters
        ----------
        loss_function : str
            loss function for weighted classifier, options: 'logistic',
            'quadratic', 'hinge' (def: 'logistic')
        l2_regularization : float
            l2-regularization parameter value (def:0.01)
        subspace_dim : int
            Dimensionality of subspace to retain (def: 1)

        Returns
        -------
        None

        """
        # Set atttributes
        self.loss = loss_function
        self.l2 = l2_regularization
        self.subdim = subspace_dim

        # Initialize untrained classifiers
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
            self.clf = LinearSVC(C=self.l2)

        elif self.loss in ('rbfsvc', 'rbfsvm'):

            # Radial basis function support vector machine
            self.clf = SVC(C=self.l2)

        else:
            # Other loss functions are not implemented
            raise NotImplementedError('Loss function not implemented.')

        # Whether model has been trained
        self.is_trained = False

    def is_pos_def(self, A):
        """Check for positive definiteness."""
        return np.all(np.real(np.linalg.eigvals(A)) > 0)

    def reg_cov(self, X):
        """
        Regularize covariance matrix until non-singular.

        Parameters
        ----------
        C : array
            square symmetric covariance matrix.

        Returns
        -------
        C : array
            regularized covariance matrix.

        """
        # Compute mean of data
        muX = np.mean(X, axis=0, keepdims=1)

        # Compute covariance matrix without regularization
        SX = np.cov((X - muX).T)

        # Initialize regularization parameter
        reg = 1e-6

        # Keep going until non-singular
        while not self.is_pos_def(SX):

            # Compute covariance matrix with regularization
            SX = np.cov((X - muX).T) + reg*np.eye(X.shape[1])

            # Increment reg
            reg *= 10

        # Report regularization
        print('Final regularization parameter = {}'.format(reg))

        return SX

    def zca_whiten(self, X):
        """
        Perform ZCA whitening (aka Mahalanobis whitening).

        Parameters
        ----------
        X : array (M samples x D features)
            data matrix.

        Returns
        -------
        X : array (M samples x D features)
            whitened data.

        """
        # Covariance matrix
        Sigma = np.cov(X.T)

        # Singular value decomposition
        U, S, V = svd(Sigma)

        # Whitening constant to prevent division by zero
        epsilon = 1e-5

        # ZCA whitening matrix
        W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), V))

        # Apply whitening matrix
        return np.dot(X, W)

    def align_data(self, X, Z, CX, CZ, V):
        """
        Align data to components and transform source.

        Parameters
        ----------
        X : array
            source data set (N samples x D features)
        Z : array
            target data set (M samples x D features)
        CX : array
            source principal components (D features x d subspaces)
        CZ : array
            target principal component (D features x d subspaces)
        V : array
            transformation matrix (d subspaces x d subspaces)

        Returns
        -------
        X : array
            transformed source data (N samples x d subspaces)
        Z : array
            projected target data (M samples x d subspaces)

        """
        # Map source data onto source principal components
        XC = np.dot(X, CX)

        # Align projected source data to target components
        XV = np.dot(XC, V)

        # Map target data onto target principal components
        ZC = np.dot(Z, CZ)

        return XV, ZC

    def subspace_alignment(self, X, Z, subspace_dim=1):
        """
        Compute subspace and alignment matrix.

        Parameters
        ----------
        X : array
            source data set (N samples x D features)
        Z : array
            target data set (M samples x D features)
        subspace_dim : int
            Dimensionality of subspace to retain (def: 1)

        Returns
        -------
        V : array
            transformation matrix (D features x D features)
        CX : array
            source principal component coefficients
        CZ : array
            target principal component coefficients

        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Check for sufficient samples
        if (N < subspace_dim) or (M < subspace_dim):
            raise ValueError('Too few samples for subspace dimensionality.')

        # Assert equivalent dimensionalities
        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        # Compute covariance matrices
        SX = np.cov(X.T)
        SZ = np.cov(Z.T)

        # Eigendecomposition for d largest eigenvectors
        valX, vecX = eigh(SX, eigvals=(DX - subspace_dim, DX-1))
        valZ, vecZ = eigh(SZ, eigvals=(DZ - subspace_dim, DZ-1))

        # Sort eigenvectors x descending eigenvalues
        CX = vecX[:, np.argsort(np.real(valX))[::-1]]
        CZ = vecZ[:, np.argsort(np.real(valZ))[::-1]]

        # Optimal linear transformation matrix
        V = np.dot(CX.T, CZ)

        # Return transformation matrix and principal component coefficients
        return V, CX, CZ

    def fit(self, X, Y, Z):
        """
        Fit/train a classifier on data mapped onto transfer components.

        Parameters
        ----------
        X : array
            source data (N samples x D features).
        Y : array
            source labels (N samples x 1).
        Z : array
            target data (M samples x D features).

        Returns
        -------
        None

        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Check for sufficient samples
        if (N < self.subdim) or (M < self.subdim):
            raise ValueError('Too few samples for subspace dimensionality.')

        # Assert equivalent dimensionalities
        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        # Transfer component analysis
        V, CX, CZ = self.subspace_alignment(X, Z, subspace_dim=self.subdim)

        # Store target subspace
        self.target_subspace = CZ

        # Align source data to target subspace
        X, Z = self.align_data(X, Z, CX, CZ, V)

        # Train a weighted classifier
        if self.loss in ('lr', 'logr', 'logistic'):
            # Logistic regression model with sample weights
            self.clf.fit(X, Y)

        elif self.loss in ('square', 'qd', 'quadratic'):
            # Least-squares model with sample weights
            self.clf.fit(X, Y)

        elif self.loss in ('hinge', 'linsvm', 'linsvc'):
            # Linear support vector machine with sample weights
            self.clf.fit(X, Y)

        elif self.loss in ('rbfsvc', 'rbfsvm'):
            # Radial basis function support vector machine
            self.clf.fit(X, Y)

        else:
            # Other loss functions are not implemented
            raise NotImplementedError('Loss function not implemented')

        # Mark classifier as trained
        self.is_trained = True

        # Store training data dimensionality
        self.train_data_dim = DX

        # Store classes in training data
        self.K = np.unique(Y)

    def score(self, Z, U, zscore=False):
        """
        Compute classification error on test set.

        Parameters
        ----------
        Z : array
            new data set (M samples x D features)
        zscore : boolean
            whether to transform the data using z-scoring (def: false)

        Returns
        -------
        preds : array
            label predictions (M samples x 1)

        """
        # If classifier is trained, check for same dimensionality
        if self.is_trained:
            if not self.train_data_dim == Z.shape[1]:
                raise ValueError("""Test data is of different dimensionality
                                 than training data.""")

        # Make predictions
        preds = self.predict(Z, zscore=zscore)

        # Compute error
        return np.mean(preds != U)

    def predict(self, Z, zscore=False):
        """
        Make predictions on new dataset.

        Parameters
        ----------
        Z : array
            new data set (M samples x D features)
        zscore : boolean
            whether to transform the data using z-scoring (def: false)

        Returns
        -------
        preds : array
            label predictions (M samples x 1)

        """
        # If classifier is trained, check for same dimensionality
        if self.is_trained:
            if not self.train_data_dim == Z.shape[1]:
                raise ValueError("""Test data is of different dimensionality
                                 than training data.""")

        # Call predict_proba() for posterior probabilities
        probs = self.predict_proba(Z, zscore=zscore)

        # Take maximum over classes for indexing class list
        preds = self.K[np.argmax(probs, axis=1)]

        # Return predictions array
        return preds

    def predict_proba(self, Z, zscore=False, signed_classes=False):
        """
        Make predictions on new dataset.

        Parameters
        ----------
        Z : array
            new data set (M samples x D features)
        zscore : boolean
            whether to transform the data using z-scoring (def: false)

        Returns
        -------
        preds : array
            label predictions (M samples x 1)

        """
        # If classifier is trained, check for same dimensionality
        if self.is_trained:
            if not self.train_data_dim == Z.shape[1]:
                raise ValueError("""Test data is of different dimensionality
                                 than training data.""")

        # Check for need to whiten data beforehand
        if zscore:
            Z = st.zscore(Z)

        # Map new target data onto target subspace
        Z = np.dot(Z, self.target_subspace)

        # Use Platt scaling for ridge regressor
        if self.loss in ('squared', 'qd', 'quadratic'):

            # Call scikit's calibrator
            calibrator = CalibratedClassifierCV(self.clf, cv='prefit')

            # Apply Platt scaling
            probs = calibrator.predict_proba(Z)

        else:

            # Call scikit's predict function
            probs = self.clf.predict_proba(Z)

        # Return predictions array
        return probs

    def get_params(self):
        """Get classifier parameters."""
        return self.clf.get_params()


class SemiSubspaceAlignedClassifier(object):
    """
    Class of classifiers based on semi-supervised Subspace Alignment.

    Methods contain the alignment itself, classifiers and general utilities.

    Examples
    --------
    | >>>> X = np.random.randn(10, 2)
    | >>>> y = np.vstack((-np.ones((5,)), np.ones((5,))))
    | >>>> Z = np.random.randn(10, 2)
    | >>>> clf = SubspaceAlignedClassifier()
    | >>>> clf.fit(X, y, Z)
    | >>>> preds = clf.predict(Z)

    """

    def __init__(self,
                 loss_function='logistic',
                 l2_regularization=None,
                 subspace_dim=1):
        """
        Select a particular type of subspace aligned classifier.

        Parameters
        ----------
        loss_function : str
            loss function for weighted classifier, options: 'logistic',
            'quadratic', 'hinge' (def: 'logistic')
        l2_regularization : float
            l2-regularization parameter value (def:0.01)
        subspace_dim : int
            Dimensionality of subspace to retain (def: 1)

        Returns
        -------
        None

        """
        # Set atttributes
        self.loss = loss_function
        self.l2 = l2_regularization
        self.subdim = subspace_dim

        # Initialize untrained classifiers
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
            self.clf = LinearSVC(C=self.l2)

        elif self.loss in ('rbfsvc', 'rbfsvm'):

            # Radial basis function support vector machine
            self.clf = SVC(C=self.l2)

        else:
            # Other loss functions are not implemented
            raise NotImplementedError('Loss function not implemented.')

        # Whether model has been trained
        self.is_trained = False

    def is_pos_def(self, A):
        """
        Check for positive definiteness.

        Parameters
        ---------
        A : array
            square symmetric matrix.

        Returns
        -------
        bool
            whether matrix is positive-definite.
            Warning! Returns false for arrays containing inf or NaN.


        """
        # Check for valid numbers
        if np.any(np.isnan(A)) or np.any(np.isinf(A)):
            return False

        else:
            return np.all(np.real(np.linalg.eigvals(A)) > 0)

    def find_medioid(self, X, Y):
        """
        Find point with minimal distance to all other points.

        Parameters
        ----------
        X : array
            data set, with N samples x D features.
        Y : array
            labels to select for which samples to compute distances.

        Returns
        -------
        x : array
            medioid
        ix : int
            index of medioid

        """
        # Initiate an array with infinities
        A = np.full((X.shape[0],), np.inf)

        # Insert sum of distances to other points
        A[Y] = np.sum(squareform(pdist(X[Y, :])), axis=1)

        # Find the index of the point with the smallest distance
        ix = np.argmin(A)

        return X[ix, :], ix

    def reg_cov(self, X):
        """
        Regularize covariance matrix until non-singular.

        Parameters
        ----------
        C : array
            square symmetric covariance matrix.

        Returns
        -------
        C : array
            regularized covariance matrix.

        """
        # Number of data points
        N = X.shape[0]

        # Compute mean of data
        muX = np.mean(X, axis=0, keepdims=1)

        # Compute covariance matrix without regularization
        SX = np.dot((X - muX).T, (X - muX)) / N

        # Initialize regularization parameter
        reg = 1e-6

        # Keep going until non-singular
        while not self.is_pos_def(SX):

            # Compute covariance matrix with regularization
            SX = np.dot((X - muX).T, (X - muX)) / N + reg*np.eye(X.shape[1])

            # Increment reg
            reg *= 10

        # Report regularization
        print('Final regularization parameter = {}'.format(reg))

        return SX

    def align_classes(self, X, Y, Z, u, CX, CZ, V):
        """
        Project each class separately.

        Parameters
        ----------
        X : array
            source data set (N samples x D features)
        Y : array
            source labels (N samples x 1)
        Z : array
            target data set (M samples x D features)
        u : array
            target labels (m samples x 2)
        CX : array
            source principal components (K classes x D features x d subspaces)
        CZ : array
            target principal components (K classes x D features x d subspaces)
        V : array
            transformation matrix (K classes x d subspaces x d subspaces)

        Returns
        -------
        X : array
            transformed X (N samples x d features)
        Z : array
            transformed Z (M samples x d features)

        """
        # Number of source samples
        N = X.shape[0]

        # Number of classes
        K = len(np.unique(Y))

        # Subspace dimensionality
        d = V.shape[1]

        # Preallocate
        XV = np.zeros((N, d))

        for k in range(K):

            # Project the k-th class
            XV[Y == k, :] = np.dot(np.dot(X[Y == k, :], CX[k]), V[k])

            # Indices of all target samples with label k
            uk = u[u[:, 1] == k, 0]

            # Mean of labeled target samples
            muZk = np.mean(Z[uk, :], axis=0, keepdims=1)

            # Remove mean after projection
            XV[Y == k, :] -= np.mean(XV[Y == k, :], axis=0, keepdims=1)

            # Center the projected class on mean of labeled target samples
            XV[Y == k, :] += np.dot(muZk, CZ)

        # Project target data onto components
        Z = np.dot(Z, CZ)

        return XV, Z

    def semi_subspace_alignment(self, X, Y, Z, u, subspace_dim=1):
        """
        Compute subspace and alignment matrix, for each class.

        Parameters
        ----------
        X : array
            source data set (N samples x D features)
        Y : array
            source labels (N samples x 1)
        Z : array
            target data set (M samples x D features)
        u : array
            target labels, first column is index in Z, second column is label
            (m samples x 2)
        subspace_dim : int
            Dimensionality of subspace to retain (def: 1)

        Returns
        -------
        V : array
            transformation matrix (K, D features x D features)
        CX : array
            source principal component coefficients
        CZ : array
            target principal component coefficients

        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Check for sufficient samples
        if (N < subspace_dim) or (M < subspace_dim):
            raise ValueError('Too few samples for subspace dimensionality.')

        # Assert equivalent dimensionalities
        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        # Number of classes
        K = len(np.unique(Y))

        for k in range(K):

            # Check number of samples per class
            Nk = np.sum(Y == k)

            # Check if subspace dim is too large
            if (Nk < subspace_dim):

                # Reduce subspace dim
                subspace_dim = min(subspace_dim, Nk)

                # Report
                print('Reducing subspace dim to {}'.format(subspace_dim))

        # Total covariance matrix of target data
        SZ = self.reg_cov(Z)

        # Eigendecomposition for first d eigenvectors
        valZ, vecZ = eigh(SZ, eigvals=(DZ - subspace_dim, DZ-1))

        # Sort eigenvectors x descending eigenvalues
        CZ = vecZ[:, np.argsort(np.real(valZ))[::-1]]

        # Use k-nn to label target samples
        kNN = KNeighborsClassifier(n_neighbors=1)
        U = kNN.fit(Z[u[:, 0], :], u[:, 1]).predict(Z)

        # Preallocate
        CX = np.zeros((K, DX, subspace_dim))
        V = np.zeros((K, subspace_dim, subspace_dim))

        # For each class, align components
        for k in range(K):

            # Take means
            muXk = np.mean(X[Y == k, :], axis=0, keepdims=1)
            muZk = np.mean(Z[U == k, :], axis=0, keepdims=1)

            # Compute covariance matrix of current class
            SXk = self.reg_cov(X[Y == k, :])
            SZk = self.reg_cov(Z[U == k, :])

            # Eigendecomposition for first d eigenvectors
            valX, vecX = eigh(SXk, eigvals=(DX - subspace_dim, DX-1))
            valZ, vecZ = eigh(SZk, eigvals=(DZ - subspace_dim, DZ-1))

            # Sort based on descending eigenvalues
            CX[k] = vecX[:, np.argsort(np.real(valX))[::-1]]
            vecZ = vecZ[:, np.argsort(np.real(valZ))[::-1]]

            # Aligned source components
            V[k] = np.dot(CX[k].T, vecZ)

        # Return transformation matrix and principal component coefficients
        return V, CX, CZ

    def fit(self, X, Y, Z, u=None):
        """
        Fit/train a classifier on data mapped onto transfer components.

        Parameters
        ----------
        X : array
            source data (N samples x D features).
        Y : array
            source labels (N samples x 1).
        Z : array
            target data (M samples x D features).
        u : array
            target labels, first column corresponds to index of Z and second
            column corresponds to actual label (number of labels x 2).

        Returns
        -------
        None

        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Check for sufficient samples
        if (N < self.subdim) or (M < self.subdim):
            raise ValueError('Too few samples for subspace dimensionality.')

        # Assert equivalent dimensionalities
        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        # Transfer component analysis
        V, CX, CZ = self.semi_subspace_alignment(X, Y, Z, u,
                                                 subspace_dim=self.subdim)

        # Store target subspace
        self.target_subspace = CZ

        # Align classes
        X, Z = self.align_classes(X, Y, Z, u, CX, CZ, V)

        # Train a weighted classifier
        if self.loss in ('lr', 'logr', 'logistic'):
            # Logistic regression model with sample weights
            self.clf.fit(X, Y)

        elif self.loss in ('square', 'qd', 'quadratic'):
            # Least-squares model with sample weights
            self.clf.fit(X, Y)

        elif self.loss == 'hinge':
            # Linear support vector machine with sample weights
            self.clf.fit(X, Y)

        elif self.loss == 'rbfsvc':
            # Radial basis function support vector machine
            self.clf.fit(X, Y)

        else:
            # Other loss functions are not implemented
            raise NotImplementedError('Loss function not implemented')

        # Mark classifier as trained
        self.is_trained = True

        # Store training data dimensionality
        self.train_data_dim = DX

        # Store labels in training data
        self.K = np.unique(Y)

    def score(self, Z, U, zscore=False):
        """
        Compute classification error on test set.

        Parameters
        ----------
        Z : array
            new data set (M samples x D features)
        zscore : boolean
            whether to transform the data using z-scoring (def: false)

        Returns
        -------
        preds : array
            label predictions (M samples x 1)

        """
        # If classifier is trained, check for same dimensionality
        if self.is_trained:
            if not self.train_data_dim == Z.shape[1]:
                raise ValueError("""Test data is of different dimensionality
                                 than training data.""")

        # Make predictions
        preds = self.predict(Z, zscore=zscore)

        # Compute error
        return np.mean(preds != U)

    def predict(self, Z, zscore=False):
        """
        Make predictions on new dataset.

        Parameters
        ----------
        Z : array
            new data set (M samples x D features)
        zscore : boolean
            whether to transform the data using z-scoring (def: false)

        Returns
        -------
        preds : array
            label predictions (M samples x 1)

        """
        # If classifier is trained, check for same dimensionality
        if self.is_trained:
            if not self.train_data_dim == Z.shape[1]:
                raise ValueError("""Test data is of different dimensionality
                                 than training data.""")

        # Call predict_proba() for posterior probabilities
        probs = self.predict_proba(Z, zscore=zscore)

        # Take maximum over classes for indexing class list
        preds = self.K[np.argmax(probs, axis=1)]

        # Return predictions array
        return preds

    def predict_proba(self, Z, zscore=False, signed_classes=False):
        """
        Make predictions on new dataset.

        Parameters
        ----------
        Z : array
            new data set (M samples x D features)
        zscore : boolean
            whether to transform the data using z-scoring (def: false)

        Returns
        -------
        preds : array
            label predictions (M samples x 1)

        """
        # If classifier is trained, check for same dimensionality
        if self.is_trained:
            if not self.train_data_dim == Z.shape[1]:
                raise ValueError("""Test data is of different dimensionality
                                 than training data.""")

        # Check for need to whiten data beforehand
        if zscore:
            Z = st.zscore(Z)

        # Map new target data onto target subspace
        Z = np.dot(Z, self.target_subspace)

        # Use Platt scaling for ridge regressor
        if self.loss in ('squared', 'qd', 'quadratic'):

            # Call scikit's calibrator
            calibrator = CalibratedClassifierCV(self.clf, cv='prefit')

            # Apply Platt scaling
            probs = calibrator.predict_proba(Z)

        else:

            # Call scikit's predict function
            probs = self.clf.predict_proba(Z)

        # Return predictions array
        return probs

    def get_params(self):
        """Get classifier parameters."""
        return self.clf.get_params()
