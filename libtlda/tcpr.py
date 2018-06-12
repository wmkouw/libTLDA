#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as al
from scipy.stats import multivariate_normal as mvn
import sklearn as sk
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_predict
from os.path import basename

from .util import one_hot, regularize_matrix


class TargetContrastivePessimisticClassifier(object):
    """
    Classifiers based on Target Contrastive Pessimistic Risk minimization.

    Methods contain models, risk functions, parameter estimation, etc.
    """

    def __init__(self, loss='lda', l2=1.0, max_iter=500, tolerance=1e-12,
                 learning_rate=1.0, rate_decay='linear', verbosity=0):
        """
        Select a particular type of TCPR classifier.

        INPUT   (1) str 'loss': loss function for TCP Risk, options: 'ls',
                    'least_squares', 'lda', 'linear_discriminant_analysis',
                    'qda', 'quadratic_discriminant_analysis' (def: 'lda')
                (2) float 'l2': l2-regularization parameter value (def:0.01)
                (3) int 'max_iter': maximum number of iterations for
                    optimization (def: 500)
                (4) float 'tolerance': convergence criterion on the TCP
                    parameters (def: 1e-5)
                (5) float 'learning_rate': parameter for size of update of
                    gradient (def: 1.0)
                (6) str 'rate_decay': type of learning rate decay, options:
                    'linear', 'quadratic', 'geometric', 'exponential'
                    (def: 'linear')
        """
        # Classifier options
        self.loss = loss
        self.l2 = l2

        # Optimization parameters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.learning_rate = learning_rate
        self.rate_decay = rate_decay
        self.verbosity = verbosity

        if self.loss in ['linear discriminant analysis', 'lda']:

            # Set to short name
            self.loss = 'lda'

        elif self.loss in ['quadratic discriminant analysis', 'qda']:

            # Set to short name
            self.loss = 'qda'

        else:
            # Other loss functions are not implemented
            raise ValueError('Model not implemented.')

        # Initialize classifier and classes parameters
        self.parameters = []
        self.classes = []

        # Whether model has been trained
        self.is_trained = False

        # Dimensionality of training data
        self.train_data_dim = 0

    def add_intercept(self, X):
        """Add 1's to data as last features."""
        # Data shape
        N, D = X.shape

        # Check if there's not already an intercept column
        if np.any(np.sum(X, axis=0) == N):

            # Report
            print('Intercept is not the last feature. Swapping..')

            # Find which column contains the intercept
            intercept_index = np.argwhere(np.sum(X, axis=0) == N)

            # Swap intercept to last
            X = X[:, np.setdiff1d(np.arange(D), intercept_index)]

        # Add intercept as last column
        X = np.hstack((X, np.ones((N, 1))))

        # Append column of 1's to data, and increment dimensionality
        return X, D+1

    def remove_intercept(self, X):
        """Remove 1's from data as last features."""
        # Data shape
        N, D = X.shape

        # Find which column contains the intercept
        intercept_index = np.argwhere(np.sum(X, axis=0) == N)

        # Swap intercept to last
        X = X[:, np.setdiff1d(np.arange(D), intercept_index)]

        return X, D-1

    def project_simplex(self, v, z=1.0):
        """
        Project vector onto simplex using sorting.

        Reference: "Efficient Projections onto the L1-Ball for Learning in High
        Dimensions (Duchi, Shalev-Shwartz, Singer, Chandra, 2006)."

        INPUT   (1) array 'v': vector to be projected (n dimensions by 0)
                (2) float 'z': constant (def: 1.0)
        OUTPUT  (1) array 'w': projected vector (n dimensions by 0)
        """
        # Number of dimensions
        n = v.shape[0]

        # Sort vector
        mu = np.sort(v, axis=0)[::-1]

        # Find rho
        C = np.cumsum(mu) - z
        j = np.arange(n) + 1
        rho = j[mu - C/j > 0][-1]

        # Define theta
        theta = C[mu - C/j > 0][-1] / float(rho)

        # Subtract theta from original vector and cap at 0
        w = np.maximum(v - theta, 0)

        # Return projected vector
        return w

    def learning_rate_t(self, t):
        """Compute current learning rate after decay."""
        # Select rate decay
        if self.rate_decay == 'linear':

            # Linear dropoff between t=0 and t=T
            alpha = (self.max_iter - t)/(self.learning_rate*self.max_iter)

        elif self.rate_decay == 'quadratic':

            # Quadratic dropoff between t=0 and t=T
            alpha = ((self.max_iter - t)/(self.learning_rate*self.max_iter))**2

        elif self.rate_decay == 'geometric':

            # Drop rate off inversely to time
            alpha = 1 / (self.learning_rate * t)

        elif self.rate_decay == 'exponential':

            # Exponential dropoff
            alpha = np.exp(-self.learning_rate * t)

        else:
            raise ValueError('Rate decay type unknown.')

        return alpha

    def risk(self, Z, theta, q):
        """
        Compute target contrastive pessimistic risk.

        INPUT   (1) array 'Z': target samples (M samples by D features)
                (2) array 'theta': classifier parameters (D features by
                    K classes)
                (3) array 'q': soft labels (M samples by K classes)
        OUTPUT  (1) float: risk
        """
        # Number of classes
        K = q.shape[1]

        # Compute negative log-likelihood
        L = self.neg_log_likelihood(Z, theta)

        # Weight loss by soft labels
        for k in range(K):
            L[:, k] *= q[:, k]

        # Sum over weighted losses
        L = np.sum(L, axis=1)

        # Risk is average loss
        return np.mean(L, axis=0)

    def neg_log_likelihood(self, X, theta):
        """
        Compute negative log-likelihood under Gaussian distributions.

        INPUT   (1) array 'X': data (N samples by D features)
                (2) tuple 'theta': containing class proportions 'pi', class
                    means 'mu', and class-covariances 'Si'
        OUTPUT  (1) array 'L': loss (N samples by K classes)
        """
        # Unpack parameters
        pi, mu, Si = theta

        # Check if parameter sets match
        assert pi.shape[1] == mu.shape[0]
        assert mu.shape[1] == Si.shape[0]
        assert Si.shape[0] == Si.shape[1]

        # Number of classes
        K = pi.shape[1]

        # Data shape
        N, D = X.shape

        # Preallocate loss array
        L = np.zeros((N, K))

        for k in range(K):

            # Check for linear or quadratic
            if self.loss == 'lda':

                try:
                    # Probability under k-th Gaussian with shared covariance
                    probs = mvn.pdf(X, mu[k, :], Si)

                except al.LinAlgError as err:
                    print('Covariance matrix is singular. Add regularization.')
                    raise err

            elif self.loss == 'qda':

                try:
                    # Probability under k-th Gaussian with own covariance
                    probs = mvn.pdf(X, mu[k, :], Si[:, :, k])

                except al.LinAlgError as err:
                    print('Covariance matrix is singular. Add regularization.')
                    raise err

            else:
                raise ValueError('Loss unknown.')

            # Negative log-likelihood
            L[:, k] = -np.log(pi[0, k]) - np.log(probs)

        return L

    def discriminant_parameters(self, X, Y):
        """
        Estimate parameters of Gaussian distribution for discriminant analysis.

        INPUT   (1) array 'X': data array (N samples by D features)
                (2) array 'Y': label array (N samples by K classes)
        OUTPUT  (1) array 'pi': class proportions (1 by K classes)
                (2) array 'mu': class means (K classes by D features)
                (3) array 'Si': class covariances (D features D features by
                    K classes)
        """
        # Check labels
        K = Y.shape[1]
        assert K > 1

        # Data shape
        N, D = X.shape

        # Preallocate parameter arrays
        pi = np.zeros((1, K))
        mu = np.zeros((K, D))
        Si = np.zeros((D, D, K))

        # For each class
        for k in range(K):

            # Number of samples for current class
            Nk = np.sum(Y[:, k], axis=0)

            # Check for no samples assigned to certain class
            if Nk == 0:

                # Proportion of samples for current class
                pi[0, k] = 0

                # Mean of current class
                mu[k, :] = np.zeros((1, D))

                # Covariance of current class
                Si[:, :, k] = np.eye(D, D)

            else:

                # Proportion of samples for current class
                pi[0, k] = Nk / N

                # Mean of current class
                mu[k, :] = np.dot(Y[:, k].T, X) / Nk

                # Subtract mean from data
                X_ = X - mu[k, :]

                # Diagonalize current label vector
                dYk = np.diag(Y[:, k])

                # Covariance of current class
                Si[:, :, k] = np.dot(np.dot(X_.T, dYk), X_) / Nk

                # Regularization
                Si[:, :, k] = regularize_matrix(Si[:, :, k], a=self.l2)

        # Check for linear or quadratic discriminant analysis
        if self.loss == 'lda':

            # In LDA, the class-covariance matrices are combined
            Si = self.combine_class_covariances(Si, pi)

        return pi, mu, Si

    def combine_class_covariances(self, Si, pi):
        """
        Linear combination of class covariance matrices.

        INPUT   (1) array 'Si': Covariance matrix (D features by D features by
                    K classes)
                (2) array 'pi': class proportions (1 by K classes)
        OUTPUT  (1) array 'Si': Combined covariance matrix (D by D)
        """
        # Number of classes
        K = Si.shape[2]

        # Check if w is size K
        assert pi.shape[1] == K

        # For each class
        for k in range(K):

            # Weight each class-covariance
            Si[:, :, k] = Si[:, :, k] * pi[0, k]

        # Sum over weighted class-covariances
        return np.sum(Si, axis=2)

    def tcpr_da(self, X, y, Z):
        """
        Target Contrastive Pessimistic Risk - discriminant analysis.

        INPUT   (1) array 'X': source data (N samples by D features)
                (2) array 'y': source labels (N samples by 1)
                (3) array 'Z': target data (M samples by D features)
        OUTPUT  (1) array 'theta': classifier parameters (D features by K
                classes)
        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Assert equivalent dimensionalities
        assert DX == DZ

        # Augment data with bias if necessary
        X, DX = self.remove_intercept(X)
        Z, DZ = self.remove_intercept(Z)

        # Label properties
        classes = np.unique(y)
        K = len(classes)

        # Check for at least 2 classes
        assert K > 1

        # Map labels to one-hot-encoding
        Y = one_hot(y)

        # Estimate parameters of source model
        theta_ref = self.discriminant_parameters(X, Y)

        # Loss is negative log-likelihood under reference parameters
        L_ref = self.neg_log_likelihood(Z, theta_ref)

        # Initialize target posterior
        q = np.ones((M, K)) / K

        print('Starting TCP optimization')

        TCPRt = np.inf
        for t in range(self.max_iter):

            # Maximization phase

            # Estimate parameters using TCP risk
            theta_tcp = self.discriminant_parameters(Z, q)

            # Minimization phase

            # Compute loss under new parameters
            L_tcp = self.neg_log_likelihood(Z, theta_tcp)

            # Gradient is difference in losses
            Dq = L_tcp - L_ref

            # Update learning rate
            alpha = self.learning_rate_t(t)

            # Steepest descent step
            q -= alpha*Dq

            # Project back onto simplex
            for m in range(M):
                q[m, :] = self.project_simplex(q[m, :])

            # Monitor progress

            # Risks of current parameters
            R_tcp = self.risk(Z, theta_tcp, q)
            R_ref = self.risk(Z, theta_ref, q)

            # Assert no numerical problems
            assert not np.isnan(R_tcp)
            assert not np.isnan(R_ref)

            # Current TCP risk
            TCPRt_ = R_tcp - R_ref

            # Change in risk difference for this iteration
            dR = al.norm(TCPRt - TCPRt_)

            # Check for change smaller than tolerance
            if (dR < self.tolerance):
                print('Broke at iteration '+str(t)+', TCP Risk = '+str(TCPRt_))
                break

            # Report progress
            if (t % 100) == 1:
                print('Iteration ' + str(t) + '/' + str(self.max_iter) +
                      ', TCP Risk = ' + str(TCPRt_))

            # Update
            TCPRt = TCPRt_

        # Return estimated parameters
        return theta_tcp

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
        assert DX == DZ

        if self.loss in ['lda', 'qda']:

            # Discriminant analysis model for TCPR
            self.parameters = self.tcpr_da(X, y, Z)

        else:
            # Other loss functions are not implemented
            raise ValueError('Loss function unknown.')

        # Extract and store classes
        self.classes = np.unique(y)

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
            assert D == self.train_data_dim

        if self.loss in ['lda', 'qda']:

            # Compute probabilities under each distribution
            probs = self.neg_log_likelihood(Z_, self.parameters)

            # Take largest probability as predictions
            preds = np.argmax(probs, axis=1)

        # Return predictions array
        return preds

    def get_params(self):
        """Return classifier parameters."""
        # Check if classifier is trained
        if self.is_trained:
            return self.parameters

        else:
            # Throw soft error
            print('Classifier is not trained yet.')
            return []

    def error_rate(self, preds, u_):
        """Compute classification error rate."""
        return np.mean(preds != u_, axis=0)
