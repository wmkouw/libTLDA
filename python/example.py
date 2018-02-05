"""
libTLDA Example script to show how to run an adaptive classifier

Generate synthetic data sets with either normal distributions or Poisson
distributions (for scl and flda)

X = N samples by D features data matrix for source domain
y = N samples by 1 vector of labels in {1,...,K} for source domain
Z = M samples by D features data matrix for target domain
u = M samples by 1 vector of labels in {1,...,K} for target domain

Options for adaptive classifiers:
iw        importance-weighting
suba      subspace alignment
gfk       geodesic flow kernel
tca       transfer component analysis
rba       robust bias-aware
scl       structural correspondence learning
flda      feature-level domain-adaptation

Last update: 25-02-2018
"""

import numpy as np
import numpy.random as rnd
import scipy.stats as st
from sklearn.linear_model import LogisticRegression

from iw import ImportanceWeightedClassifier
# from tca import TransferComponentAnalysisClassifier

"""Select adaptive classifier"""

aclfr = 'iw'

"""Generate synthetic data set"""

# Sample sizes
N = 100
M = 50

# Class properties
labels = [0, 1]
nK = 2

# Dimensionality
D = 2

# Source domain
pi_S = [1./2, 1./2]
N0 = int(np.round(N*pi_S[0]))
N1 = N - N0
X0 = rnd.randn(N0, D) + -1*np.ones((1, D))
X1 = rnd.randn(N1, D) + +1*np.ones((1, D))
X = np.concatenate((X0, X1), axis=0)
y = np.concatenate((labels[0]*np.ones((N0,)),
                    labels[1]*np.ones((N1,))), axis=0)

# Target domain
pi_T = [1./2, 1./2]
M0 = int(np.round(M*pi_T[0]))
M1 = M - M0
Z0 = rnd.randn(M0, D) + -1*np.ones((1, D))
Z1 = rnd.randn(M1, D) + +1*np.ones((1, D))
Z = np.concatenate((Z0, Z1), axis=0)
u = np.concatenate((labels[0]*np.ones((M0,)),
                    labels[1]*np.ones((M1,))), axis=0)

"""Classifiers"""

# Train a naive logistic regressor
lr = LogisticRegression().fit(X, y)

# Make predictions
pred_n = lr.predict(Z)

# Train an adaptive classifier
if aclfr == 'iw':
    # Train an importance-weighted classifier
    _, pred_a = ImportanceWeightedClassifier(iwe='kmm').fit(X, y, Z)

# elif aclfr == 'tca':
#     # Train an adaptive classifier based on transfer component analysis
#     _, pred_a = TransferComponentAnalysisClassifier(X, y)

# Compute error rates
err_naive = np.mean(pred_n != u, axis=0)
err_adapt = np.mean(pred_a != u, axis=0)

# Report results
print('Error naive: ' + str(err_naive))
print('Error adapt: ' + str(err_adapt))
