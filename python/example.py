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
import matplotlib.pyplot as plt

from iw import ImportanceWeightedClassifier
from tca import TransferComponentClassifier
from suba import SubspaceAlignedClassifier
from scl import StructuralCorrespondenceClassifier
from rba import RobustBiasAwareClassifier
from flda import FeatureLevelDomainAdaptiveClassifier

"""Select adaptive classifier"""

aclfr = 'flda'
viz = False

"""Generate synthetic data set"""

# Sample sizes
N = 100
M = 50

# Class properties
labels = [0, 1]
nK = 2

# Dimensionality
D = 3

# Source domain
pi_S = [1./2, 1./2]
si_S = 1.0
N0 = int(np.round(N*pi_S[0]))
N1 = N - N0
X0 = rnd.randn(N0, D)*si_S + (-2, 0, 0)
X1 = rnd.randn(N1, D)*si_S + (+2, 0, 0)
X = np.concatenate((X0, X1), axis=0)
y = np.concatenate((labels[0]*np.ones((N0,), dtype='int'),
                    labels[1]*np.ones((N1,), dtype='int')), axis=0)

# Target domain
pi_T = [1./2, 1./2]
si_T = 3.0
M0 = int(np.round(M*pi_T[0]))
M1 = M - M0
Z0 = rnd.randn(M0, D)*si_T + (-2, -2, 0)
Z1 = rnd.randn(M1, D)*si_T + (+2, +2, 0)
Z = np.concatenate((Z0, Z1), axis=0)
u = np.concatenate((labels[0]*np.ones((M0,), dtype='int'),
                    labels[1]*np.ones((M1,), dtype='int')), axis=0)

"""Classifiers"""

# Train a naive logistic regressor
lr = LogisticRegression().fit(X, y)

# Make predictions
pred_n = lr.predict(Z)

# Select adaptive classifier
if aclfr == 'iw':

    # Call an importance-weighted classifier
    clf = ImportanceWeightedClassifier(iwe='kde', loss='logistic')

elif aclfr == 'tca':

    # Classifier based on transfer component analysis
    clf = TransferComponentClassifier(loss='logistic', mu=1.)

elif aclfr == 'suba':

    # Classifier based on subspace alignment
    clf = SubspaceAlignedClassifier(loss='logistic')

elif aclfr == 'scl':
    # Classifier based on subspace alignment
    clf = StructuralCorrespondenceClassifier(num_pivots=2, num_components=1)

elif aclfr == 'rba':
    # Robust bias-aware classifier
    clf = RobustBiasAwareClassifier(l2=0.1, max_iter=1000)

elif aclfr == 'flda':
    # Feature-level domain-adaptive classifier
    clf = FeatureLevelDomainAdaptiveClassifier(l2=0.1, max_iter=1000)

else:
    raise ValueError('Adaptive classifier not recognized.')

# Train classifier
clf.fit(X, y, Z)

# Make predictions
pred_a = clf.predict(Z)

# Compute error rates
err_naive = np.mean(pred_n != u, axis=0)
err_adapt = np.mean(pred_a != u, axis=0)

# Report results
print('Error naive: ' + str(err_naive))
print('Error adapt: ' + str(err_adapt))
