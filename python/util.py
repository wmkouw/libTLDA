import numpy as np
import scipy.stats as st

"""
Set of utility functions necessary for different classifiers.
"""


def is_pos_def(X):
    """Check for positive definiteness."""
    return np.all(np.linalg.eigvals(X) > 0)
