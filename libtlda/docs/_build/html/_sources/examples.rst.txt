********
Examples
********

In the /demos folder, there are a number of example scripts. These show a potential use case on synthetic data.

Here we walk through a simple version.

First, we import a number of modules and generate a synthetic data set:

.. code-block:: python

    import numpy as np
    import numpy.random as rnd
    
    from sklearn.linear_model import LogisticRegression
    from libtlda.iw import ImportanceWeightedClassifier

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
    si_S = 1.0
    N0 = int(np.round(N*pi_S[0]))
    N1 = N - N0
    X0 = rnd.randn(N0, D)*si_S + (-2, 0)
    X1 = rnd.randn(N1, D)*si_S + (+2, 0)
    X = np.concatenate((X0, X1), axis=0)
    y = np.concatenate((labels[0]*np.ones((N0,), dtype='int'),
                        labels[1]*np.ones((N1,), dtype='int')), axis=0)

    # Target domain
    pi_T = [1./2, 1./2]
    si_T = 3.0
    M0 = int(np.round(M*pi_T[0]))
    M1 = M - M0
    Z0 = rnd.randn(M0, D)*si_T + (-2, -2)
    Z1 = rnd.randn(M1, D)*si_T + (+2, +2)
    Z = np.concatenate((Z0, Z1), axis=0)
    u = np.concatenate((labels[0]*np.ones((M0,), dtype='int'),
                        labels[1]*np.ones((M1,), dtype='int')), axis=0)


Next, we create an adaptive classifier:

.. code-block:: python

    # Call an importance-weighted classifier
    clf = ImportanceWeightedClassifier(iwe='lr', loss='logistic')

    # Train classifier
    clf.fit(X, y, Z)

    # Make predictions
    pred_adapt = clf.predict(Z)

We can compare this with a non-adaptive classifier:

.. code-block:: python

    # Train a naive logistic regressor
    lr = LogisticRegression().fit(X, y)

    # Make predictions
    pred_naive = lr.predict(Z)

And compute error rates:

.. code-block:: python

    # Compute error rates
    print('Error naive: ' + str(np.mean(pred_naive != u, axis=0)))
    print('Error adapt: ' + str(np.mean(pred_adapt != u, axis=0)))