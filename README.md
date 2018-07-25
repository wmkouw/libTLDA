# libTLDA

[![Coverage](https://scrutinizer-ci.com/g/wmkouw/libTLDA/badges/coverage.png?b=master)](https://scrutinizer-ci.com/g/wmkouw/libTLDA/statistics/) [![BuildStatus](https://travis-ci.org/wmkouw/libTLDA.svg?branch=master)](https://travis-ci.org/wmkouw/libTLDA) [![docs](https://media.readthedocs.org/static/projects/badges/passing-flat.svg)](https://libtlda.readthedocs.io/en/latest/) [![DOI](https://zenodo.org/badge/41360294.svg)](https://zenodo.org/badge/latestdoi/41360294)

## Library of transfer learners and domain-adaptive classifiers.
This package contains the following classifiers:
- Importance-weighted classifier, with weight estimators:<br>
	- Kernel density estimation <br>
	- Ratio of Gaussians [(Shimodaira, 2000)](https://www.sciencedirect.com/science/article/pii/S0378375800001154) <br>
	- Logistic discrimination [(Bickel et al., 2009)](http://www.jmlr.org/papers/v10/bickel09a.html) <br>
	- Kernel Mean Matching [(Huang et al., 2006)](https://papers.nips.cc/paper/3075-correcting-sample-selection-bias-by-unlabeled-data) <br>
	- Nearest-neighbour-based weighting [(Loog, 2015)](http://ieeexplore.ieee.org/document/6349714/) <br>
- Transfer Component Analysis [(Pan et al, 2009)](http://ieeexplore.ieee.org/document/5640675/) <br>
- Subspace Alignment [(Fernando et al., 2013)](https://dl.acm.org/citation.cfm?id=1610094) <br>
- Structural Correspondence Learning [(Blitzer et al., 2006)](https://dl.acm.org/citation.cfm?id=1610094) <br>
- Robust Bias-Aware [(Liu & Ziebart, 2014)](https://papers.nips.cc/paper/5458-robust-classification-under-sample-selection-bias) <br>
- Feature-Level Domain Adaptation [(Kouw et al., 2016)](http://jmlr.org/papers/v17/15-206.html) <br>

#### Python-specific classifiers:
- Target Contrastive Pessimistic Risk [(Kouw et al., 2017)](https://arxiv.org/abs/1706.08082)

#### Matlab-specific classifiers:
- Geodesic Flow Kernel [(Gong et al., 2012)](https://dl.acm.org/citation.cfm?id=1610094)

## Python
![Python version](https://img.shields.io/badge/python-2.7%2C%203.4%2C%203.5%2C%203.6-blue.svg)

#### Installation

Installation can be done through pip:
```shell
pip install libtlda
```

The pip package installs all dependencies. To ensure that these dependencies that don't mess up your current python environment, you should set up a virtual environment. If you're using [conda](https://conda.io/docs/), this can be taken care of by running:
```
conda env create -f environment.yml
source activate libtlda
```

#### Usage

LibTLDA follows a similar structure as [scikit-learn](http://scikit-learn.org/). There are several classes of classifiers that can be imported through for instance:

```python
from libtlda.iw import ImportanceWeightedClassifier
```

With a data set of labeled source samples `(X,y)` and unlabeled target samples `Z`, the classifier can be called and trained using:

```python
clf = ImportanceWeightedClasssifier().fit(X, y, Z)
```

Given a trained classifier, predictions can be made as follows:
```python
predictions = clf.predict(Z)
```

Check the [documentation](https://libtlda.readthedocs.io/en/latest/) for more information on specific classes, methods and functions.

## Matlab
![Matlab version](https://img.shields.io/badge/matlab-R2017a-blue.svg)

#### Installation:

First clone the repository and change directory to matlab:
```shell
git clone https://github.com/wmkouw/libTLDA
cd libTLDA/matlab/
```

In the matlab command window, call the installation script. It downloads all dependencies ([minFunc](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html), [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)) and adds them, along with `libtlda`, to your path:
```MATLAB
install.m
```

#### Usage

There is an example script that can be edited to test the different classifiers:
```MATLAB
example.m
```

## Contact:

Questions, comments and bugs can be submitted in the [issues tracker](https://github.com/wmkouw/libTLDA/issues). If you have a particular method / algorithm / technique in mind that you feel should be included, you can also submit this in the tracker.
