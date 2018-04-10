## libTLDA: library of transfer learning and domain adaptation classifiers.

[![BuildStatus](https://travis-ci.org/wmkouw/libTLDA.svg?branch=master)](https://travis-ci.org/wmkouw/libTLDA) [![PyPI version](https://badge.fury.io/py/libtlda.svg)](https://badge.fury.io/py/libtlda) ![Python version](https://img.shields.io/badge/python-2.7%2C%203.4%2C%203.5%2C%203.6-blue.svg) ![Matlab version](https://img.shields.io/badge/matlab-R2017a-blue.svg) [![DOI](https://zenodo.org/badge/41360294.svg)](https://zenodo.org/badge/latestdoi/41360294)

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

### Python

#### Installation
Installation can be done through pip:
```shell
pip install libtlda
```

Environment management is generally a good idea. To create a [conda](https://conda.io/docs/) environment, run the following commands:
```
conda env create -f environment.yml
source activate libtlda
```

#### Usage
Libtlda follows a similar logic as [scikit-learn](http://scikit-learn.org/). Each type of adaptive classifier is a submodule, from which the classifiers can be imported:
```python
from libtlda.iw import ImportanceWeightedClassifier
from libtlda.tca import TransferComponentClassifier
from libtlda.suba import SubspaceAlignedClassifier
from libtlda.scl import StructuralCorrespondenceClassifier
from libtlda.rba import RobustBiasAwareClassifier
from libtlda.flda import FeatureLevelDomainAdaptiveClassifier
from libtlda.tcpr import TargetContrastivePessimisticClassifier
```
From there on, training is a matter of calling the `fit` method on your labeled source dataset `(X,y)` and unlabeled target dataset `Z`. For example:
```python
classifier = ImportanceWeightedClassifier().fit(X, y, Z)
```

Predictions can be made by calling the `predict` method:
```python
y_pred = classifier.predict(Z)
```

Documentation will be improved soon. For now, have a look at the `example.py` script. It shows a couple of options for training adaptive classifiers.

<!-- ### Python-specific classifiers
- dann: Domain-Adversarial Neural Network (Ganin et al., 2015) (TODO) -->

### Matlab

#### Installation:
First clone the repository and change directory to matlab:
```shell
git clone https://github.com/wmkouw/libTLDA
cd libTLDA/matlab/
```

In the matlab command window, call the installation script. It downloads all dependencies ([minFunc](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html), [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)) and adds them - along with `libtlda` - to your path:
```MATLAB
install.m
```

#### Usage
There is an example script that can be edited to test the different classifiers:
```MATLAB
example.m
```

### Contact:
Questions, comments and bugs can be submitted in the [issues tracker](https://github.com/wmkouw/libTLDA/issues).
