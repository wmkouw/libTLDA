[![BuildStatus](https://travis-ci.org/wmkouw/libTLDA.svg?branch=master)](https://travis-ci.org/wmkouw/libTLDA)
## libTLDA: library of transfer learning and domain adaptation classifiers.

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

### Python

#### Installation
Installation can be done through pip:
```
pip install libtlda
```

Environment management is generally a good idea. To create a [conda](https://conda.io/docs/) environment, run the following commands:
```
conda env create -f environment.yml
source activate libtlda
```

#### Usage
Libtlda follows a similar logic as [scikit-learn](http://scikit-learn.org/_). Each type of adaptive classifier is a submodule, from which the classifiers can be imported:
```python
from libtlda.iw import ImportanceWeightedClassifier
from libtlda.tca import TransferComponentClassifier
from libtlda.suba import SubspaceAlignedClassifier
from libtlda.scl import StructuralCorrespondenceClassifier
from libtlda.rba import RobustBiasAwareClassifier
from libtlda.flda import FeatureLevelDomainAdaptiveClassifier
```
From there on, training is a matter of calling the fit method on your labeled source dataset `(X,y)` and unlabeled target dataset `Z`. For example:
```
parameters = ImportanceWeightedClassifier().fit(X, y, Z)
```

Documentation will be improved soon. For now, have a look at the `example.py` script. It shows a couple of options for training adaptive classifiers.

<!-- ### Python-specific classifiers
- dann: Domain-Adversarial Neural Network (Ganin et al., 2015) (TODO) -->

### Matlab
Version: 9.2.0.556344 (R2017a) <br>

#### Installation:
First clone the repository and change directory to matlab:
```shell
git clone https://github.com/wmkouw/libTLDA
cd libTLDA/matlab/
```

In the matlab command window, call the installation script. It downloads all dependencies ([minFunc](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html), [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)) and adds them - along with `libtlda` - to your path:
```
install.m
```

#### Usage
There is an example script that can be edited to test the different classifiers:
```
example.m
```

#### Matlab-specific classifiers:
- Geodesic Flow Kernel [(Gong et al., 2012)](https://dl.acm.org/citation.cfm?id=1610094) <br>

### Contact:
Questions, comments and bugs can be submitted in the [issues tracker](https://github.com/wmkouw/libTLDA/issues).
