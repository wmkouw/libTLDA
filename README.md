[![BuildStatus](https://travis-ci.org/wmkouw/libTLDA.svg?branch=master)](https://travis-ci.org/wmkouw/libTLDA)
# LibTLDA: transfer learning and domain-adaptive classifiers

This package contains the following classifiers: <br>
- Importance-weighted classifier, with weight estimators:<br>
	- Kernel density estimation <br>
	- [Ratio of Gaussians (Shimodaira, 2000)](https://www.sciencedirect.com/science/article/pii/S0378375800001154) <br>
	- [Logistic discrimination (Bickel et al., 2009)](http://www.jmlr.org/papers/v10/bickel09a.html) <br>
	- [Kernel Mean Matching (Huang et al., 2006)](https://papers.nips.cc/paper/3075-correcting-sample-selection-bias-by-unlabeled-data) <br>
	- [Nearest-neighbour-based weighting (Loog, 2015)](http://ieeexplore.ieee.org/document/6349714/) <br>
- [Transfer Component Analysis (Pan et al, 2009)](http://ieeexplore.ieee.org/document/5640675/) <br>
- [Subspace Alignment (Fernando et al., 2013)](https://dl.acm.org/citation.cfm?id=1610094) <br>
- [Structural Correspondence Learning (Blitzer et al., 2006)](https://dl.acm.org/citation.cfm?id=1610094) <br>
- [Robust Bias-Aware (Liu & Ziebart, 2014)](https://papers.nips.cc/paper/5458-robust-classification-under-sample-selection-bias) <br>
- [Feature-Level Domain Adaptation (Kouw et al., 2016)](http://jmlr.org/papers/v17/15-206.html) <br>

## Python
Python-2.7 only, at the moment.

### Installation
First clone and enter the repository:
```
sudo apt-get install git
git clone https://github.com/wmkouw/libTLDA
cd libTLDA/
```

Creating a new conda environment takes care of all dependencies. <br>
First, get conda (skip this step if you already have it):
```
wget http://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
```
Then, create an activate a new environment:
```
conda env create -f environment.yml
source activate libtlda
```

Afterwards, enter python directory and run the following setup script:
```
cd python/
python setup.py install
```

### Usage
Example script - can be edited to test different classifiers:
```
python example.py
```

### Python-specific classifiers
- dann: Domain-Adversarial Neural Network (Ganin et al., 2015) (TODO)

## Matlab
Version: 9.2.0.556344 (R2017a) <br>

### Installation:
First clone the repository and change directory to matlab:
```
sudo apt-get install git
git clone https://github.com/wmkouw/libTLDA
cd libTLDA/matlab/
```

In the matlab command window, call the installation script. It downloads all dependencies ([minFunc](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html), [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)) and adds them - along with libTLDA - to your path (matlab command window):
```
install.m
```

### Usage
In matlab, call the example script (can be edited to test different classifiers):
```
example.m
```

### Matlab-specific classifiers:
- [Geodesic Flow Kernel (Gong et al., 2012)](https://dl.acm.org/citation.cfm?id=1610094) <br>

## Contact:
Questions, comments and bugs can be submitted in the issues tracker.
