# LibTLDA: transfer learning and domain-adaptive classifiers

This package contains the following classifiers: <br>
- iw: Importance-weighted classifier, with weight estimators:<br>
	- Kernel density estimation <br>
	- Ratio of Gaussians (Shimodaira, 2000) <br>
	- Logistic discrimination (Bicket et al., 2009) <br>
	- Kernel Mean Matching (Huang et al., 2006) <br>
	- Nearest-neighbour-based weighting (Loog, 2015) <br>

## Python
Python-2.7 only, at the moment.

### Installation
First clone the repository:
```
sudo apt-get install git
git clone https://github.com/wmkouw/libTLDA
```

Creating a new conda environment takes care of all dependencies:
```
conda env create -f environment.yml
source activate libtlda
```

Afterwards, run the following setup script:
```
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
The installation script downloads all dependencies ([minFunc](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html), [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)) and adds them as well libTLDA to your path (matlab command window):
```
install.m
```

### Usage
Example script - can be edited to test different classifiers (matlab command window):
```
example.m
```

### Matlab-specific classifiers:
- scl: Structural Correspondence Learning (Blitzer et al., 2006) <br>
- gfk: Geodesic Flow Kernel (Gong et al., 2012) <br>
- tca: Transfer Component Analysis (Pan et al, 2009) <br>
- suba: Subspace Alignment (Fernando et al., 2013) <br>
- rba: Robust Bias-Aware (Liu & Ziebart, 2014) <br>
- flda: Feature-Level Domain Adaptation (Kouw et al., 2016) <br>

## Contact:
Questions, comments and bugs can be submitted in the issues tracker.
