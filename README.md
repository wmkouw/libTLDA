[![BuildStatus](https://travis-ci.org/wmkouw/libTLDA.svg?branch=master)](https://travis-ci.org/wmkouw/libTLDA)
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
- scl: Structural Correspondence Learning (Blitzer et al., 2006) <br>
- gfk: Geodesic Flow Kernel (Gong et al., 2012) <br>
- tca: Transfer Component Analysis (Pan et al, 2009) <br>
- suba: Subspace Alignment (Fernando et al., 2013) <br>
- rba: Robust Bias-Aware (Liu & Ziebart, 2014) <br>
- flda: Feature-Level Domain Adaptation (Kouw et al., 2016) <br>

## Contact:
Questions, comments and bugs can be submitted in the issues tracker.
