# LibTLDA: library of transfer learning and domain-adaptive classifiers
Warning: development is ongoing. All usage is at your own risk.

## Installation:
For Matlab, add the file of the classifier to your path.

## Usage:
- Supply labeled source data (X,yX) and target data (Z) and the methods output a trained linear classifier or a label prediction for Z. <br>

## Python:
Python translations and implementations will arrive shortly.

## Matlab:
Dependencies: <br>
- MinFunc (https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html)

Contains:<br>
- irw: Instance Reweighting <br>
	- Ratio of Gaussians (Shimodaira, 2000) <br>
	- Kernel Mean Matching (Huang et al., 2006) <br>
	- Nearest-neighbour Weighting (Loog, 2015) <br>
- scl: Structural Correspondence Learning (Blitzer et al., 2006) <br>
- gfk: Geodesic Flow Kernel (Gong et al., 2012) <br>
- tca: Transfer Component Analysis (Pan et al, 2009) <br>
- sa: Subspace Alignment (Fernando et al., 2013) <br>
- rba: Robust Bias-Aware (Mansour & Schain, 2014) <br>
- \lambda-svma: \lambda-shift Support Vector Machine Adaptation (Liu & Ziebart, 2014) <br>
- flda: Feature-Level Domain Adaptation (Kouw et al., 2016) <br>

## Contact:
If you have any questions or discover bugs, contact me at wmkouw at gmail dot nl.
