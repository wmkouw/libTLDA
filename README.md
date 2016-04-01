# da-tools
Matlab toolbox for domain adaptation algorithms. More methods (and better documentation) will be added soon. All usage is at your own risk.

Dependencies: <br>
- MinFunc (https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html)

Usage: <br>
- Supply labeled source data (X,yX) and target data (Z) and the methods output a trained linear classifier or a label prediction for Z. <br>

Contains:<br>
- irw: Instance Reweighting (includes importance weight estimation methods such as Kernel Mean Matching from Huang et al., 2006) <br>
- scl: Structural Correspondence Learning (Blitzer et al., 2006) <br>
- gfk: Geodesic Flow Kernel (Gong et al., 2012) <br>
- tca: Transfer Component Analysis (Pan et al, 2009) <br>
- sa: Subspace Alignment (Fernando et al., 2013) <br>
- rba: Robust Bias-Aware (Mansour & Schain, 2014) <br>
- \lambda-svma: \lambda-shift Support Vector Machine Adaptation (Liu & Ziebart, 2014) <br>
- flda: Feature-Level Domain Adaptation (under review) <br>
- tce: Target Contrastive Estimator (under review) <br>

Questions:
If you have any questions or discover bugs, contact me at w.m.kouw at tudelft dot nl.
