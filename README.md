# da-tools
Matlab toolbox for domain adaptation methods. They are organized based on the types of domain dissimilarities for which they are most appropriate.

More methods (and better documentation) will be added soon.

Dependencies:
- MinFunc (https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html)

Usage:
- The functions mWLR and mWLS perform multi-class instance reweighted linear classification. They optimize a loss function and output a number of classes by number of features (plus bias) matrix as a decision rule.
- The functions irw_est_* are methods for estimating the weights (gauss is the original implementation but is impractical in high-dimensional space, log is estimating the weights based on the posterior probabilities of belonging to the target domain and kmm is a more advanced method from Gretton et al. (2009)) 
- The function subalign is performing subspace alignment from Fernando et al. (2013).

Questions:
If you have any questions during implementation, don't hesitate to contact me at w.m.kouw@tudelft.nl
