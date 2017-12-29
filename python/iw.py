import numpy as np
import scipy as sp
import scipy.optimize as opt


class iw(object):
    def __init__(self, iwe='lr', loss='qd', l2=0):
        '''
        A class of importance-weighted classifiers for domain adaptation and transfer learning
        '''

        # Importance-weight estimation
        self.iwe = iwe

        # Loss function
        self.loss = loss

        # Regularization
        self.l2 = l2

    def iwe_rg(self, X,Z):
        '''
        Importance weight estimation by ratio of Gaussians
        '''

    def iwe_lr(self, X,Z):
        '''
        Importance weight estimation by logistic discrimination
        '''

    def iwe_kmm(self, X,Z):
        '''
        Importance weight estimation by kernel mean matching
        '''

    def iwe_NN(self, X,Z):
        '''
        Importance weight estimation by nearest-neighbours
        '''



if __name__ == 'main':
