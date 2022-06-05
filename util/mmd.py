from abc import abstractmethod
from numpy import eye, concatenate, zeros, shape, mean, reshape, arange, exp, outer, median, sqrt
from numpy.random import permutation,shuffle
from numpy.lib.index_tricks import fill_diagonal
from matplotlib.pyplot import imshow,show
from scipy.spatial.distance import squareform, pdist, cdist
import numpy as np
from numba import jit, generated_jit, types
import scipy

class Kernel(object):
    def __init__(self):
        pass

    @abstractmethod
    def kernel(self, X, Y=None):
        raise NotImplementedError()
    
    @abstractmethod
    def gradient(self, x, Y):
        
        # ensure this in every implementation
        assert(len(shape(x))==1)
        assert(len(shape(Y))==2)
        assert(len(x)==shape(Y)[1])
        
        raise NotImplementedError()
    
    @staticmethod
    def centring_matrix(n):
        """
        Returns the centering matrix eye(n) - 1.0 / n
        """
        return eye(n) - 1.0 / n
    
    @staticmethod
    def center_kernel_matrix(K):
        """
        Centers the kernel matrix via a centering matrix H=I-1/n and returns HKH
        """
        n = shape(K)[0]
        H = eye(n) - 1.0 / n
        return  1.0 / n * H.dot(K.dot(H))
    
    @abstractmethod
    def show_kernel_matrix(self,X,Y=None):
        K=self.kernel(X,Y)
        imshow(K, interpolation="nearest")
        show()
    
    @abstractmethod
    def estimateMMD(self,sample1,sample2,unbiased=False):
        """
        Compute the MMD between two samples
        """
        K11 = self.kernel(sample1,sample1)
        K22 = self.kernel(sample2,sample2)
        K12 = self.kernel(sample1,sample2)
        if unbiased:
            fill_diagonal(K11,0.0)
            fill_diagonal(K22,0.0)
            n=float(shape(K11)[0])
            m=float(shape(K22)[0])
            return sum(sum(K11))/(pow(n,2)-n) + sum(sum(K22))/(pow(m,2)-m) - 2*mean(K12[:])
        else:
            return mean(K11[:])+mean(K22[:])-2*mean(K12[:])
        
class GaussianKernel(Kernel):
    def __init__(self, sigma):
        Kernel.__init__(self)
        
        self.width = sigma
    
    def kernel(self, X, Y=None):

        if Y is None:
            sq_dists = scipy.spatial.distance.squareform(pdist(X, 'sqeuclidean'))
        else:
            assert(len(np.shape(Y))==2)
            assert(np.shape(X)[1]==np.shape(Y)[1])
            sq_dists = scipy.spatial.distance.cdist(X, Y, 'sqeuclidean')
    
        K = np.exp(-0.5 * (sq_dists) / self.width ** 2)
        return K
    
    @staticmethod
    def get_sigma_median_heuristic(Z, num_subsample=5000):
        
        inds = np.random.permutation(len(Z))[:np.max([num_subsample, len(Z)])]
        dists = squareform(pdist(Z[inds], 'sqeuclidean'))
        median_dist = np.median(dists[dists > 0])
        sigma = np.sqrt(0.5 * median_dist)
        
        return sigma