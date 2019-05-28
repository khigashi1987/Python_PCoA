import numpy as np
from scipy.spatial.distance import squareform
import itertools

def Jaccard(X):
    '''
    compute Jaccard similarity and then convert it into distance by subtracting it from 1.0.
    Args:
      X: input N x K data matrix. N ... the number of samples, K ... the number of features.
    Return:
      N x N data matrix. The value of (i,j) shows the distance between sample-i and sample-j.
    '''
    X = np.array(X)
    n_samples = X.shape[0]
    n_distance = int(n_samples * (n_samples - 1) / 2)
    d_array = np.zeros((n_distance))
    for i, (idx1, idx2) in enumerate(itertools.combinations(range(n_samples),2)):
        v1_nonzero_index = np.flatnonzero(X[idx1])
        v2_nonzero_index = np.flatnonzero(X[idx2])
        intersection = len(np.intersect1d(v1_nonzero_index, v2_nonzero_index))
        union = len(np.union1d(v1_nonzero_index, v2_nonzero_index))
        d_array[i] = 1.0 - (float(intersection) / float(union))
    return squareform(d_array)


def BrayCurtis(X):
    '''
    compute Bray-Curtis dissimilarity.
    Args:
      X: input N x K data matrix. N ... the number of samples, K ... the number of features.
    Return:
      N x N data matrix. The value of (i,j) shows the distance between sample-i and sample-j.
    '''
    from scipy.spatial.distance import braycurtis
    X = np.array(X)
    n_samples = X.shape[0]
    n_distance = int(n_samples * (n_samples - 1) / 2)
    d_array = np.zeros((n_distance))
    for i, (idx1, idx2) in enumerate(itertools.combinations(range(n_samples),2)):
        d_array[i] = braycurtis(X[idx1], X[idx2])
    return squareform(d_array)


def JSDivergence(X):
    '''
    compute Jensen-Shannon divergence.
    Args:
      X: input N x K data matrix. N ... the number of samples, K ... the number of features.
    Return:
      N x N data matrix. The value of (i,j) shows the distance between sample-i and sample-j.
    '''
    from informationTheory import JSdivergence
    X = np.array(X)
    n_samples = X.shape[0]
    n_distance = int(n_samples * (n_samples - 1) / 2)
    d_array = np.zeros((n_distance))
    for i, (idx1, idx2) in enumerate(itertools.combinations(range(n_samples),2)):
        d_array[i] = JSdivergence(X[idx1], X[idx2])
    return squareform(d_array)
