import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

def compute_knn(X, k):
    """
    Compute k-nearest neighbors for each point in the dataset.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input data.
    k : int
        Number of nearest neighbors to find.
    
    Returns
    -------
    knn_indices : ndarray of shape (n_samples, k)
        Indices of the k-nearest neighbors for each sample.
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    return indices

def cal_snn(knn_indices):
    """
    Compute the shared nearest neighbor (SNN) graph using Jaccard index.
    
    Parameters
    ----------
    knn_indices : ndarray of shape (n_samples, k)
        Indices of the k-nearest neighbors for each sample.
    
    Returns
    -------
    snn_graph : csr_matrix of shape (n_samples, n_samples)
        The SNN graph represented as a sparse matrix.
    """
    n_samples = knn_indices.shape[0]
    rows = []
    cols = []
    data = []
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            neighbors_i = set(knn_indices[i])
            neighbors_j = set(knn_indices[j])
            intersection = len(neighbors_i & neighbors_j)
            union = len(neighbors_i | neighbors_j)
            jaccard_index = intersection / union if union != 0 else 0
            if jaccard_index > 0:
                rows.append(i)
                cols.append(j)
                data.append(jaccard_index)
                rows.append(j)
                cols.append(i)
                data.append(jaccard_index)
    
    snn_graph = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
    return snn_graph


