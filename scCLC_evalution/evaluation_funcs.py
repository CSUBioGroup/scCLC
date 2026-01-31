import os

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from sklearn import metrics
from munkres import Munkres
from sklearn.cluster import KMeans
import igraph as ig

sc.settings.verbosity = 0

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind_row, ind_col = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size


def evaluate_kmeans(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    f = metrics.fowlkes_mallows_score(label, pred)

    pred_adjusted = get_y_preds(label, pred, len(set(label)))
    ACC = metrics.accuracy_score(pred_adjusted, label)

    return nmi, ari, f, ACC


def evaluate_cluster(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    ami = metrics.adjusted_mutual_info_score(label, pred)
    #ACC = acc(label, pred)
    ACC = metrics.accuracy_score(pred, label)
    
    fm = metrics.fowlkes_mallows_score(label, pred)

    return ari, nmi, ami, ACC, fm


def run_leiden_scanpy(adata, resolution=0.6):
    #adata = sc.AnnData(latent_vector, dtype=np.float32)

    #sc.pp.neighbors(adata, n_neighbors=20, n_pcs=0, use_rep='X')
    sc.tl.leiden(adata, resolution=resolution)
    leiden_pred = adata.obs['leiden'].astype("int32")
   
    return adata, leiden_pred

def run_louvain_scanpy(adata, resolution=0.6):
    #adata = sc.AnnData(latent_vector, dtype=np.float32)

    #sc.pp.neighbors(adata, n_neighbors=20, n_pcs=0, use_rep='X')
    sc.tl.louvain(adata, resolution=resolution)
    louvain_pred = adata.obs['louvain'].astype("int32")
        
    return adata, louvain_pred


def run_kmeans(latent_vector, n_clusters, random_state=0):
    kmeans = KMeans(init="k-means++", 
                    n_init=10,
                    max_iter=100,
                    n_clusters=n_clusters, 
                    random_state=random_state)

    pred = kmeans.fit_predict(latent_vector)

    return pred



