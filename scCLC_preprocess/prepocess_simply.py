import os
import scanpy as sc
import numpy as np

import math

from .epiScanpy_fct import *
from .cal_KNN import cal_nn


def normalize_new(adata, 
              copy=True,
              flavor=None, 
              highly_genes=None, 
              filter_min_counts=True, 
              normalize_input=True, 
              logtrans_input=True,
              scale_input=False):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    else:
        raise NotImplementedError

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_cells=3)

    if normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if flavor == 'seurat_v3':
        print("seurat_v3")
        sc.pp.highly_variable_genes(adata, flavor=flavor, n_top_genes = highly_genes)

    if normalize_input:
        sc.pp.normalize_total(adata, target_sum=1e4)

    if logtrans_input:
        sc.pp.log1p(adata)

    if flavor is None:
        if highly_genes is not None:
            print("routine hvg")
            sc.pp.highly_variable_genes(adata, n_top_genes=highly_genes)
        else:
            sc.pp.highly_variable_genes(adata)
    
    adata_hvg = adata[:, adata.var.highly_variable].copy()

    if scale_input:
        sc.pp.scale(adata_hvg)

    return adata, adata_hvg

def data_process_RNA_new(adata, 
                 num_genes, 
                 RNA_pca,
                 k=6, 
                 max_element=95536, 
                 scale=False):


    adata, adata_hvg = normalize_new(adata, 
                                 copy=True,
                                 flavor=None,
                                 highly_genes=num_genes,
                                 normalize_input=True,
                                 logtrans_input=True,
                                 scale_input=scale)
    
    x_array = adata_hvg.to_df().values
    y_array = adata_hvg.obs['true_label'].values

    print(f"X shape: {x_array.shape}")
    print(f"Y shape: {y_array.shape}")


    return adata, adata_hvg


def data_process_RNA_PCA_KNN_new(adata, 
                 num_genes, 
                 RNA_pca,
                 k=6, 
                 max_element=95536, 
                 scale=False):


    adata, adata_hvg = normalize_new(adata, 
                                 copy=True,
                                 flavor=None,
                                 highly_genes=num_genes,
                                 normalize_input=True,
                                 logtrans_input=True,
                                 scale_input=scale)
    
    x_array = adata_hvg.to_df().values
    y_array = adata_hvg.obs['true_label'].values

    print(f"X shape: {x_array.shape}")
    print(f"Y shape: {y_array.shape}")
    
    x_array = sc.tl.pca(x_array, svd_solver='arpack', n_comps = RNA_pca)
    
    print(f"X shape: {x_array.shape}")
    print(f"Y shape: {y_array.shape}")
    
    if k > 0:
        neighbors_cosine, dis_consine = cal_nn(x_array, k=k, max_element=max_element)
        #neighbors_l2, dis_l2 = cal_nn_l2(x_array, k=k, max_element=max_element)
        
    else:
        return x_array, y_array, None

    return adata, adata_hvg, neighbors_cosine, dis_consine


from sklearn.feature_extraction.text import TfidfTransformer
def normalize_ATAC_new(adata, 
              copy=True,
              flavor=None, 
              min_score_value = None,
              nb_feature_selected=None, 
              filter_min_counts=True, 
              normalize_input=True, 
              logtrans_input=True,
              scale_input=False):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    else:
        raise NotImplementedError

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_cells=1)
        #epi.pp.filter_cells(adata, min_features=100)

    if normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if flavor == 'seurat_v3':
        print("seurat_v3")
        sc.pp.highly_variable_genes(adata, flavor=flavor, n_top_genes = highly_genes)

    # 创建 TfidfTransformer 对象
    tfidf_transformer = TfidfTransformer()

    if normalize_input:
        adata.X = tfidf_transformer.fit_transform(adata.X)
        
    if logtrans_input:
        sc.pp.log1p(adata)

    if flavor is None:
        if nb_feature_selected is not None:
            print("routine hvg")
            adata_hvg = select_var_feature(adata, min_score=min_score_value, nb_features=nb_feature_selected, show=False, copy=True) # nb_feature_selected优先级高于min_score
        else:
            adata_hvg = select_var_feature(adata, min_score=min_score_value,show=False, copy=True)
    
    #adata_hvg = adata[:, adata.var.highly_variable].copy()

    if scale_input:
        sc.pp.scale(adata_hvg)

    return adata, adata_hvg
    

def data_process_ATAC_PCA_KNN_new(adata, 
                 num_peaks, 
                 min_var_score_peak,
                 ATAC_pca,
                 k=6, 
                 max_element=95536, 
                 scale=False):

    
    num_peaks_refine = math.ceil(num_peaks * adata.X.shape[1])
    adata, adata_hvg = normalize_ATAC_new(adata, 
                                 copy=True,
                                 flavor=None,
                                 min_score_value = min_var_score_peak,
                                 nb_feature_selected=num_peaks_refine,
                                 normalize_input=True,
                                 logtrans_input=True,
                                 scale_input=scale)
    
    x_array = adata_hvg.to_df().values
    y_array = adata_hvg.obs['true_label'].values

    print(f"X shape: {x_array.shape}")
    print(f"Y shape: {y_array.shape}")
    
    x_array = sc.tl.pca(x_array, svd_solver='arpack', n_comps = ATAC_pca)
    
    print(f"X shape: {x_array.shape}")
    print(f"Y shape: {y_array.shape}")
   
    if k > 0:
        neighbors_cosine, dis_cosine = cal_nn(x_array, k=k, max_element=max_element)
        #neighbors_l2, _ = cal_nn_l2(x_array, k=k, max_element=max_element)
    else:
        return x_array, y_array, None

    return adata, adata_hvg, neighbors_cosine, dis_cosine


def normalize_ADT_new(
    adata,
    PROT_pca=False,
    pca_num = 10,
    k=6,
    min_cells=1,
    scale=True
):
    adata = adata.copy()
    
    adata.var["n_cells"] = np.array(
        (adata.X > 0).sum(axis=0)
    ).flatten()

    adata = adata[:, adata.var["n_cells"] > min_cells].copy()

    if scale:
        sc.pp.scale(
            adata,
            zero_center=True,
            max_value=None
        )
    if PROT_pca:
        sc.tl.pca(
            adata,
            n_comps=pca_num,
            svd_solver="arpack"
        )

    x_array = adata.X
    y_array = adata.obs["true_label"].values

    print(f"Protein X shape: {x_array.shape}")
    print(f"Protein Y shape: {y_array.shape}")

    return adata


def data_process_ADT_KNN_new(adata, 
                 k=6, 
                 max_element=95536, 
                 scale=True):

    adata = normalize_ADT_new(adata,
                            PROT_pca=False,
                            pca_num = 10,
                            k=6,
                            min_cells=1,
                            scale=True)
    
    x_array = adata.to_df().values
    y_array = adata.obs['true_label'].values

    print(f"X shape: {x_array.shape}")
    print(f"Y shape: {y_array.shape}")
      
    if k > 0:
        neighbors_cosine, dis_consine = cal_nn(x_array, k=k, max_element=max_element)
        #neighbors_l2, dis_l2 = cal_nn_l2(x_array, k=k, max_element=max_element)
        
    else:
        return x_array, y_array, None

    return adata, neighbors_cosine, dis_consine