import os
import scanpy as sc
import numpy as np
from .epiScanpy_fct import *


def normalize(adata, 
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


def normalize_ATAC(adata, 
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

    if normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if flavor == 'seurat_v3':
        print("seurat_v3")
        sc.pp.highly_variable_genes(adata, flavor=flavor, n_top_genes = highly_genes)

    if normalize_input:
        #sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.normalize_total(adata)
        

    if logtrans_input:
        sc.pp.log1p(adata)

    if flavor is None:
        if nb_feature_selected is not None:
            print("routine hvg")
            adata_hvg = select_var_feature(adata, min_score=min_score_value, nb_features=nb_feature_selected, show=False, copy=True)
        else:
            adata_hvg = select_var_feature(adata, min_score=min_score_value,show=False, copy=True)
    
    #adata_hvg = adata[:, adata.var.highly_variable].copy()

    if scale_input:
        sc.pp.scale(adata_hvg)

    return adata, adata_hvg
    


    