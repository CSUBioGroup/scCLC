import math
import scanpy as sc
import os
import numpy as np
from .epiScanpy_fct import *


from .cal_KNN import cal_nn
from .normalize import normalize_ATAC
from .read_data import *


def data_process_ATAC_KNN(root_dir, 
                 data_type, 
                 dataset_name, 
                 num_peaks, 
                 min_var_score_peak,
                 k=6, 
                 max_element=95536, 
                 scale=True):
    file_name = os.path.join(root_dir, f"{data_type}_dir", dataset_name + "_ATAC")
    print(f"Current Processed Dataset is: {dataset_name}")

    if data_type == "npz":
        X, Y, cell_type = prepare_npz(file_name)
    elif data_type == 'h5_nested':
        X, Y, cell_type = prepare_nested_h5(file_name)
    elif data_type == "h5":
        X, Y, cell_type = prepare_h5(file_name)
    elif data_type == 'h5ad':
        adata_origin, Y, cell_type = prepare_h5ad_ATAC(file_name)
    else:
        raise Exception("Please Input Proper Data Type!")
    
    if data_type == 'h5ad':
        adata = adata_origin
    else:
        adata = sc.AnnData(X, dtype=np.float32)
    adata.obs['Group'] = Y
    adata.obs['annotation'] = cell_type
    
    num_peaks_refine = math.ceil(num_peaks * adata.X.shape[1])
    adata, adata_hvg = normalize_ATAC(adata, 
                                 copy=True,
                                 flavor=None,
                                 min_score_value = min_var_score_peak,
                                 nb_feature_selected=num_peaks_refine,
                                 normalize_input=True,
                                 logtrans_input=True,
                                 scale_input=scale)
    
    x_array = adata_hvg.to_df().values
    y_array = adata_hvg.obs['Group'].values

    print(f"X shape: {x_array.shape}")
    print(f"Y shape: {y_array.shape}")
    
    if k > 0:
        neighbors, dis = cal_nn(x_array, k=k, max_element=max_element)
    else:
        return x_array, y_array, None

    return x_array, y_array, neighbors,dis



def data_process_ATAC_PCA_KNN(root_dir, 
                 data_type, 
                 dataset_name, 
                 num_peaks, 
                 min_var_score_peak,
                 ATAC_pca,
                 k=6, 
                 max_element=95536, 
                 scale=True):
    file_name = os.path.join(root_dir, f"{data_type}_dir", dataset_name + "_ATAC")
    print(f"Current Processed Dataset is: {dataset_name}")

    if data_type == "npz":
        X, Y, cell_type = prepare_npz(file_name)
    elif data_type == 'h5_nested':
        X, Y, cell_type = prepare_nested_h5(file_name)
    elif data_type == "h5":
        X, Y, cell_type = prepare_h5(file_name)
    elif data_type == 'h5ad':
        adata_origin, Y, cell_type = prepare_h5ad_ATAC(file_name)
    else:
        raise Exception("Please Input Proper Data Type!")
    
    if data_type == 'h5ad':
        adata = adata_origin
    else:
        adata = sc.AnnData(X, dtype=np.float32)
    adata.obs['Group'] = Y
    adata.obs['annotation'] = cell_type
    
    num_peaks_refine = math.ceil(num_peaks * adata.X.shape[1])
    

    adata, adata_hvg = normalize_ATAC(adata, 
                                 copy=True,
                                 flavor=None,
                                 min_score_value = min_var_score_peak,
                                 nb_feature_selected=num_peaks_refine,
                                 normalize_input=True,
                                 logtrans_input=True,
                                 scale_input=scale)
    
    x_array = adata_hvg.to_df().values
    y_array = adata_hvg.obs['Group'].values

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

    return x_array, y_array, neighbors_cosine, dis_cosine

def data_process_ATAC_PCA(root_dir, 
                 data_type, 
                 dataset_name, 
                 num_peaks, 
                          ATAC_pca,
                 min_var_score_peak,
                 k=6, 
                 max_element=95536, 
                 scale=True):
    file_name = os.path.join(root_dir, f"{data_type}_dir", dataset_name + "_ATAC")
    print(f"Current Processed Dataset is: {dataset_name}")

    if data_type == "npz":
        X, Y, cell_type = prepare_npz(file_name)
    elif data_type == 'h5_nested':
        X, Y, cell_type = prepare_nested_h5(file_name)
    elif data_type == "h5":
        X, Y, cell_type = prepare_h5(file_name)
    elif data_type == 'h5ad':
        adata_origin, Y, cell_type = prepare_h5ad_ATAC(file_name)
    else:
        raise Exception("Please Input Proper Data Type!")
    
    if data_type == 'h5ad':
        adata = adata_origin
    else:
        adata = sc.AnnData(X, dtype=np.float32)
    adata.obs['Group'] = Y
    adata.obs['annotation'] = cell_type
    
    num_peaks_refine = math.ceil(num_peaks * adata.X.shape[1])
    

    adata, adata_hvg = normalize_ATAC(adata, 
                                 copy=True,
                                 flavor=None,
                                 min_score_value = min_var_score_peak,
                                 nb_feature_selected=num_peaks_refine,
                                 normalize_input=True,
                                 logtrans_input=True,
                                 scale_input=scale)
    
    x_array = adata_hvg.to_df().values
    y_array = adata_hvg.obs['Group'].values

    print(f"X shape: {x_array.shape}")
    print(f"Y shape: {y_array.shape}")
    
    x_array = sc.tl.pca(x_array, svd_solver='arpack', n_comps = ATAC_pca)
    
    print(f"X shape: {x_array.shape}")
    print(f"Y shape: {y_array.shape}")
   
    return x_array, y_array
