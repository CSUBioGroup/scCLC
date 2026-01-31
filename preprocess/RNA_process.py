import scanpy as sc
import os
import numpy as np


from .cal_KNN import cal_nn
from .normalize import normalize
from .read_data import *


def data_process_RNA_CellOrder(root_dir, 
                 data_type, 
                 dataset_name, 
                 num_genes, 
                 k=6, 
                 max_element=95536, 
                 scale=True):
    file_name = os.path.join(root_dir, f"{data_type}_dir", dataset_name)
    print(f"Current Processed Dataset is: {dataset_name}")

    if data_type == "npz":
        X, Y, cell_type = prepare_npz(file_name)
    elif data_type == 'h5_nested':
        X, Y, cell_type = prepare_nested_h5(file_name)
    elif data_type == "h5":
        X, Y, cell_type = prepare_h5(file_name)
    elif data_type == 'h5ad':
        X, Y, cell_type = prepare_h5ad(file_name)
    else:
        raise Exception("Please Input Proper Data Type!")
    
    adata = sc.AnnData(X, dtype=np.float32)
    adata.obs['Group'] = Y
    adata.obs['annotation'] = cell_type
    
    ## add on 0722
    # 根据 Group 升序排序
    adata.obs = adata.obs.sort_values(by='Group')
    adata = adata[adata.obs.index]  # 根据排序后的索引重新排列 adata
    ####

    adata, adata_hvg = normalize(adata, 
                                 copy=True,
                                 flavor=None,
                                 highly_genes=num_genes,
                                 normalize_input=True,
                                 logtrans_input=True,
                                 scale_input=scale)
    
    x_array = adata_hvg.to_df().values
    y_array = adata_hvg.obs['Group'].values

    print(f"X shape: {x_array.shape}")
    print(f"Y shape: {y_array.shape}")

    return x_array, y_array, adata, adata_hvg

def data_process_RNA(root_dir, 
                 data_type, 
                 dataset_name, 
                 num_genes, 
                 k=6, 
                 max_element=95536, 
                 scale=True):
    file_name = os.path.join(root_dir, f"{data_type}_dir", dataset_name)
    print(f"Current Processed Dataset is: {dataset_name}")

    if data_type == "npz":
        X, Y, cell_type = prepare_npz(file_name)
    elif data_type == 'h5_nested':
        X, Y, cell_type = prepare_nested_h5(file_name)
    elif data_type == "h5":
        X, Y, cell_type = prepare_h5(file_name)
    elif data_type == 'h5ad':
        X, Y, cell_type = prepare_h5ad(file_name)
    else:
        raise Exception("Please Input Proper Data Type!")
    
    adata = sc.AnnData(X, dtype=np.float32)
    adata.obs['Group'] = Y
    adata.obs['annotation'] = cell_type


    adata, adata_hvg = normalize(adata, 
                                 copy=True,
                                 flavor=None,
                                 highly_genes=num_genes,
                                 normalize_input=True,
                                 logtrans_input=True,
                                 scale_input=scale)
    
    x_array = adata_hvg.to_df().values
    y_array = adata_hvg.obs['Group'].values

    print(f"X shape: {x_array.shape}")
    print(f"Y shape: {y_array.shape}")

    return x_array, y_array, adata, adata_hvg

def data_process_RNA_KNN(root_dir, 
                 data_type, 
                 dataset_name, 
                 num_genes, 
                 k=6, 
                 max_element=95536, 
                 scale=True):
    file_name = os.path.join(root_dir, f"{data_type}_dir", dataset_name)
    print(f"Current Processed Dataset is: {dataset_name}")

    if data_type == "npz":
        X, Y, cell_type = prepare_npz(file_name)
    elif data_type == 'h5_nested':
        X, Y, cell_type = prepare_nested_h5(file_name)
    elif data_type == "h5":
        X, Y, cell_type = prepare_h5(file_name)
    elif data_type == 'h5ad':
        X, Y, cell_type = prepare_h5ad(file_name)
    else:
        raise Exception("Please Input Proper Data Type!")
    
    adata = sc.AnnData(X, dtype=np.float32)
    adata.obs['Group'] = Y
    adata.obs['annotation'] = cell_type

    adata, adata_hvg = normalize(adata, 
                                 copy=True,
                                 flavor=None,
                                 highly_genes=num_genes,
                                 normalize_input=True,
                                 logtrans_input=True,
                                 scale_input=scale)
    
    x_array = adata_hvg.to_df().values
    y_array = adata_hvg.obs['Group'].values

    print(f"X shape: {x_array.shape}")
    print(f"Y shape: {y_array.shape}")
    
    if k > 0:
        neighbors_cosine, dis_consine = cal_nn(x_array, k=k, max_element=max_element)
        #neighbors_l2, dis_l2 = cal_nn_l2(x_array, k=k, max_element=max_element)
        
    else:
        return x_array, y_array, None

    return x_array, y_array, neighbors_cosine, dis_consine


def data_process_RNA_PCA(root_dir, 
                 data_type, 
                 dataset_name, 
                 num_genes, 
                     RNA_pca,
                 k=6, 
                 max_element=95536, 
                 scale=True):
    file_name = os.path.join(root_dir, f"{data_type}_dir", dataset_name)
    print(f"Current Processed Dataset is: {dataset_name}")

    if data_type == "npz":
        X, Y, cell_type = prepare_npz(file_name)
    elif data_type == 'h5_nested':
        X, Y, cell_type = prepare_nested_h5(file_name)
    elif data_type == "h5":
        X, Y, cell_type = prepare_h5(file_name)
    elif data_type == 'h5ad':
        X, Y, cell_type = prepare_h5ad(file_name)
    else:
        raise Exception("Please Input Proper Data Type!")
    
    adata = sc.AnnData(X, dtype=np.float32)
    adata.obs['Group'] = Y
    adata.obs['annotation'] = cell_type

    adata, adata_hvg = normalize(adata, 
                                 copy=True,
                                 flavor=None,
                                 highly_genes=num_genes,
                                 normalize_input=True,
                                 logtrans_input=True,
                                 scale_input=scale)
    
    x_array = adata_hvg.to_df().values
    y_array = adata_hvg.obs['Group'].values

    print(f"X shape: {x_array.shape}")
    print(f"Y shape: {y_array.shape}")
    
    x_array = sc.tl.pca(x_array, svd_solver='arpack', n_comps = RNA_pca)
    
    print(f"X shape: {x_array.shape}")
    print(f"Y shape: {y_array.shape}")
    

    return x_array, y_array

def data_process_RNA_PCA_KNN(root_dir, 
                 data_type, 
                 dataset_name, 
                 num_genes, 
                 RNA_pca,
                 k=6, 
                 max_element=95536, 
                 scale=True):
    file_name = os.path.join(root_dir, f"{data_type}_dir", dataset_name)
    print(f"Current Processed Dataset is: {dataset_name}")

    if data_type == "npz":
        X, Y, cell_type = prepare_npz(file_name)
    elif data_type == 'h5_nested':
        X, Y, cell_type = prepare_nested_h5(file_name)
    elif data_type == "h5":
        X, Y, cell_type = prepare_h5(file_name)
    elif data_type == 'h5ad':
        X, Y, cell_type = prepare_h5ad(file_name)
    else:
        raise Exception("Please Input Proper Data Type!")
    
    adata = sc.AnnData(X, dtype=np.float32)
    adata.obs['Group'] = Y
    adata.obs['annotation'] = cell_type
    
    ## add on 0722
    # 根据 Group 升序排序
    adata.obs = adata.obs.sort_values(by='Group')
    adata = adata[adata.obs.index]  # 根据排序后的索引重新排列 adata
    ####

    adata, adata_hvg = normalize(adata, 
                                 copy=True,
                                 flavor=None,
                                 highly_genes=num_genes,
                                 normalize_input=True,
                                 logtrans_input=True,
                                 scale_input=scale)
    
    x_array = adata_hvg.to_df().values
    y_array = adata_hvg.obs['Group'].values

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

    return x_array, y_array, neighbors_cosine, dis_consine