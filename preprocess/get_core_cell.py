import os
import numpy as np
import scanpy as sc
import random
import math
import hnswlib
import pandas as pd
 
from .cal_KNN import cal_nn
    
# from cake
def get_core_cell(adata, 
               adata_embedding, 
               pseudo_label='leiden', 
               seed=42,
               k=30, 
               percent=0.4, 
               max_element=95536):
    max_element = max(max_element, adata_embedding.shape[0] + 1)
    _, distance = cal_nn(x=adata_embedding.X, k=k, max_element=max_element)
    
    mean_distance = np.mean(distance, axis=1)
    
    dis_col_name = f"{pseudo_label}_distance"
    adata.obs[dis_col_name] = mean_distance
    adata_embedding.obs[dis_col_name] = mean_distance
    
    anchor_cells = []
    for c in np.unique(adata_embedding.obs[pseudo_label]):
        cell_type = adata_embedding.obs[adata_embedding.obs[pseudo_label] == c]
        num_cells = math.ceil(percent * cell_type.shape[0])
        threshold = np.sort(cell_type[dis_col_name].values)[num_cells]
        cells = cell_type.index[np.where(cell_type[dis_col_name] <= threshold)[0]].to_list()
        
        if len(cells) > num_cells:
            random.seed(seed)
            selected_cells = random.sample(cells, num_cells)
        else:
            selected_cells = cells
        
        anchor_cells.extend(selected_cells)
    
    non_anchor_cells = list(set(adata.obs_names) - set(anchor_cells))
    
    adata_embedding.obs.loc[anchor_cells, f"{pseudo_label}_density_status"] = "high"
    adata_embedding.obs.loc[non_anchor_cells, f"{pseudo_label}_density_status"] = "low"
    
    adata.obs.loc[anchor_cells, f"{pseudo_label}_density_status"] = "high"
    adata.obs.loc[non_anchor_cells, f"{pseudo_label}_density_status"] = "low"
    
    return adata, adata_embedding


def get_core_cell_new_sim(adata, 
                  adata_embedding, 
                  pseudo_label='leiden', 
                  seed=42,
                  k=30, 
                  percent=0.2,  # 修改percent为0.2
                  max_element=95536):
    
    max_element = max(max_element, adata_embedding.shape[0] + 1)
   
    anchor_cells = []
    origin_k = k
    for c in np.unique(adata_embedding.obs[pseudo_label]):
        cell_type_indices = adata_embedding.obs[adata_embedding.obs[pseudo_label] == c].index
        #cell_type_indices = adata_embedding.obs.index.get_indexer(cell_type_indices)
        
        # 计算每个细胞到同类细胞的平均距离
        k = origin_k
        
        if len(cell_type_indices) <= k:
            k = len(cell_type_indices)-1


        # 你需要一个映射列表或数组，将整数索引转换为字符串    
        sun_adata = adata_embedding[adata_embedding.obs[pseudo_label] == c].copy()
        #adata = adata[:, adata.var.highly_variable].copy()
        #sun_adata = adata[adata.obs[pseudo_label] == c].copy()
        expr = sun_adata.X
        #print(expr.shape)
        _, distance = cal_nn(x=expr, k=k, max_element=max_element)
        mean_distances = np.mean(distance, axis=1)
        
        var_distances = np.var(distance, axis=1)
        #print('range of mean var')
        """
        # 打印分位数
        percentiles = [0, 25, 50, 75, 100]  # 你可以选择需要的分位数
        percentile_values = np.percentile(mean_distances, percentiles)

        for p, value in zip(percentiles, percentile_values):
            print(f"{p}th percentile: {value}")
            
        percentile_values = np.percentile(var_distances, percentiles)

        for p, value in zip(percentiles, percentile_values):
            print(f"{p}th percentile: {value}")
        
        # 计算综合评分：可以是平均距离和方差的加权组合
        # 这里我们选择一个简单的加权平均作为示例
        combined_score = mean_distances + var_distances
        """
                   
        # 选取top 20%的细胞作为anchor细胞
        #num_cells = math.ceil(percent * cell_type_indices.shape[0])
        num_cells = math.floor(percent * cell_type_indices.shape[0])
        
        threshold = np.sort(mean_distances)[num_cells - 1]  # 使用num_cells - 1作为阈值索引
        
        threshold_var = np.sort(var_distances)[num_cells - 1]  # 使用num_cells - 1作为阈值索引
        
                
        cells = cell_type_indices[np.where((mean_distances <= threshold) & (var_distances <= threshold_var))[0]].tolist()
 
        #cells = cell_type_indices[np.where((mean_distances <= threshold))[0]].tolist()
        
        if len(cells) > num_cells:
            random.seed(seed)
            selected_cells = random.sample(cells, num_cells)
        else:
            selected_cells = cells
        
        anchor_cells.extend(selected_cells)
    
    non_anchor_cells = list(set(adata.obs_names) - set(anchor_cells))
    
    missing_indices = [idx for idx in anchor_cells if idx not in adata_embedding.obs.index]
    if missing_indices:
        print("Missing indices:", missing_indices)

    
    adata_embedding.obs.loc[anchor_cells, f"{pseudo_label}_density_status"] = "high"
    adata_embedding.obs.loc[non_anchor_cells, f"{pseudo_label}_density_status"] = "low"
    
    adata.obs.loc[anchor_cells, f"{pseudo_label}_density_status"] = "high"
    adata.obs.loc[non_anchor_cells, f"{pseudo_label}_density_status"] = "low"
    
    return adata, adata_embedding

def get_core_cell_new_sim_2(adata, 
               adata_embedding, 
               pseudo_label='leiden', 
               seed=42,
               k=30, 
               percent=0.4, 
               max_element=95536):
    max_element = max(max_element, adata_embedding.shape[0] + 1)
    _, distance = cal_nn(x=adata_embedding.X, k=k, max_element=max_element)
    
    mean_distance = np.mean(distance, axis=1)
    
    dis_col_name = f"{pseudo_label}_distance"
    adata.obs[dis_col_name] = mean_distance
    adata_embedding.obs[dis_col_name] = mean_distance
    
    anchor_cells = []
    for c in np.unique(adata_embedding.obs[pseudo_label]):
        cell_type_indices = adata_embedding.obs[adata_embedding.obs[pseudo_label] == c].index
        
        #
        # 计算每个细胞到同类细胞的平均距离
        sub_distance = distance[cell_type_indices][:, cell_type_indices]
        mean_distances = np.mean(sub_distance, axis=1)
        
        # 将计算的mean_distances更新到adata_embedding.obs中
        adata_embedding.obs.loc[cell_type_indices, dis_col_name] = mean_distances        
        #       
        cell_type = adata_embedding.obs[adata_embedding.obs[pseudo_label] == c]        
        num_cells = math.ceil(percent * cell_type.shape[0])
        threshold = np.sort(cell_type[dis_col_name].values)[num_cells]
        cells = cell_type.index[np.where(cell_type[dis_col_name] <= threshold)[0]].to_list()
        
        if len(cells) > num_cells:
            random.seed(seed)
            selected_cells = random.sample(cells, num_cells)
        else:
            selected_cells = cells
        
        anchor_cells.extend(selected_cells)
    
    non_anchor_cells = list(set(adata.obs_names) - set(anchor_cells))
    
    adata_embedding.obs.loc[anchor_cells, f"{pseudo_label}_density_status"] = "high"
    adata_embedding.obs.loc[non_anchor_cells, f"{pseudo_label}_density_status"] = "low"
    
    adata.obs.loc[anchor_cells, f"{pseudo_label}_density_status"] = "high"
    adata.obs.loc[non_anchor_cells, f"{pseudo_label}_density_status"] = "low"
    
    return adata, adata_embedding