import pandas as pd
import numpy as np

def calculate_knn_intersection_and_complements(A, B):
    # A, B是cal_NN函数的输出，knn矩阵
    def get_intersection(row1, row2):
        return list(set(row1).intersection(set(row2)))

    def get_complement(row, intersection):
        return list(set(row) - set(intersection))
    
    # Initialize empty lists to store the results
    strong_knn = []
    weak_case1_A = []
    weak_case1_B = []
    
    # Iterate over each row in A and B
    for i in range(len(A)):
        intersection = get_intersection(A.iloc[i], B.iloc[i])
        complement_A = get_complement(A.iloc[i], intersection)
        complement_B = get_complement(B.iloc[i], intersection)
        
        strong_knn.append([i] + intersection)
        weak_case1_A.append([i] + complement_A)
        weak_case1_B.append([i] + complement_B)
    
    # Create dataframes from the results
    max_len = max(len(max(strong_knn, key=len)), len(max(weak_case1_A, key=len)), len(max(weak_case1_B, key=len)))
    
    strong_knn_df = pd.DataFrame([x + [None] * (max_len - len(x)) for x in strong_knn], columns=["Row_Index"] + [f"Neighbor_{j+1}" for j in range(max_len-1)])
    weak_case1_A_df = pd.DataFrame([x + [None] * (max_len - len(x)) for x in weak_case1_A], columns=["Row_Index"] + [f"Neighbor_{j+1}" for j in range(max_len-1)])
    weak_case1_B_df = pd.DataFrame([x + [None] * (max_len - len(x)) for x in weak_case1_B], columns=["Row_Index"] + [f"Neighbor_{j+1}" for j in range(max_len-1)])
    
    # Remove rows where all neighbors are NaN
    strong_knn_df = strong_knn_df.dropna(how='all', subset=strong_knn_df.columns[1:])
    weak_case1_A_df = weak_case1_A_df.dropna(how='all', subset=weak_case1_A_df.columns[1:])
    weak_case1_B_df = weak_case1_B_df.dropna(how='all', subset=weak_case1_B_df.columns[1:])
    
    # Convert Neighbor columns to int
    strong_knn_df.iloc[:, 1:] = strong_knn_df.iloc[:, 1:].astype(pd.Int64Dtype())
    weak_case1_A_df.iloc[:, 1:] = weak_case1_A_df.iloc[:, 1:].astype(pd.Int64Dtype())
    weak_case1_B_df.iloc[:, 1:] = weak_case1_B_df.iloc[:, 1:].astype(pd.Int64Dtype())
    
    return strong_knn_df, weak_case1_A_df, weak_case1_B_df

def knn_filter_and_downsample(strong_knn_df, weak_case1_A_df, weak_case1_B_df, c):
    ## here knn_df with row_index column, so need minus 1 in downsample_row function
    # Step 1: Find common row indices
    common_indices = set(strong_knn_df['Row_Index']).intersection(set(weak_case1_A_df['Row_Index'])).intersection(set(weak_case1_B_df['Row_Index']))
    
    # Step 2: Filter dataframes to keep only common indices
    strong_knn_df = strong_knn_df[strong_knn_df['Row_Index'].isin(common_indices)].reset_index(drop=True)
    weak_case1_A_df = weak_case1_A_df[weak_case1_A_df['Row_Index'].isin(common_indices)].reset_index(drop=True)
    weak_case1_B_df = weak_case1_B_df[weak_case1_B_df['Row_Index'].isin(common_indices)].reset_index(drop=True)
    
    # Function to downsample each row
    def downsample_row(row, c):
        row = row.dropna()
        row_index = int(row.iloc[0])
        non_null_elements = row[1:]
        sampled_elements = np.random.choice(non_null_elements, c, replace=False)
        return [row_index] + list(sampled_elements)
    
    # Step 3: Determine the value of c
    min_non_null = min(strong_knn_df.apply(lambda x: x.count(), axis=1).min(),
                       weak_case1_A_df.apply(lambda x: x.count(), axis=1).min(),
                       weak_case1_B_df.apply(lambda x: x.count(), axis=1).min()) - 1  # Subtract 1 to exclude the row index
    
    c = min(c, min_non_null)

    # Step 4: Apply downsample_row function to each dataframe
    strong_knn_downSample = strong_knn_df.apply(downsample_row, axis=1, c=c)
    weak_case1_A_downSample = weak_case1_A_df.apply(downsample_row, axis=1, c=c)
    weak_case1_B_downSample = weak_case1_B_df.apply(downsample_row, axis=1, c=c)
    
    # Convert the results to dataframes and int type
    strong_knn_downSample = pd.DataFrame(strong_knn_downSample.tolist(), columns=["Row_Index"] + [f"Neighbor_{j+1}" for j in range(c)]).astype(int)
    weak_case1_A_downSample = pd.DataFrame(weak_case1_A_downSample.tolist(), columns=["Row_Index"] + [f"Neighbor_{j+1}" for j in range(c)]).astype(int)
    weak_case1_B_downSample = pd.DataFrame(weak_case1_B_downSample.tolist(), columns=["Row_Index"] + [f"Neighbor_{j+1}" for j in range(c)]).astype(int)
    
    snn_row_indices = strong_knn_downSample['Row_Index'].values
    
    strong_knn_downSample = strong_knn_downSample.iloc[:, 1:].values
    weak_case1_A_downSample = weak_case1_A_downSample.iloc[:, 1:].values
    weak_case1_B_downSample = weak_case1_B_downSample.iloc[:, 1:].values

    return strong_knn_downSample, weak_case1_A_downSample, weak_case1_B_downSample, snn_row_indices

def knn_downsample_old(strong_knn_df, c):
    # here knn_df without row_index column    
    # Function to downsample each row
    def downsample_row(row, c):
        row = row.dropna()
        non_null_elements = row[0:]
        sampled_elements = np.random.choice(non_null_elements, c, replace=False)
        return list(sampled_elements)
    
    # Step 3: Determine the value of c
    min_non_null = min(strong_knn_df.apply(lambda x: x.count(), axis=1).min()) # Subtract 1 to exclude the row index
    
    c = min(c, min_non_null)

    # Step 4: Apply downsample_row function to each dataframe
    strong_knn_downSample = strong_knn_df.apply(downsample_row, axis=1, c=c)
    
    # Convert the results to dataframes and int type
    strong_knn_downSample = pd.DataFrame(strong_knn_downSample.tolist(), columns=[f"Neighbor_{j+1}" for j in range(c)]).astype(int)
    
    strong_knn_downSample = strong_knn_downSample.iloc[:, 1:].values

    return strong_knn_downSample

def knn_downsample(strong_knn_array, c):
    # Function to downsample each row
    def downsample_row(row, c):
        row = row[~np.isnan(row)]
        sampled_elements = np.random.choice(row, c, replace=False)
        return sampled_elements
    
    # Step 3: Determine the value of c
    min_non_null = np.min(np.sum(~np.isnan(strong_knn_array), axis=1))
    c = min(c, min_non_null)

    # Step 4: Apply downsample_row function to each row
    strong_knn_downSample = np.apply_along_axis(downsample_row, axis=1, arr=strong_knn_array, c=c)
    strong_knn_downSample = strong_knn_downSample.astype(int)
    
    return strong_knn_downSample

def knn_to_pairMatrix_old(strong_knn_df, sample_id):
    pairs = []
    for index, row in strong_knn_df.iterrows():       
        neighbors = row[0:].dropna().tolist()  # 获取所有非空的最近邻ID
        for neighbor in neighbors:
            pairs.append([sample_id, neighbor])

    pair_matrix = np.array(pairs, dtype=int)
    return pair_matrix

def knn_to_pairMatrix_old2(strong_knn_array, sample_id):
    pairs = []

    # 遍历 NumPy 数组的每一行
    for row in strong_knn_array:
        # 获取所有非空的最近邻ID
        neighbors = row[~np.isnan(row)].tolist()
        for neighbor in neighbors:
            pairs.append([sample_id, neighbor])
            
    # 打印列表长度
    print(f"列表长度: {len(pairs)}")

    # 打印列表的前几项 (例如前5项)
    print(f"列表的前5项: {pairs[:5]}")
    
    pair_matrix = np.array(pairs, dtype=int)
    return pair_matrix

import numpy as np

def knn_to_pairMatrix(strong_knn_array):
    pairs = []

    # 遍历 NumPy 数组的每一行
    for i, row in enumerate(strong_knn_array):
        # 获取第i行的最近邻样本ID列表
        neighbors = row[~np.isnan(row)].astype(int)
        # 将第i行样本与其最近邻样本构成样本对
        for neighbor in neighbors:
            pairs.append([i, neighbor])

    # 将样本对列表转换为 NumPy 数组
    pair_matrix = np.array(pairs)

    return pair_matrix

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture

def refine_pos_pair(emb, pairs, fltr='gmm', yita=.5):
    cos_sim = np.sum(emb[pairs[:, 0]] * emb[pairs[:, 1]], axis=1)  # dot prod

    if fltr=='gmm':    
        sim_pairs = cos_sim.reshape(-1, 1)
        gm = GaussianMixture(n_components=2, random_state=0).fit(sim_pairs)

        gmm_c = gm.predict(sim_pairs)
        gmm_p = gm.predict_proba(sim_pairs)

        # take the major component
        _, num_c = np.unique(gmm_c, return_counts=True)  
        c = np.argmax(num_c)

        filter_mask = gmm_p[:, c]>=yita
    # if filter is not gmm => naive filter
    # given similarity, taking quantile
    else:
        filter_thr = np.quantile(cos_sim, yita)   
        filter_mask = cos_sim >= filter_thr

    pairs = pairs[filter_mask]

    return pairs

from collections import defaultdict

def pairMatrix_to_knn(pair_matrix):
    # Step 1: Collect neighbors for each sample
    neighbors_dict = defaultdict(list)
    
    for i, j in pair_matrix:
        neighbors_dict[i].append(j)
    
    # Step 2: Determine the maximum number of neighbors
    # Check if there are any valid neighbors

    max_neighbors = max(len(neighbors) for neighbors in neighbors_dict.values())
    
    # Step 3: Create DataFrame with samples and their neighbors
    data = []
    for sample_id in sorted(neighbors_dict.keys()):
        neighbors = neighbors_dict[sample_id]
        row = neighbors + [None] * (max_neighbors - len(neighbors))
        data.append(row)
    
    # Create column names
    columns = [f"Neighbor_{i+1}" for i in range(max_neighbors)]
    
    # Step 4: Create DataFrame
    knn_df = pd.DataFrame(data, columns=columns)
    
    
    return knn_df
