# nearest neighbors accuracy
from preprocess.cal_KNN import cal_nn
import numpy as np

def cal_knn_accuracy(latent, labels, n_neighbors=6, n_pools=50, n_samples_per_pool=100):
    '''
        suppose latent, labels boch saved in array

    '''
    def correct_ratio_along_sample(ind):
        return ind.sum()*1.0 / len(ind)
    
    neighbors, dis = cal_nn(latent, k=n_neighbors, max_element=999536)
    # 将浮点数数组转换为整数数组
    neighbors = neighbors.astype(int)
    indices = neighbors[:,:]    

    correct_nn = labels[indices.reshape(-1)].reshape((-1, n_neighbors-1)) == labels.reshape((-1, 1))
    correct_ratio_per_sample = np.apply_along_axis(correct_ratio_along_sample, axis=1, arr=correct_nn)

    if n_pools == 1:
        return np.mean(correct_ratio_per_sample)
    else:
        scores = np.mean(
                [np.mean(correct_ratio_per_sample[np.random.choice(len(correct_ratio_per_sample), size=n_samples_per_pool)])
                for _ in range(n_pools)]
            )
        return scores
    
def cal_knn_accuracy_initial(neighbors, labels, n_neighbors=6, n_pools=50, n_samples_per_pool=100):
    '''
        suppose latent, labels boch saved in array
    '''
    def correct_ratio_along_sample(ind):
        return ind.sum()*1.0 / len(ind)
    
    
    # 将浮点数数组转换为整数数组
    neighbors = neighbors.astype(int)
    indices = neighbors[:,:]    

    correct_nn = labels[indices.reshape(-1)].reshape((-1, n_neighbors-1)) == labels.reshape((-1, 1))
    correct_ratio_per_sample = np.apply_along_axis(correct_ratio_along_sample, axis=1, arr=correct_nn)

    if n_pools == 1:
        return np.mean(correct_ratio_per_sample)
    else:
        scores = np.mean(
                [np.mean(correct_ratio_per_sample[np.random.choice(len(correct_ratio_per_sample), size=n_samples_per_pool)])
                for _ in range(n_pools)]
            )
        return scores
    
def cal_knn_accuracy_core_cell(neighbors, labels,all_label, n_neighbors=6, n_pools=50, n_samples_per_pool=100):
    '''
        suppose latent, labels boch saved in array
    '''
    def correct_ratio_along_sample(ind):
        return ind.sum()*1.0 / len(ind)
    
    
    # 将浮点数数组转换为整数数组
    neighbors = neighbors.astype(int)
    indices = neighbors[:,:]    

    correct_nn = all_label[indices.reshape(-1)].reshape((-1, n_neighbors-1)) == labels.reshape((-1, 1))
    correct_ratio_per_sample = np.apply_along_axis(correct_ratio_along_sample, axis=1, arr=correct_nn)

    if n_pools == 1:
        return np.mean(correct_ratio_per_sample)
    else:
        scores = np.mean(
                [np.mean(correct_ratio_per_sample[np.random.choice(len(correct_ratio_per_sample), size=n_samples_per_pool)])
                for _ in range(n_pools)]
            )
        return scores
    
    
    
def cal_knn_accuracy_from_df(knn_array, labels, sample_ID, n_neighbors=6, n_pools=50, n_samples_per_pool=100):
    """
    Calculate the k-nearest neighbors accuracy from a DataFrame.
    
    Parameters:
    - knn_df: DataFrame, each row contains the nearest neighbors' IDs for a sample.
    - labels: Array-like, labels corresponding to each sample.
    - n_neighbors: Int, the number of neighbors to consider.
    - n_pools: Int, the number of pools for bootstrapping.
    - n_samples_per_pool: Int, the number of samples per pool.
    
    Returns:
    - Float, the accuracy score.
    """
    def correct_ratio_along_sample(ind):
        return ind.sum() * 1.0 / len(ind)
    
    # Convert DataFrame to NumPy array and handle NaN values
    #knn_array = knn_df.to_numpy()
    
    # Convert neighbors to int, filling NaNs with a placeholder value
    neighbors = np.nan_to_num(knn_array, nan=-1).astype(int)

    # Calculate correct nearest neighbors
    correct_nn = labels[neighbors] == labels[sample_ID, np.newaxis]
    correct_nn = np.where(neighbors == -1, False, correct_nn)  # Exclude NaNs from accuracy calculation
    
    # Calculate the correct ratio per sample
    correct_ratio_per_sample = np.apply_along_axis(correct_ratio_along_sample, axis=1, arr=correct_nn)

    if n_pools == 1:
        return np.mean(correct_ratio_per_sample)
    else:
        scores = np.mean([
            np.mean(correct_ratio_per_sample[np.random.choice(len(correct_ratio_per_sample), size=n_samples_per_pool)])
            for _ in range(n_pools)
        ])
        return scores
    
    
def cal_knn_accuracy_from_df_new(knn_array, labels, sample_ID, n_neighbors=6, n_pools=1, n_samples_per_pool=100):
    """
    Calculate the k-nearest neighbors accuracy from a DataFrame.
    
    Parameters:
    - knn_array: Array-like, each row contains the nearest neighbors' IDs for a sample.
    - labels: Array-like, labels corresponding to each sample.
    - n_neighbors: Int, the number of neighbors to consider.
    - n_pools: Int, the number of pools for bootstrapping.
    - n_samples_per_pool: Int, the number of samples per pool.
    
    Returns:
    - Float, the accuracy score.
    """
    def correct_ratio_along_sample(ind, valid_len):
        return ind.sum() * 1.0 / valid_len
    
    # Convert neighbors to int, filling NaNs with a placeholder value
    neighbors = np.nan_to_num(knn_array, nan=-1).astype(int)

    # Calculate correct nearest neighbors
    correct_nn = labels[neighbors] == labels[sample_ID, np.newaxis]
    correct_nn = np.where(neighbors == -1, False, correct_nn)  # Exclude NaNs from accuracy calculation
    
    # Calculate the correct ratio per sample
    correct_ratio_per_sample = []
    for row in range(correct_nn.shape[0]):
        valid_len = (neighbors[row] != -1).sum()
        correct_ratio_per_sample.append(correct_ratio_along_sample(correct_nn[row], valid_len))
    
    correct_ratio_per_sample = np.array(correct_ratio_per_sample)
    percentiles = [0, 25, 50, 75,100]


    if n_pools == 1:
        mean_accuracy = np.mean(correct_ratio_per_sample)
        median_accuracy = np.median(correct_ratio_per_sample)
        percentile_accuracy = np.percentile(correct_ratio_per_sample, percentiles)
        return mean_accuracy, median_accuracy, percentile_accuracy
    else:
        scores = np.mean([
            np.mean(correct_ratio_per_sample[np.random.choice(len(correct_ratio_per_sample), size=n_samples_per_pool)])
            for _ in range(n_pools)
        ])
        return scores
    
    
def cal_info_refineKNN(knn_refine_array, sample_ID, true_label, tb, method_name, args):
    # cal info of refine_knn 
    # 计算每一行中非 NaN 元素的个数
    NoNan_count = np.sum(~np.isnan(knn_refine_array), axis=1)
    all_NoNan_count = np.sum(NoNan_count)
    total_elements = knn_refine_array.size

    nonan_ratio = all_NoNan_count / total_elements
    refine_len_quantiles = np.percentile(NoNan_count, [0, 25, 50, 75, 100])
    refine_len_mean = np.mean(NoNan_count)


    mean_accuracy, median_accuracy, percentile_25_accuracy = cal_knn_accuracy_from_df_new(knn_refine_array, true_label, sample_ID = sample_ID, n_neighbors=6, n_pools=1, n_samples_per_pool=100)
    # save orign and refined knn_acc
    tb.add_row([str(args.dataset_name), method_name, args.epochs, round(mean_accuracy, 4), round(median_accuracy, 4), (' '.join(map(str, percentile_25_accuracy))), str(knn_refine_array.shape), round(nonan_ratio,4), round(refine_len_mean,4), (' '.join(map(str, refine_len_quantiles)))])
    print(tb)  
    
    return tb   