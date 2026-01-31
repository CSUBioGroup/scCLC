

def filter_cell_withoutSNN(data, strong_knn_downSample):
    # 从 strong_knn_downSample 中提取 Row_Index 列
    row_indices = strong_knn_downSample['Row_Index'].values
    # 使用提取出的行索引选择 data 数组中的相应行
    selected_data = data[row_indices]
    
    return selected_data