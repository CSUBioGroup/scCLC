import hnswlib

def cal_nn(x, k=500, max_element=955360):
    p = hnswlib.Index(space='cosine', dim=x.shape[1])
    p.init_index(max_elements=max_element, 
                 ef_construction=600, 
                 random_seed=600,
                 M=100)
    
    p.set_num_threads(20)
    p.set_ef(600)
    p.add_items(x)

    neighbors, distance = p.knn_query(x, k = k)
    neighbors = neighbors[:, 1:]  # 返回样本数*k最近邻维度的数组，其中值为最近邻的索引。自己与自己的相似性为1，最相近，通过neighbors[:, 1:]将其排除
    distance = distance[:, 1:]

    return neighbors, distance