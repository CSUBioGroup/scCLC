
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
 
import os
import argparse
import time
import pandas as pd
import prettytable as pt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


from scCLC_preprocess import data_process_RNA_PCA_KNN_new, data_process_ATAC_PCA_KNN_new, data_process_RNA_new
from preprocess import *
from preprocess.utils import *
from scCLC_models import train_model_new_PosSimWeight, Encoder_AE, ContrastiveLearning_AE_new_ReviseLoss
from scCLC_evalution import save_core_cell_acc
from scCLC_evalution import *
from scCLC_evalution import inference_evalidate, cal_knn_accuracy

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
import torch
from torch import nn
from torch.utils.data import DataLoader

from pathlib import Path

def main(dname, classnum, learning_rate, K, T, alpha, num_genes, lams, p):
    parser = argparse.ArgumentParser()
    # 获取脚本所在目录
    SCRIPT_DIR = Path(__file__).parent
    DEMO_DATA_DIR = SCRIPT_DIR / "demo_data"
    NEIGHBOR_SAVE_DIR = DEMO_DATA_DIR / "Neighbor_save"
    
    config_path = DEMO_DATA_DIR / "config_Data.yaml"
    print(f"配置文件路径: {config_path}")
    config = yaml_config_hook(str(config_path))

    for k, v in config.items():      
        parser.add_argument(f"--{k}", default=v, type=type(v))
        
    args = parser.parse_args()       
    args.model_path = "/states_0307/" + dname       
    
    args.seed = 301
    args.epochs = 200
    args.n = 6
    args.c = 1
    args.p = p   
    args.classnum = classnum
    args.dataset_name = str(dname)
    args.data_type = "h5ad" # h5_nested
    args.learning_rate = learning_rate
    args.K = K
    args.batch_size = 128        
    args.num_genes = num_genes
    args.latent_feature = [512, 128]
    args.num_peaks = 0.75
    args.T = T
    args.min_var_score_peak = 0.515
    
    args.pca = 500
    args.repeat = 1
    args.root_dir = str(SCRIPT_DIR / "demo_data") + "/"
    #args.resolution = leiden_re_RNA
    args.RNA_pca = 50
    args.ATAC_pca = 50
 
        
    # load label
        ##### 
    print(args.dataset_name)
    # 设置数据路径
    path2 = str(DEMO_DATA_DIR) + "/"
    path1 = str(DEMO_DATA_DIR) + "/"
    label=pd.read_csv(path1 + f"{args.dataset_name}" + "_cluster_typeName.tsv",sep='\t',header=0)
    type_name=np.array(label['cell_type'])
    y=np.array(label['cluster_id'])  #PBMC10_scMVP  lymph SHARE sci_CAR
    
    args.cell_type_name = type_name
    
    # setting paras
    epoch_check_set = ([-1])
    clus_method_set = {'kmeans': True, 'leiden': False, 'louvain': False}

    # loadrna-seq data
    gene_count = sc.read_mtx("{}".format(path2 + args.dataset_name + "/Processed_dataset/RNA/matrix.mtx"))
    gene_count.shape
    gene_count = gene_count.transpose()
    print(gene_count.shape)

    # 添加行列名
    gene_names = pd.read_csv(path2 + args.dataset_name + "/Processed_dataset/RNA/genes.tsv", header=None)[0].tolist()
    rna_cell_names = pd.read_csv(path2 + args.dataset_name + "/Processed_dataset/RNA/barcodes.tsv", header=None)[0].tolist()

    rna_adata = sc.AnnData(gene_count)
    rna_adata.obs['true_label'] = y
    rna_adata.obs['annotation'] = type_name

    # 设置行列名
    rna_adata.obs_names = rna_cell_names
    rna_adata.var_names = gene_names

    adata_norm, adata_hvg_norm = data_process_RNA_new(rna_adata, args.num_genes, args.RNA_pca, args.n, max_element=95536, scale=False)
    
    data = adata_hvg_norm.X.toarray()
    true_label = adata_hvg_norm.obs['true_label'].values
    
    neighbors_simulate_dir = str(NEIGHBOR_SAVE_DIR) + "/"
    dis_simulate_dir = str(NEIGHBOR_SAVE_DIR) + "/"
    
    neighbors_save_path = os.path.join(neighbors_simulate_dir, f"{args.dataset_name}_wsnn_knn_rna.tsv")      
    dis_save_path = os.path.join(dis_simulate_dir, f"{args.dataset_name}_wsnn_knn_weight_rna.tsv")  
    
    dis_cosine_RNA = np.loadtxt(dis_save_path)
    dis_cosine_RNA = dis_cosine_RNA[:,0:5]      
    print("start load RNA-SEQ neighbor knn")  
    neighbors_cosine = np.loadtxt(neighbors_save_path)
    neighbors_cosine = neighbors_cosine[:,0:5]       
    neighbors_cosine = neighbors_cosine.astype(int)    
    print(f"Neighbors Size: {neighbors_cosine.shape}")

    dis_save_path = os.path.join(dis_simulate_dir, f"{args.dataset_name}_wsnn_knn_weight_atac.tsv")        
    dis_cosine_ATAC = np.loadtxt(dis_save_path)
    dis_cosine_ATAC = dis_cosine_RNA[:,0:5] 
      
    neighbors_ATAC_save_path = os.path.join(neighbors_simulate_dir, f"{args.dataset_name}_wsnn_knn_atac.tsv")        
    print("start load ATAC-SEQ neighbor knn")      
    neighbors_cosine_ATAC = np.loadtxt(neighbors_ATAC_save_path)
    neighbors_cosine_ATAC = neighbors_cosine_ATAC[:,0:5]       
    neighbors_cosine_ATAC = neighbors_cosine_ATAC.astype(int)    
    print(f"Neighbors Size: {neighbors_cosine_ATAC.shape}")
    ######################
    print("finish load ATAC-SEQ neighbor knn")   


    print("alpha is", alpha)
    print("lamda is", lams)
    
    args.alpha = {'recovery': alpha, 'contrastive':1-alpha}                
    args.lam = {'pos1': lams, 'pos2': 1, 'pos3': 1, "neg": 1} 

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(device)

    ################################ 1. train model for RNA + ATAC cosine             
    print("---------- Step 1.1: train MoCo_AE for RNA + ATAC ----------")   
    start_time_2 = time.time() 
    model = ContrastiveLearning_AE_new_ReviseLoss(Encoder_AE,
         in_features=data.shape[1],
         num_cluster=args.classnum,
         latent_features=args.latent_feature, 
         device=device,
         mlp=False,
         K=args.K,
         m=args.m,
         T=args.T,
         p=args.p,
         lam=args.lam,
         alpha=args.alpha)

    ##########
    print('---------- Step 1.2: Get cluster performance----------')                         
    model, Y, adata_embedding, latent = train_model_new_PosSimWeight(data, data, model, true_label,clus_method_set, neighbors_cosine, neighbors_cosine_ATAC, dis_cosine_RNA, dis_cosine_ATAC, args, device, pretrain = True)      

    kmeans_pred = adata_embedding.obs['kmeans']
    ari, nmi, ami, acc, fm = evaluate_cluster(true_label, kmeans_pred)                
    # 1.2save result 
    tb = pt.PrettyTable()
    tb.field_names = ['Method Name' + '_' + str(dname), 'alpha', 'lam', 'ARI', 'NMI','AMI', 'ACC', 'FM']
    tb.add_row(['Ours_RNA_ATAC_cosine', str(alpha), str(lams), round(ari, 4), round(nmi, 4), round(ami, 4), round(acc, 4), round(fm, 4)])
    print(tb)
           

import sys
import argparse                
                
if __name__ == "__main__":
    # 修改了数据集顺序和参数
    datasets = ["SNARE_cellline"]
    classnums = [4]
    
    lrs = [0.0001]
    K_set = [4608]
    
    num_genes_set = [2000] #default  
    p_set = [0.2] #default    
    t1_set = [0.1] # 根据default   

    
    lams_set = [0.1] #根据default  
    alphas_set = [0.4]        
    
    for ii in range(0,1):                                       
        dname = datasets[ii]
        classnum = classnums[ii]
        K = K_set[ii]
        learning_rate = lrs[ii]
        lams = lams_set[ii]
        alphas = alphas_set[ii]
        num_genes = num_genes_set[ii]
        T = t1_set[ii]
        p = p_set[ii]
        main(dname, classnum, learning_rate, K, T, alphas, num_genes, lams, p)

