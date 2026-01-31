import os
import argparse
import time
import prettytable as pt

# copy from cluster_final_knnEnhance_simple
    
from preprocess import *
#from moco import Encoder, MoCo
#from utils import yaml_config_hook, set_seed
from .evaluation_funcs import evaluate_cluster, run_leiden_scanpy, run_louvain_scanpy, run_kmeans

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
 
def inference(x, y, args, model, device):
    model.eval()
    # x = data, y = true_label

    val_datasets = CellDataset(x, y)
    in_features = val_datasets.data.size(1)

    print(f"Validation Dataset size: {len(val_datasets)}")
    print(f"The in_features is: {in_features}")

    val_loader = DataLoader(val_datasets,
                            batch_size=256,
                            shuffle=False,
                            num_workers=args.workers,
                            drop_last=False)

    labels_vector = []
    latent_vector = []
    final_out_vector = []
    
    for step, (x, y) in enumerate(val_loader):
        x = x.to(device)

        with torch.no_grad():
            latent = model.get_embedding(x)            
            latent = F.normalize(latent, dim=1)
            
        latent = latent.detach()
        latent_vector.extend(latent.cpu().detach().numpy())      
        labels_vector.extend(y.numpy())

        if step % 50 == 0:
            print(f"Step [{step}/{len(val_loader)}]\t Computing features...")

    labels_vector = np.array(labels_vector)
    latent_vector = np.array(latent_vector)

    return labels_vector, latent_vector, val_loader

def get_clus_label(data_ATAC, true_label_ATAC, clus_method_set, cell_type_name,  args, model, device):
    
    Y, latent, val_loader= inference(data_ATAC, true_label_ATAC, args, model, device)
    
    adata_embedding = sc.AnnData(latent, dtype=np.float32)
    sc.pp.neighbors(adata_embedding, n_neighbors=20, n_pcs=0, use_rep='X')
    adata_embedding.obs['true_label'] = Y
    adata_embedding.obs['true_label'] = adata_embedding.obs['true_label'].astype("category")
    
    if clus_method_set['leiden']:
        print("### Performming Leiden clustering method on latent vector ###")
        adata_embedding, leiden_pred = run_leiden_scanpy(adata=adata_embedding, 
                                                  resolution=args.resolution)
        adata_embedding.obs['leiden'] = leiden_pred
        adata_embedding.obs['leiden'] = adata_embedding.obs['leiden'].astype("category")
        
    if clus_method_set['louvain']:
        print("### Performming Louvain clustering method on latent vector ###")
        adata_embedding, louvain_pred = run_louvain_scanpy(adata=adata_embedding, 
                                                  resolution=args.resolution)
        adata_embedding.obs['louvain'] = louvain_pred
        adata_embedding.obs['louvain'] = adata_embedding.obs['louvain'].astype("category")
        
    if clus_method_set['kmeans']:
        print("### Performming KMeans clustering method on latent vector ###")
        kmeans_pred = run_kmeans(latent, args.classnum, random_state=args.seed)
        adata_embedding.obs['kmeans'] = kmeans_pred
        adata_embedding.obs['kmeans'] = adata_embedding.obs['kmeans'].astype("category")
    
    if cell_type_name is not None:
        adata_embedding.obs['annotation'] = cell_type_name
        adata_embedding.obs['annotation'] = np.array(list(map(str, adata_embedding.obs['annotation'].values)))

    return Y, adata_embedding, latent

def evaluate_clus(true_label, adata_embedding, ClusterIndex_all_cases, table_save_path, clus_method_set, epoch, args, mean_accuracy_cells_true, mean_accuracy_cells_leiden, mean_accuracy_cells_louvain, mean_accuracy_cells_kmean):
    # Evaluation 
    tb = pt.PrettyTable()
    tb.field_names = ["Method Name", "epoch", "ARI", "NMI", "knn_mean_acc","AMI", "ACC", "FM"]
    
    
    if clus_method_set['kmeans']:
        kmeans_pred = adata_embedding.obs['kmeans']
        ari, nmi, ami, acc, fm = evaluate_cluster(true_label, kmeans_pred)
        # 创建一个包含评价指标的字典
        eval_metrics = {
            "Method":"kmeans",
            "Epoch":epoch,
            "ARI": ari,
            "NMI": nmi,
            "knn_mean_acc": mean_accuracy_cells_kmean,
            "AMI": ami,
            "ACC": acc,
            "FM": fm
        }
        # 将评价指标添加到数据框中的新行
        eval_row = pd.DataFrame([eval_metrics])
        ClusterIndex_all_cases = pd.concat([ClusterIndex_all_cases, eval_row], ignore_index=True)
        
        # Print out related results
        print(f"### Epoch {epoch} Results: ###")
        tb.add_row(["KMeans", epoch, round(ari, 4), round(nmi, 4), round(mean_accuracy_cells_kmean, 4), round(ami, 4), round(acc, 4), round(fm, 4)])
        print(tb)
        
        
    if clus_method_set['leiden']:
        leiden_pred = adata_embedding.obs['leiden']
        ari, nmi, ami, acc, f = evaluate_cluster(true_label, leiden_pred)
        # 创建一个包含评价指标的字典
        eval_metrics = {
            "Method":"leiden",
            "Epoch":epoch,
            "ARI": ari,
            "NMI": nmi,
            "knn_mean_acc": mean_accuracy_cells_leiden,
            "AMI": ami,
            "ACC": acc,
            "FM": fm
        }
        # 将评价指标添加到数据框中的新行
        eval_row = pd.DataFrame([eval_metrics])
        ClusterIndex_all_cases = pd.concat([ClusterIndex_all_cases, eval_row], ignore_index=True)
        
        # Print out related results
        print(f"### Epoch {epoch} Results: ###")
        tb.add_row(["Leiden", epoch, round(ari, 4), round(nmi, 4), round(mean_accuracy_cells_leiden, 4), round(ami, 4), round(acc, 4), round(fm, 4)])
        print(tb)
        
        
    if clus_method_set['louvain']:
        louvain_pred = adata_embedding.obs['louvain']
        ari, nmi, ami, acc, f = evaluate_cluster(true_label, louvain_pred)
        # 创建一个包含评价指标的字典
        eval_metrics = {
            "Method":"louvain",
            "Epoch":epoch,
            "ARI": ari,
            "NMI": nmi,
            "knn_mean_acc": mean_accuracy_cells_louvain,
            "AMI": ami,
            "ACC": acc,
            "FM": fm
        }
        # 将评价指标添加到数据框中的新行
        eval_row = pd.DataFrame([eval_metrics])
        ClusterIndex_all_cases = pd.concat([ClusterIndex_all_cases, eval_row], ignore_index=True)
        
        # Print out related results
        print(f"### Epoch {epoch} Results: ###")
        tb.add_row(["Louvain", epoch, round(ari, 4), round(nmi, 4), round(mean_accuracy_cells_louvain, 4), round(ami, 4), round(acc, 4), round(fm, 4)])
        print(tb)
        
    # save the true knn acc   
    # 创建一个包含评价指标的字典
    eval_metrics = {
        "Method":"true_label",
        "Epoch":epoch,
        "ARI": 0,
        "NMI": 0,
        "knn_mean_acc": mean_accuracy_cells_true,
        "AMI": 0,
        "ACC": 0,
        "FM": 0
    }
    # 将评价指标添加到数据框中的新行
    eval_row = pd.DataFrame([eval_metrics])
    ClusterIndex_all_cases = pd.concat([ClusterIndex_all_cases, eval_row], ignore_index=True)

    # Print out related results
    print(f"### Epoch {epoch} Results: ###")
    tb.add_row(["true_label", epoch, round(0, 4), round(0, 4), round(mean_accuracy_cells_true, 4), round(0, 4), round(0, 4), round(0, 4)])
    print(tb)
        
    file_name = table_save_path
    f = open(file_name, "a")
    current_time = time.strftime("%Y_%m_%d %H_%M_%S", time.localtime(time.time()))

    f.write(f"Epoch {args.epochs} -----------> {current_time}\n")
    f.write(str(tb) + '\n')
    f.close()
        
    return ClusterIndex_all_cases
    
def plot_visualize(adata_embedding, 
         Y, 
         epoch, 
         seed = 42, 
         colors=['leiden', 'label','kmeans', 'true_label'],
         titles=['Leiden Predictions','True Label','Leiden Predictions', 'True Label'],
         fig_save_path="pictures"):
    sc.settings.verbosity = 0
    sc.settings.set_figure_params(dpi=160)

    fig_rows = len(colors)//2
    _, axes = plt.subplots(fig_rows, 2, figsize=(18, 8*fig_rows))
    plt.subplots_adjust(wspace=0.3)
    

    for row in axes:
        for ax in row:
            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_visible(True)

    if 'X_umap' not in adata_embedding.obsm_keys():
        sc.tl.umap(adata_embedding)

        
    for fig_num, (color, title, ax) in enumerate(zip(colors, titles, axes.flatten())):        
        sc.pl.umap(adata_embedding, 
                   color=color, 
                   title=title, 
                   wspace=0.36, 
                   show=False,
                   ax=ax)

    for row_index in range(axes.shape[0]):
        # 设置第一列的图例
        axes[row_index, 0].legend(frameon=False, 
                                   loc='center', 
                                   bbox_to_anchor=(0.50, -0.12),
                                   ncol=len(np.unique(Y)) // 2, 
                                   fontsize=8,
                                   markerscale=0.9)

    # 如果第二列需要根据条件设置图例
    if "annotation" in colors:
        for row_index in range(axes.shape[0]):
            # 设置第二列的图例
            axes[row_index, 1].legend(frameon=False, 
                                       loc='center', 
                                       bbox_to_anchor=(0.50, -0.12),
                                       ncol=3, 
                                       fontsize=8,
                                       markerscale=0.9)
    else:
        for row_index in range(axes.shape[0]):
            # 设置第二列的图例
            axes[row_index, 1].legend(frameon=False, 
                                       loc='center', 
                                       bbox_to_anchor=(0.50, -0.12),
                                       ncol=len(np.unique(Y)) // 2, 
                                       fontsize=8,
                                       markerscale=0.9)

    plt.grid(False)

    path_2_save = os.path.join(fig_save_path, f"Epoch_{epoch}_{seed}.png")

    plt.savefig(path_2_save, dpi=160.0)

    
    
def evaluate_origin_knn(true_label, ClusterIndex_all_cases, table_save_path, clus_method_set, epoch, args, mean_accuracy_cells_true):
    # Evaluation 
    tb = pt.PrettyTable()
    tb.field_names = ["Method Name", "epoch", "ARI", "NMI", "knn_mean_acc","AMI", "ACC", "FM"]

        
    # save the true knn acc   
    # 创建一个包含评价指标的字典
    eval_metrics = {
        "Method":"origin_knn",
        "Epoch":0,
        "ARI": 0,
        "NMI": 0,
        "knn_mean_acc": mean_accuracy_cells_true,
        "AMI": 0,
        "ACC": 0,
        "FM": 0
    }
    # 将评价指标添加到数据框中的新行
    eval_row = pd.DataFrame([eval_metrics])
    ClusterIndex_all_cases = pd.concat([ClusterIndex_all_cases, eval_row], ignore_index=True)

    # Print out related results
    print(f"### Epoch {epoch} Results: ###")
    tb.add_row(["true_label", round(0, 4), round(0, 4), round(0, 4), round(mean_accuracy_cells_true, 4), round(0, 4), round(0, 4), round(0, 4)])
    print(tb)
        
    file_name = table_save_path
    f = open(file_name, "a")
    current_time = time.strftime("%Y_%m_%d %H_%M_%S", time.localtime(time.time()))

    f.write(f"Epoch {args.epochs} -----------> {current_time}\n")
    f.write(str(tb) + '\n')
    f.close()
        
    return ClusterIndex_all_cases
