########### get core cells and cal the cluster acc of core cells
from preprocess.get_core_cell import get_core_cell_new_sim
from .evaluation_funcs import evaluate_cluster
import prettytable as pt
import os
import numpy as np
import scanpy as sc
import random
import math
import hnswlib
import time

def save_core_cell_acc(adata, adata_embedding, table_save_path, pseudo_label = 'kmeans', k=30, percent = 0.3):
    print("start save_core_cell_acc")
    _, adata_embedding = get_core_cell(adata, 
                                adata_embedding, 
                                pseudo_label=pseudo_label,
                                k=k, 
                                percent=percent)
    #train_adata = adata[adata.obs.leiden_density_status == 'low', :].copy()
    #test_adata = adata[adata.obs.leiden_density_status == 'high', :].copy()

    train_adata = adata_embedding[adata_embedding.obs.leiden_density_status == 'high', :].copy()
    test_adata = adata_embedding[adata_embedding.obs.leiden_density_status == 'low', :].copy()

    pseudo_labels = np.array(list(map(int, train_adata.obs['leiden'].values)))
    print(f"extracted_nmi: {normalized_mutual_info_score(train_adata.obs['true_label'].values, pseudo_labels):.4f}")
    print(f"extracted_ari : {adjusted_rand_score(train_adata.obs['true_label'].values, pseudo_labels):.4f}")
    ari, nmi, ami, acc, fm = evaluate_cluster(train_adata.obs['true_label'].values, pseudo_labels)

     # Evaluation 
    tb = pt.PrettyTable()
    tb.field_names = ["Method Name", "epoch", "ARI", "NMI", "AMI", "ACC", "FM"]
    print(f"### Epoch {args.epochs} Results: ###")
    tb.add_row(["core_cells", args.epochs, round(ari, 4), round(nmi, 4), round(ami, 4), round(acc, 4), round(fm, 4)])
    print(tb)

    file_name = table_save_path
    f = open(file_name, "a")
    current_time = time.strftime("%Y_%m_%d %H_%M_%S", time.localtime(time.time()))

    f.write(f"Epoch {args.epochs} -----------> {current_time}\n")
    f.write(str(tb) + '\n')
    f.close()
    
    
def save_core_cell_acc_new_old(adata, adata_embedding, table_save_path, epoch, pseudo_label = 'kmeans', k=30, percent = 0.3):
    print("start save_core_cell_acc")
    _, adata_embedding = get_core_cell(adata, 
                                adata_embedding, 
                                pseudo_label=pseudo_label,
                                k=k, 
                                percent=percent)
    #train_adata = adata[adata.obs.leiden_density_status == 'low', :].copy()
    #test_adata = adata[adata.obs.leiden_density_status == 'high', :].copy()

    train_adata = adata_embedding[adata_embedding.obs[f"{pseudo_label}_density_status"] == 'high', :].copy()
    test_adata = adata_embedding[adata_embedding.obs[f"{pseudo_label}_density_status"] == 'low', :].copy()

     # Evaluation 
    tb = pt.PrettyTable()
    tb.field_names = ["Method Name", "epoch", "ARI", "NMI", "AMI", "ACC", "FM"]
    print(f"### Epoch {epoch} Results: ###")
    pseudo_labels = np.array(list(map(int, train_adata.obs[pseudo_label].values)))
    ari, nmi, ami, acc, fm = evaluate_cluster(train_adata.obs['true_label'].values, pseudo_labels)
    tb.add_row([f"core_cells_acc_{pseudo_label}", epoch, round(ari, 4), round(nmi, 4), round(ami, 4), round(acc, 4), round(fm, 4)])
    print(tb)

    
    file_name = table_save_path
    f = open(file_name, "a")

    f.write(str(tb) + '\n')
    f.close()
    
def save_core_cell_acc_new(adata, adata_embedding, table_save_path, epoch, pseudo_label = 'kmeans', k=30, percent = 0.3):
    print("start save_core_cell_acc")
    _, adata_embedding = get_core_cell_new_sim(adata, 
                                adata_embedding, 
                                pseudo_label=pseudo_label,
                                k=k, 
                                percent=percent)
    #train_adata = adata[adata.obs.leiden_density_status == 'low', :].copy()
    #test_adata = adata[adata.obs.leiden_density_status == 'high', :].copy()

    train_adata = adata_embedding[adata_embedding.obs[f"{pseudo_label}_density_status"] == 'high', :].copy()
    test_adata = adata_embedding[adata_embedding.obs[f"{pseudo_label}_density_status"] == 'low', :].copy()

     # Evaluation 
    tb = pt.PrettyTable()
    tb.field_names = ["Method Name", "epoch", "ARI", "NMI", "AMI", "ACC", "FM"]
    print(f"### Epoch {epoch} Results: ###")
    pseudo_labels = np.array(list(map(int, train_adata.obs[pseudo_label].values)))
    ari, nmi, ami, acc, fm = evaluate_cluster(train_adata.obs['true_label'].values, pseudo_labels)
    tb.add_row([f"core_cells_acc_{pseudo_label}", epoch, round(ari, 4), round(nmi, 4), round(ami, 4), round(acc, 4), round(fm, 4)])
    print(tb)

    
    file_name = table_save_path
    f = open(file_name, "a")

    f.write(str(tb) + '\n')
    f.close()
    
    
def save_core_cell_acc_new_error(adata, adata_embedding, table_save_path, epoch, pseudo_label = 'kmeans', k=30, percent = 0.3):
    print("start save_core_cell_acc")
    _, adata_embedding = get_core_cell(adata, 
                                adata_embedding, 
                                pseudo_label=pseudo_label,
                                k=k, 
                                percent=percent)
    #train_adata = adata[adata.obs.leiden_density_status == 'low', :].copy()
    #test_adata = adata[adata.obs.leiden_density_status == 'high', :].copy()

    train_adata = adata_embedding[adata_embedding.obs.leiden_density_status == 'high', :].copy()
    test_adata = adata_embedding[adata_embedding.obs.leiden_density_status == 'low', :].copy()

     # Evaluation 
    tb = pt.PrettyTable()
    tb.field_names = ["Method Name", "epoch", "ARI", "NMI", "AMI", "ACC", "FM"]
    print(f"### Epoch {epoch} Results: ###")
    pseudo_labels = np.array(list(map(int, train_adata.obs['kmeans'].values)))
    ari, nmi, ami, acc, fm = evaluate_cluster(train_adata.obs['true_label'].values, pseudo_labels)
    tb.add_row(["core_cells_acc_kmeans", epoch, round(ari, 4), round(nmi, 4), round(ami, 4), round(acc, 4), round(fm, 4)])
    print(tb)
    
    pseudo_labels = np.array(list(map(int, train_adata.obs['leiden'].values)))
    ari, nmi, ami, acc, fm = evaluate_cluster(train_adata.obs['true_label'].values, pseudo_labels)
    tb.add_row(["core_cells_acc_leiden", epoch, round(ari, 4), round(nmi, 4), round(ami, 4), round(acc, 4), round(fm, 4)])
    print(tb)
    
    pseudo_labels = np.array(list(map(int, train_adata.obs['louvain'].values)))
    ari, nmi, ami, acc, fm = evaluate_cluster(train_adata.obs['true_label'].values, pseudo_labels)
    tb.add_row(["core_cells_acc_louvain", epoch, round(ari, 4), round(nmi, 4), round(ami, 4), round(acc, 4), round(fm, 4)])
    print(tb)
  
    
    file_name = table_save_path
    f = open(file_name, "a")
    current_time = time.strftime("%Y_%m_%d %H_%M_%S", time.localtime(time.time()))

    f.write(f"Epoch {epoch} -----------> {current_time}\n")
    f.write(str(tb) + '\n')
    f.close()