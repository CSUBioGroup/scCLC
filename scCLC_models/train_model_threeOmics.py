import os
import argparse
import time

from scCLC_evalution import inference_evalidate, cal_knn_accuracy, save_core_cell_acc_new
 
import torch
from torch import nn
import numpy as np

#cop from pretrain_final_AERNA_knnEnhance

def train_epoch(model, 
                   data, 
                origin_data,
                   neighbors_cosine, neighbors_l2, neighbors1_cosine_ADT,
                dis_cosine, dis_l2,dis_cosine_ADT, 
                   batch_size, 
                   criterion, 
                   optimizer, 
                   device, 
                   c=1, 
                   flag='aug_nn'):
    loss_epoch = 0.0
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    count = 0

    # model.train()
    print(f"Model Traning Phase: {model.training}")
    for step, pre_index in enumerate(range(data.shape[0] // batch_size + 1)):
        indices_idx = np.arange(pre_index * batch_size, min(data.shape[0], (pre_index + 1) * batch_size))
        
        if len(indices_idx) < batch_size:
            continue
        
        count += 1
        
        batch_indices = indices[indices_idx]
        x = data[batch_indices]
        x = torch.FloatTensor(x).to(device)

        # Use Neighbors as positive instances
        if neighbors_cosine is not None:
            batch_dis_cosine = dis_cosine[batch_indices]
            batch_nei = neighbors_cosine[batch_indices]
            '''
            #batch_nei_idx = np.array([np.random.choice(nei, c) for nei in batch_nei])####k=6个邻居中随机选择一个邻居？是的
            batch_nei_idx = np.array([np.random.choice(nei, c, replace=False) for nei in batch_nei])####k=6个邻居中随机选择一个邻居？是的
            batch_nei_idx = batch_nei_idx.flatten()
            '''
            
            # 随机选择每行的c个索引
            batch_indices_random = np.array([np.random.choice(range(batch_nei.shape[1]), c, replace=False) for _ in range(batch_nei.shape[0])])

            # 获取每行对应的邻居和距离值
            batch_nei_idx = np.array([batch_nei[row_idx, indices] for row_idx, indices in enumerate(batch_indices_random)])
            batch_nei_idx = batch_nei_idx.flatten()

            batch_dis_cosine_weight = np.array([batch_dis_cosine[row_idx, indices] for row_idx, indices in enumerate(batch_indices_random)])
            batch_dis_cosine_weight = batch_dis_cosine_weight.flatten()
            
            x_nei_cosine = origin_data[batch_nei_idx]
            x_nei_cosine = torch.FloatTensor(x_nei_cosine).to(device)
            batch_dis_cosine_weight = torch.FloatTensor(batch_dis_cosine_weight).to(device)
                   
            assert int(x_nei_cosine.size(0) // x.size(0)) == int(c)

        if neighbors_l2 is not None:
            batch_nei = neighbors_l2[batch_indices]
            batch_dis_l2 = dis_l2[batch_indices]
            
            '''
            batch_nei_idx = np.array([np.random.choice(nei, c) for nei in batch_nei])####k=6个邻居中随机选择一个邻居？是的
            batch_nei_idx = batch_nei_idx.flatten()        
            '''
            # 随机选择每行的c个索引
            batch_indices_random = np.array([np.random.choice(range(batch_nei.shape[1]), c, replace=False) for _ in range(batch_nei.shape[0])])

            # 获取每行对应的邻居和距离值
            batch_nei_idx = np.array([batch_nei[row_idx, indices] for row_idx, indices in enumerate(batch_indices_random)])
            batch_nei_idx = batch_nei_idx.flatten()

            batch_dis_l2_weight = np.array([batch_dis_l2[row_idx, indices] for row_idx, indices in enumerate(batch_indices_random)])
            batch_dis_l2_weight = batch_dis_l2_weight.flatten()
            
            x_nei_l2 = origin_data[batch_nei_idx]
            x_nei_l2 = torch.FloatTensor(x_nei_l2).to(device)
            batch_dis_l2_weight = torch.FloatTensor(batch_dis_l2_weight).to(device)
      
            assert int(x_nei_l2.size(0) // x.size(0)) == int(c)
        
        if neighbors1_cosine_ADT is not None:
            batch_nei = neighbors1_cosine_ADT[batch_indices]
            batch_dis_ADT = dis_cosine_ADT[batch_indices]
            
            '''
            batch_nei_idx = np.array([np.random.choice(nei, c) for nei in batch_nei])####k=6个邻居中随机选择一个邻居？是的
            batch_nei_idx = batch_nei_idx.flatten()        
            '''
            # 随机选择每行的c个索引
            batch_indices_random = np.array([np.random.choice(range(batch_nei.shape[1]), c, replace=False) for _ in range(batch_nei.shape[0])])

            # 获取每行对应的邻居和距离值
            batch_nei_idx = np.array([batch_nei[row_idx, indices] for row_idx, indices in enumerate(batch_indices_random)])
            batch_nei_idx = batch_nei_idx.flatten()

            batch_dis_ADT_weight = np.array([batch_dis_ADT[row_idx, indices] for row_idx, indices in enumerate(batch_indices_random)])
            batch_dis_ADT_weight = batch_dis_ADT_weight.flatten()
            
            x_nei_ADT = origin_data[batch_nei_idx]
            x_nei_ADT = torch.FloatTensor(x_nei_ADT).to(device)
            batch_dis_ADT_weight = torch.FloatTensor(batch_dis_ADT_weight).to(device)
      
            assert int(x_nei_l2.size(0) // x.size(0)) == int(c)
        
        
        if flag == 'aug_nn' and neighbors_cosine is not None and neighbors_l2 is not None and neighbors1_cosine_ADT is not None:   # Using its augmentation counterpart and neighbor to form positive pairs
            #print("############ x3 is not none, using forward_aug_nn_2 !!!!")
            loss = model(x, x_nei_cosine, x_nei_l2, x_nei_ADT, batch_dis_cosine_weight, batch_dis_l2_weight, batch_dis_ADT_weight,flag=flag)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        if step % 50 == 0:
            print(f"Step [{step}/{data.shape[0]}]\t loss_instance: {loss.item()}")
        
        loss_epoch += loss.item()
    
    loss_epoch = loss_epoch / count

    return loss_epoch


def train_model_threeOmics(data, origin_data, model, ClusterIndex_all_cases, true_label, epoch_check_set, table_save_path, clus_method_set, plot_set, neighbors1_cosine, neighbors1_cosine_ATAC, neighbors1_cosine_ADT, dis_cosine, dis_l2,dis_cosine_ADT, args, device, plot_state = False, pretrain=False):
       
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=args.learning_rate, 
                                 weight_decay=0.0)

    model_path = os.path.join(args.model_path, f"seed_{args.seed}")
    if args.reload:
        model_fp = os.path.join(model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1

    # train
    t0 = time.time()
    
    for epoch in range(1, args.epochs+1):
        #model.train()
        
        print(f"Epoch [{epoch}/{args.epochs}]")
        
        loss_epoch = train_epoch(model, 
                                    data,
                                 origin_data,
                                    neighbors1_cosine, neighbors1_cosine_ATAC,neighbors1_cosine_ADT,
                                 dis_cosine, dis_l2,dis_cosine_ADT, 
                                    args.batch_size, 
                                    criterion, 
                                    optimizer, 
                                    device,
                                    c=args.c,
                                    flag=args.flag)

        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch}")
        print('-' * 60)
        
        #if epoch % 10 == 0:  
         #   print('0808 add this eval v536')
          #  model.eval()
            
        #if epoch % 20 == 0:   
         #   args.lam["neg"] = args.lam["neg"] - 0.1
          #  print(args.lam["neg"])
            #print(f"Epoch [{epoch}/{args.epochs}] args.lam[neg]: {args.lam["neg"]}")
            
        if epoch in epoch_check_set and not pretrain:
            Y, adata_embedding, latent = inference_evalidate.get_clus_label(data.copy(), true_label, clus_method_set, args.cell_type_name,  args, model, device)
            mean_accuracy_cells_true = cal_knn_accuracy(latent, true_label, n_neighbors=args.n, n_pools=1, n_samples_per_pool=100)
           
            mean_accuracy_cells_kmean = cal_knn_accuracy(latent, adata_embedding.obs['kmeans'].values, n_neighbors=args.n, n_pools=1, n_samples_per_pool=100)  
            
            record  = 'Alpha'+ str(args.aa) + '_Lam' + str(args.bb) + '_epoch' + str(epoch)
            
            ClusterIndex_all_cases = inference_evalidate.evaluate_clus(true_label, adata_embedding, ClusterIndex_all_cases, table_save_path, clus_method_set, record, args, mean_accuracy_cells_true, mean_accuracy_cells_leiden, mean_accuracy_cells_louvain, mean_accuracy_cells_kmean)
            
            save_core_cell_acc_new(adata_embedding, adata_embedding, table_save_path, epoch, pseudo_label = 'kmeans', k=10, percent = 0.2)

                        
            if plot_state:
                inference_evalidate.plot_visualize(adata_embedding, 
                                 Y, 
                                 record, 
                                 seed = 42, 
                                 colors=plot_set["colors"],
                                 titles=plot_set["titles"],
                                 fig_save_path=plot_set["fig_save_path"])   

    print("finish train_model_threeOmics, start inference_evalidate")
    t1 = time.time()
    
    Y, adata_embedding, latent = inference_evalidate.get_clus_label(data.copy(), true_label, clus_method_set, args.cell_type_name,  args, model, device)
    
    print('finish inference_evalidate, the time is')
    t2 = time.time()
    print(t2 - t1)
    return model, ClusterIndex_all_cases, Y, adata_embedding, latent
        