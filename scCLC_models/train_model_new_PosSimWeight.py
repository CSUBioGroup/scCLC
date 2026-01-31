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
                   neighbors_cosine, neighbors_l2, 
                dis_cosine, dis_l2,
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
        
        if flag == 'aug_nn' and neighbors_cosine is not None and neighbors_l2 is not None:   # Using its augmentation counterpart and neighbor to form positive pairs
            #print("############ x3 is not none, using forward_aug_nn_2 !!!!")
            loss = model(x, x_nei_cosine, x_nei_l2, batch_dis_cosine_weight, batch_dis_l2_weight, flag=flag)
        if flag == 'aug_nn' and neighbors_cosine is not None and neighbors_l2 is None:   # Using its augmentation counterpart and neighbor to form positive pairs
            #print("############ x3 is none, using forward_aug_nn_1 !!!!")
            loss = model(x, x_nei_cosine, None, batch_dis_cosine_weight, None, flag=flag)
        if flag == 'aug_nn' and neighbors_cosine is None and neighbors_l2 is None:   # Using its augmentation counterpart and neighbor to form positive pairs
            #print("############ x2 and x3 are not none, using forward_aug_nn !!!!")
            loss = model(x, None, None, None, None, flag=flag)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        if step % 50 == 0:
            print(f"Step [{step}/{data.shape[0]}]\t loss_instance: {loss.item()}")
        
        loss_epoch += loss.item()
    
    loss_epoch = loss_epoch / count

    return loss_epoch


def train_model_new_PosSimWeight(data, origin_data, model, true_label, clus_method_set, neighbors1_cosine, neighbors1_cosine_ATAC, dis_cosine, dis_cosine_ATAC, args, device, pretrain=False):
       
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
                                    neighbors1_cosine, neighbors1_cosine_ATAC,
                                 dis_cosine, dis_cosine_ATAC,
                                    args.batch_size, 
                                    criterion, 
                                    optimizer, 
                                    device,
                                    c=args.c,
                                    flag=args.flag)

        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch}")
        print('-' * 60)


    Y, adata_embedding, latent = inference_evalidate.get_clus_label(data.copy(), true_label, clus_method_set, args.cell_type_name,  args, model, device)
    return model, Y, adata_embedding, latent
        