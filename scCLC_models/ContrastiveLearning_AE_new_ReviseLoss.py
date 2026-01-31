import torch
from torch import nn
import torch.nn.functional as F
 
    
#moco_AE_knnEnhance_ReLoss

class ContrastiveLearning_AE_new_ReviseLoss(nn.Module):
    def __init__(self, 
                 encoder,
                 in_features, 
                 num_cluster,
                 latent_features=[1024, 512, 128],
                 device="cpu",
                 mlp=True,
                 K=65536,
                 m=0.999,
                 T=0.9,
                 p=0.0,
                 lam=0.1,
                 alpha=0.1,
                pos_num = 1):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        self.lam = lam
        self.alpha = alpha
        self.rep_dim = latent_features[-1]
        self.pos_num = pos_num
        
        
        #self.rep_dim = num_cluster
        self.device = device
        
        self.encoder_q = encoder(in_features=in_features,
                                 num_cluster=num_cluster, 
                                 latent_features=latent_features,
                                 device=device,
                                 p=p)
        self.encoder_k = encoder(in_features=in_features, 
                                 num_cluster=num_cluster,
                                 latent_features=latent_features,
                                 device=device,
                                 p=p)
        
        # Projection Head
        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]  #权重矩阵列数，即输出特征的维数，因为权重矩阵和网络形状是相反的
            print(f"dim_mlp: {dim_mlp}")
            
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), 
                nn.BatchNorm1d(dim_mlp),
                nn.ReLU(), 
                nn.Linear(dim_mlp, dim_mlp)
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), 
                nn.BatchNorm1d(dim_mlp),
                nn.ReLU(), 
                nn.Linear(dim_mlp, dim_mlp)
            )

        for param_k, param_q in zip(self.encoder_k.parameters(), self.encoder_q.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        self.register_buffer("queue", 
                             F.normalize(torch.randn(self.K, self.rep_dim, requires_grad=False), dim=1))
        self.ptr = 0
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_k, param_q in zip(self.encoder_k.parameters(), self.encoder_q.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)
            param_k.requires_grad = False
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.size(0)
        
        self.queue[self.ptr: self.ptr + batch_size, :] = keys.detach()  #将每个batch的k2的输出存到队列，先进先出原则，最多存入K个，当做负样本
        self.ptr = (self.ptr + batch_size) % self.K
        self.queue.requires_grad = False      

    def forward_aug_nn(self, x1):
        x_recon = self.encoder_q(x1)
        q = self.encoder_q.get_embedding(x1)
        q = F.normalize(q, dim=1)
        qc = q.unsqueeze(1)

        qc = qc.reshape(-1, q.size(1))  #将qc重塑，列数等于 q.size(1)，行数推理而得，得到（c*q.size(0), q.size(1)）张量

        with torch.no_grad():
            self._momentum_update_key_encoder()

            k1 = self.encoder_k.get_embedding(x1)
            k1 = F.normalize(k1, dim=1)
            k1_recon = self.encoder_k(x1)

        pos_sim1 = self.lam["pos1"] * torch.einsum("ic, ic -> i", [q, k1]).unsqueeze(-1)#4011
        neg_sim = torch.einsum("ic, jc -> ij", [q, self.queue.clone().detach()]) # batch_size*K        
        loss_cl = -(torch.logsumexp(pos_sim1 / self.T, dim=1) - torch.logsumexp(neg_sim / self.T, dim=1)).mean()
        
        #q_recon_loss = F.mse_loss(x_recon, x1, reduction='sum')
        q_recon_loss = F.mse_loss(x_recon, x1, reduction='mean') # 0826
        
        k_recon_loss = F.mse_loss(k1_recon, x1, reduction='sum')
        
        recon_loss = self.alpha * q_recon_loss        
                        
        #loss = 0.001 * recon_loss + ((1 -  self.alpha)) * loss_cl
        loss = recon_loss + self.alpha["contrastive"] * loss_cl #0826
        
        
        return loss
    
                
    def forward_aug_nn_1(self, x1, x2, weight2):
        x_recon = self.encoder_q(x1)
        q = self.encoder_q.get_embedding(x1)
        q = F.normalize(q, dim=1)

        c = x2.size(0) // x1.size(0)
        qc = q.unsqueeze(1)
        for _ in range(1, c):
            qc = torch.cat([qc, q.unsqueeze(1)], dim=1)#将q复制c次，按列拼接，得到（q.size(0), q.size(1)*c）张量
        qc = qc.reshape(-1, q.size(1))  #将qc重塑，列数等于 q.size(1)，行数推理而得，得到（c*q.size(0), q.size(1)）张量

        assert qc.size(0) == x2.size(0)

        with torch.no_grad():
            self._momentum_update_key_encoder()

            k1_recon = self.encoder_k(x1)
            k2_recon = self.encoder_k(x2)                 

            k1 = self.encoder_k.get_embedding(x1)
            k2 = self.encoder_k.get_embedding(x2)
            
            k1 = F.normalize(k1, dim=1)
            k2 = F.normalize(k2, dim=1)
            
        pos_sim1 = self.lam["pos1"] * torch.einsum("ic, ic -> i", [q, k1]).unsqueeze(-1)#4011
        
        pos_sim2 = self.lam["pos2"] * torch.einsum("ic, ic -> i", [qc, k2]).unsqueeze(-1)
        pos_sim2 = pos_sim2 * weight2.view(-1, 1) 
        pos_sim2 = pos_sim2.reshape(-1, c) # batch_size*c

        assert pos_sim2.size(0) == pos_sim1.size(0)
        
        pos_sim = torch.cat([pos_sim1, pos_sim2], dim=1) # 列数增加了, batch_size, c+1        
        neg_sim = self.lam["neg"] * torch.einsum("ic, jc -> ij", [q, self.queue.clone().detach()]) # batch_size*K
        
        loss_cl = -(torch.logsumexp(pos_sim / self.T, dim=1) - torch.logsumexp(neg_sim / self.T, dim=1)).mean()
        
        q_recon_loss = F.mse_loss(x_recon, x1, reduction='sum')
        #q_recon_loss = F.mse_loss(x_recon, x1, reduction='mean') # 0826
        
        k1_recon_loss = F.mse_loss(k1_recon, x1, reduction='sum')
        k2_recon_loss = F.mse_loss(k2_recon, x2, reduction='sum')
                
        recon_loss = self.alpha['recovery'] * q_recon_loss                                
        loss = 0.001 * recon_loss + self.alpha['contrastive'] * loss_cl
        #loss = recon_loss + self.alpha["contrastive"] * loss_cl #0826
        
                        
        self._dequeue_and_enqueue(k2)
        
        return loss
    

    def forward_aug_nn_2(self, x1, x2, x3, weight2, weight3):
        x_recon = self.encoder_q(x1)
        q = self.encoder_q.get_embedding(x1)
        q = F.normalize(q, dim=1)

        c = x2.size(0) // x1.size(0)

        qc = q.unsqueeze(1)
        for _ in range(1, c):
            qc = torch.cat([qc, q.unsqueeze(1)], dim=1)#将q复制c次，按列拼接，得到（q.size(0), q.size(1)*c）张量
        qc = qc.reshape(-1, q.size(1))  #将qc重塑，列数等于 q.size(1)，行数推理而得，得到（c*q.size(0), q.size(1)）张量

        assert qc.size(0) == x2.size(0)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            
            k1_recon = self.encoder_k(x1)
            k2_recon = self.encoder_k(x2) 
            k3_recon = self.encoder_k(x3)                 
            
            k1 = self.encoder_k.get_embedding(x1)
            k2 = self.encoder_k.get_embedding(x2)
            k3 = self.encoder_k.get_embedding(x3)
            
            k1 = F.normalize(k1, dim=1)
            k2 = F.normalize(k2, dim=1)
            k3 = F.normalize(k3, dim=1)
            
        pos_sim1 = self.lam["pos1"] * torch.einsum("ic, ic -> i", [q, k1]).unsqueeze(-1)#4011

        pos_sim2 = self.lam["pos2"] * torch.einsum("ic, ic -> i", [qc, k2]).unsqueeze(-1)
        pos_sim2 = pos_sim2 * weight2.view(-1, 1) 
        pos_sim2 = pos_sim2.reshape(-1, c) # batch_size*c

        pos_sim3 = self.lam["pos3"] * torch.einsum("ic, ic -> i", [qc, k3]).unsqueeze(-1)#4011 
        pos_sim3 = pos_sim3 * weight3.view(-1, 1) 
        pos_sim3 = pos_sim3.reshape(-1, c) # batch_size*c
        

        assert pos_sim2.size(0) == pos_sim1.size(0)
        assert pos_sim3.size(0) == pos_sim1.size(0)
        

        pos_sim = torch.cat([pos_sim1, pos_sim2, pos_sim3], dim=1) # 列数增加了, batch_size, c+1
        neg_sim = self.lam["neg"] * torch.einsum("ic, jc -> ij", [q, self.queue.clone().detach()]) # batch_size*K
        #print('size of logsumexp')
        #print((torch.logsumexp(pos_sim / self.T, dim=1)).size)
        #print((torch.logsumexp(neg_sim / self.T, dim=1)).size) # batch_size*1
        
        ###test
        # Calculate InfoNCE loss
        #loss_cl = -torch.log(torch.sum(torch.exp(pos_sim/ self.T)) / (torch.sum(torch.exp(pos_sim/ self.T)) + torch.sum(torch.exp(neg_sim/ self.T))))
        
        loss_cl = -(torch.logsumexp(pos_sim / self.T, dim=1) - torch.logsumexp(neg_sim / self.T, dim=1)).mean()
        #penalty = self.alpha * (torch.mean(torch.abs(latent)))
        q_recon_loss = F.mse_loss(x_recon, x1, reduction='sum')
        #q_recon_loss = F.mse_loss(x_recon, x1, reduction='mean') # 0826
        
        k1_recon_loss = F.mse_loss(k1_recon, x1, reduction='sum')
        k2_recon_loss = F.mse_loss(k2_recon, x2, reduction='sum')
        k3_recon_loss = F.mse_loss(k3_recon, x3, reduction='sum')
        
        
        recon_loss = self.alpha["recovery"] * q_recon_loss        
                        
        loss = 0.001 * recon_loss + self.alpha["contrastive"] * loss_cl
        #loss = recon_loss + self.alpha["contrastive"] * loss_cl #0826
        
        
        self._dequeue_and_enqueue(k2) # v1101
        self._dequeue_and_enqueue(k3)
        
        return loss
    
    def forward(self, x1, x2, x3,  weight2, weight3, flag="aug_nn"):
        if flag == 'aug_nn' and x2 is not None and x3 is not None:
            #print("############ x3 is not none, using forward_aug_nn_2 !!!!")
            return self.forward_aug_nn_2(x1, x2, x3, weight2, weight3)
        if flag == 'aug_nn' and x2 is not None and x3 is None:
            #print("############ x3 is none, using forward_aug_nn_1 !!!!")
            return self.forward_aug_nn_1(x1, x2, weight2)
        if flag == 'aug_nn' and x2 is None and x3 is None: 
            #print("############ x2 and x3 are not none, using forward_aug_nn !!!!")
            return self.forward_aug_nn(x1)
        q = self.encoder_q(x1)
        q = F.normalize(q, dim=1)
        
        with torch.no_grad():
            self._momentum_update_key_encoder()

            k = self.encoder_k(x2)
            k = F.normalize(k, dim=1)

        pos_sim = torch.einsum("ic, ic -> i", [q, k]).unsqueeze(-1)
        neg_sim = torch.einsum("ic, jc -> ij", [q, self.queue.clone().detach()])
        
        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.T
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(self.device)

        self._dequeue_and_enqueue(k)

        return logits, labels
     
    def get_embedding(self, x):
        # out = self.encoder_k.get_embedding(x)
        out = self.encoder_q.get_embedding(x)        
        return out
    
    def get_allEncoder(self, x):
        # out = self.encoder_k.get_embedding(x)
        h = self.encoder_q.encoder(x)
        
        out = self.encoder_q.fc(h)        
        return out
