import torch
from torch import nn
import torch.nn.functional as F



class Encoder_AE(nn.Module):
    def __init__(self,
                 in_features,
                 num_cluster,
                 latent_features = [1024, 512, 128],
                 device="cpu",
                 p=0.0):
        super().__init__()
        self.in_features = in_features
        self.latent_features = latent_features
        decoder_features = latent_features[::-1]
        #decoder_features = [128, 512, 1024]
        
        
        self.device = device

        layers = [] 
        #layers.append(nn.Dropout(p=p)) V1500test V1600test
        layers.append(nn.Dropout(p=p))       
        for i in range(len(latent_features)):
            if i == 0:
                layers.append(nn.Linear(in_features, latent_features[i]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(latent_features[i-1], latent_features[i]))
                layers.append(nn.ReLU())
        
        layers = layers[:-1]
        self.encoder = nn.Sequential(*layers)
        
        layers = []
        #layers.append(nn.Dropout(p=p)) #在0316 22：00修改的，测试301_allAlphaLams时启用的
        for i in range(len(decoder_features)):
            if i == (len(decoder_features)-1):
                layers.append(nn.Linear(decoder_features[i], in_features))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(decoder_features[i], decoder_features[i+1]))
                layers.append(nn.ReLU())
        
        layers = layers[:-1]
        self.fc = nn.Sequential(*layers)

        #self.latent_fc = nn.Linear(latent_features[-1], num_cluster)
        
        
    def forward(self, x):
        h = self.encoder(x)
        out = self.fc(h)

        return out
    
    def get_embedding_test(self, x):
        h = self.encoder(x)
        latent = self.latent_fc(h)
        
        return latent
    
    def get_embedding(self, x):
        latent = self.encoder(x)
        
        return latent
    
    
class Encoder_NoDecoder(nn.Module):
    def __init__(self,
                 in_features,
                 num_cluster,
                 latent_features = [1024, 512, 128],
                 device="cpu",
                 p=0.0):
        super().__init__()
        self.in_features = in_features
        self.latent_features = latent_features
        decoder_features = latent_features[::-1]
        #decoder_features = [128, 512, 1024]
        
        
        self.device = device

        layers = []
        layers.append(nn.Dropout(p=p))
        for i in range(len(latent_features)):
            if i == 0:
                layers.append(nn.Linear(in_features, latent_features[i]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(latent_features[i-1], latent_features[i]))
                layers.append(nn.ReLU())
        
        layers = layers[:-1]
        self.encoder = nn.Sequential(*layers)
    
    def get_embedding(self, x):
        latent = self.encoder(x)

        return latent
    
    
class Encoder_AE_0312new(nn.Module):
    def __init__(self,
                 in_features,
                 num_cluster,
                 latent_features = [1024, 512, 128],
                 device="cpu",
                 p=0.0):
        super().__init__()
        self.in_features = in_features
        self.latent_features = latent_features
        decoder_features = latent_features[::-1]
        #decoder_features = [128, 512, 1024]
        
        
        self.device = device

        layers = []
        layers.append(nn.Dropout(p=p))
        for i in range(len(latent_features)):
            if i == 0:
                layers.append(nn.Linear(in_features, latent_features[i]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(latent_features[i-1], latent_features[i]))
                layers.append(nn.ReLU())
        
        layers = layers[:-1]
        self.encoder = nn.Sequential(*layers)
        
        layers = []
        layers.append(nn.Dropout(p=p))
        for i in range(len(decoder_features)):
            if i == (len(decoder_features)-1):
                layers.append(nn.Linear(decoder_features[i], in_features))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(decoder_features[i], decoder_features[i+1]))
                layers.append(nn.ReLU())
        
        layers = layers[:-1]
        self.fc = nn.Sequential(*layers)

        #self.fc = nn.Linear(latent_features[-1], num_cluster)
        
        
    def forward(self, x):
        h = self.encoder(x)
        out = self.fc(h)

        return out
    
    def get_embedding(self, x):
        latent = self.encoder(x)

        return latent