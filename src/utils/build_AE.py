#Author: Gentry Atkinson
#Organization: Texas University
#Data: 22 March, 2022
#Build, compile and return a convolutional autoencoder

import numpy as np
import torch
from torch.utils.data import DataLoader,random_split
from torch import nn
from torch import sigmoid
import torch.nn.functional as F
import torch.optim as optim

class Encoder(nn.Module):
#From: https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac  
    def __init__(self, encoded_space_dim=32):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.LazyConv1d(64, 16),
            nn.ReLU(True),
            nn.LazyConv1d(32, 8),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.LazyConv1d(16, 8),
            nn.ReLU(True),
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten()
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.LazyLinear(64),
            nn.ReLU(True),
            nn.Linear(64, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
class Decoder(nn.Module):
#From: https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac     
    def __init__(self, encoded_space_dim=32):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 16*8),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(16,8))

        self.decoder_conv = nn.Sequential(
            nn.LazyConvTranspose1d(16, 8),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.LazyConvTranspose1d(32, 8),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.LazyConvTranspose1d(64, 16),
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = sigmoid(x)
        return x

class AE:
    enc : Encoder
    dec : Decoder
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    weight_decay=1e-05

    def __init__(self) -> None:
        self.loss_fn = nn.MSELoss()
        self.lr= 0.001
        torch.manual_seed(0)
        self.d = 4
        self.enc = Encoder()
        self.dec = Decoder()
        self.params_to_optimize = [
            {'params' : self.enc.parameters()},
            {'params' : self.dec.parameters()}
        ]

        self.optim = torch.optim.Adam(self.params_to_optimize, lr=self.lr, weight_decay=self.weight_decay)

        self.enc.to(self.device)
        self.dec.to(self.device)

    def load_dataset(self, filename):
        
        self.train_X = None

    def train_(self):
        self.enc.train()
        self.dec.train()
        train_loss = []
        
if __name__ == '__main__':
    ae = AE()
    print('Selected device: {}'.format(ae.device))
