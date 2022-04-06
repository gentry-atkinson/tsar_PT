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
from torch.autograd import Variable
from gen_ts_data import generate_pattern_data_as_array
from random import randint

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
        self.train_X = np.loadtxt(filename, delimiter=',')
        self.train_X = Variable(torch.from_numpy(self.train_X).float(), requires_grad=False)
        m=len(self.train_X)
        train_data, val_data = random_split(self.train_X, [int(m-m*0.2), int(m*0.2)])
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=10)
        self.valid_loader = torch.utils.data.DataLoader(val_data, batch_size=10)

    def train_epoch(self):
        self.enc.train()
        self.dec.train()
        train_loss = []
        for signal_batch, _ in self.train_loader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
            # Move tensor to the proper device
            signal_batch = signal_batch.to(self.device)
            # Encode data
            encoded_data = self.inc(signal_batch)
            # Decode data
            decoded_data = self.dec(encoded_data)
            # Evaluate loss
            loss = self.loss_fn(decoded_data, signal_batch)
            # Backward pass
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            # Print batch loss
            print('\t partial train loss (single batch): %f' % (loss.data))
            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)

    def test_epoch(self):
        # Set evaluation mode for encoder and decoder
        self.enc.eval()
        self.dec.eval()
        with torch.no_grad(): # No need to track the gradients
            # Define the lists to store the outputs for each batch
            conc_out = []
            conc_label = []
            for signal_batch, _ in self.valid_loader:
                # Move tensor to the proper device
                signal_batch = signal_batch.to(self.device)
                # Encode data
                encoded_data = self.enc(signal_batch)
                # Decode data
                decoded_data = self.dec(encoded_data)
                # Append the network output and the original image to the lists
                conc_out.append(decoded_data.cpu())
                conc_label.append(signal_batch.cpu())
            # Create a single tensor with all the values in the lists
            conc_out = torch.cat(conc_out)
            conc_label = torch.cat(conc_label) 
            # Evaluate global loss
            val_loss = self.loss_fn(conc_out, conc_label)
        return val_loss.data

    def train(self, X, num_epochs=30):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(
                self.encoder,self.decoder,self.device,train_loader,loss_fn,optim)
            val_loss = test_epoch(encoder,decoder,device,test_loader,loss_fn)
            print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
            diz_loss['train_loss'].append(train_loss)
            diz_loss['val_loss'].append(val_loss)
            plot_ae_outputs(encoder,decoder,n=10)
        
if __name__ == '__main__':
    ae = AE()
    print('Selected device: {}'.format(ae.device))
    demo_data = [generate_pattern_data_as_array() for _ in range(1000)]
    demo_instance = randint(0,999)
    print("Test insantce: ", demo_instance)
    ae.train(demo_data)
