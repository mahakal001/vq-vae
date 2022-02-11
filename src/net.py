from torch import nn
import torch
import torch.nn.functional as F

import math
from PIL import Image


import matplotlib.pyplot as plt
import numpy as np

def encoder_conv_block(ic, oc, ks, stride, padding):
    return nn.Sequential(
              nn.Conv2d(ic, oc, ks, stride=stride, padding=padding),
              nn.BatchNorm2d(oc),
              nn.ReLU(),
            )

def decoder_conv_block(ic, oc, ks, padding, scale_factor=2):
    return nn.Sequential(
                nn.Upsample(scale_factor = scale_factor),
                nn.Conv2d(ic, oc, ks, padding = 1),
                nn.ReLU()
            )

class Decoder(nn.Module):
    
    def __init__(self):
        super(Decoder, self).__init__()
        self.input_block =  decoder_conv_block(8, 16, 3, padding=(1,1)) 
        self.conv_2 = decoder_conv_block(16, 16, 3, padding=(1,1)) 
        self.conv_3 = decoder_conv_block(16, 8, 3, padding=(1,1)) 
        self.conv_4 = decoder_conv_block(8, 3, 3, padding=(1,1)) 
        
        self.up = nn.Upsample(size=(28, 28))
        self.conv = nn.Conv2d(3, 1, 3, padding = 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.input_block(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.up(x)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x         

class Encoder(nn.Module):
    
    def __init__(self):
        super(Encoder, self).__init__()
        self.input_block =  encoder_conv_block(1, 16, 3, stride=2, padding=(1,1)) 
        self.conv_2 = encoder_conv_block(16, 16, 3, stride=2, padding=(1,1)) 
        self.conv_3 = encoder_conv_block(16, 8, 3, stride=2, padding=(1,1)) 
        self.conv_4 = encoder_conv_block(8, 8, 3, stride=2, padding=(1,1)) 
        
    def forward(self, x):
        x = self.input_block(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta):
        super(VectorQuantizer, self).__init__()
        
        self._num_embeddings = num_embeddings
        self._embedding_dim =  embedding_dim
        self.beta = beta
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
    
    def forward(self, inputs):
        # x : (Batch, 8, 16, 16)
        
        inputs = inputs.permute( 0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # flatten input
        x = inputs.view(-1, self._embedding_dim)
        
        # now we have codebook as self._embedding and encoder output (z) as ( Batch * H * W, embedding_dim) dimensional matrix
        # for each row in z we need to compute it's distance with every embedding in codebook, then we return the index of embedding which is closest to it.
        # code inspiration from https://www.youtube.com/watch?v=VZFVUrYcig0&t=1393s, https://keras.io/examples/generative/vq_vae/
        dist = (torch.sum(x**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(x, self._embedding.weight.t()))
        
        codebook_indices = torch.argmin(dist, axis = 1)
        quantized = self._embedding.weight[codebook_indices]
        quantized = quantized.view(input_shape)
        
        # loss
        e_latent_loss = self.beta * F.mse_loss( quantized.detach(), inputs)
        q_latent_loss = F.mse_loss( quantized, inputs.detach())
        loss = e_latent_loss + q_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        return loss, quantized.permute(0, 3, 1, 2)
    

class VQVAETrainer(nn.Module):
    def __init__(self):
        super(VQVAETrainer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.vq = VectorQuantizer(1024, embedding_dim = 8, beta = 0.25)
        
    def forward(self, input):
        ze_x = self.encoder(input)
        loss, zq_x = self.vq(ze_x)
        output = self.decoder(zq_x)
        
        return output, loss

