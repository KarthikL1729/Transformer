# Transformer encoder

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# From mult_attn.py
class MultiHead_Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, d_model * 3)
        self.out_layer = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, d_k, mask=None):
        attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
        if mask != None:
            attn += mask
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        return output, attn

    def forward(self, x, mask=False):
        batch_size, L, in_dim = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, L, self.num_heads, 3 * self.head_dim)
        qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        d_k = q.size()[-1]
        val, attn = self.attention(q, k, v, d_k, mask)
        val = val.reshape(batch_size, L, self.d_model)
        out = self.out_layer(val)
        return out

class LayerNorm(nn.Module):
    def __init__(self, parameters_shape, epsilon = 1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.epsilon = epsilon
        self.gamma = nn.parameter(torch.ones(parameters_shape))     # std dev
        self.beta = nn.parameter(torch.zeros(parameters_shape))     # mean
    
    def forward(self, x):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))] # perform on last layer(s), in this case only 1 layer
        mean = x.mean(dims, keepdim = True)     # keepDim means maintain the initial dimensionality (don't reduce dimension is one dimension only has 1 element)
        var = ((x - mean)**2).mean(dims, keepdim = True)
        std = torch.sqrt(var + self.epsilon)
        y = (x - mean) / std
        out = self.gamma * y + self.beta
        return out

# standard nn.Linear layer
class FeedForward(nn.Module):
    def __init(self, d_model, hidden, drop_prob):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(p = drop_prob)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHead_Attention(d_model = d_model, num_heads = num_heads)
        self.norm1 = LayerNorm(parameters_shape = [d_model])
        self.dropout1 = nn.Dropout(p = drop_prob)
        self.ffn = FeedForward(d_model = d_model, hidden = ffn_hidden, drop_prob = drop_prob)
        self.norm2 = LayerNorm(parameters_shape = [d_model])
        self.dropout2 = nn.Dropout(p = drop_prob)

    def forward(self, x):
        #  same flow as diagram of transformer encoder
        residual_branch = x
        x = self.attn(x, mask=False)     # mask is used for decoder
        x = self.dropout1(x)
        x = self.norm1(x + residual_branch)
        residual_branch = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_branch)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, drop_prob, num_layers, ffn_hidden):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])
        # encoder contains multiple layers, num_layers in this case
    
    def forward(self, x):
        x = self.layers(x) # Pass through all 5 layers
        return x
    
d_model = 512
batch_size = 30
num_heads = 8
max_seq_len = 200
drop_prob = 0.1     # Randomly turn off some neurons to force nn to learn along different paths
num_layers = 5      # Number of encoders stacked
ffn_hidden = 2048   # feed forward network hidden layer size

encoder = Encoder(d_model, num_heads, drop_prob, num_layers, ffn_hidden)