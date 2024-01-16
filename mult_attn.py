import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

L = 10
batch_size = 2  # Parallelize
in_dim = 512    # input
d_model = 512   # output
x = torch.randn((batch_size, L, in_dim))

# x is the random data that is supposed to be the input encoding
# L is the length of the input sequence

# Map input from in_dim to 3 times output vec dimension for q k and v vectors

qkv_layer = nn.Linear(in_dim, d_model * 3)
qkv = qkv_layer(x)

# We now have all the q k and v vectors concatenated for each word in the sentence
# We will split them up into how many ever attention heads we have (8 in this case)

num_heads = 8
head_dim = d_model // num_heads
# Split into heads
qkv = qkv.reshape(batch_size, L, num_heads, 3 * head_dim)
# swap dimensions to make it easier to parallelize between heads
qkv.permute(0, 2, 1, 3)
# separate q k and v
q, k, v = qkv.chunk(3, dim=-1) # -1 for last dimension as q,k and V are concatenated

d_k = q.size()[-1]  # Size of vec

# Attention

# Attention is calculated as softmax(q.k^T / sqrt(d_k)) . v
# q and k are multiplied together to get the attention scores
# Transpose specifying dimensions because 4d tensor
def attention(q, k, v, d_k, mask=False):
    attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
    # Masking for the decoder
    if mask:
        m = torch.full(attn.size(), -np.inf)
        m = torch.triu(m, diagonal=1) # Upper triangular matrix
        attn = attn + m
    attn = F.softmax(attn, dim=-1)      # Row wise softmax
    output = torch.matmul(attn, v)
    return output, attn

val, attn = attention(q, k, v, d_k)

# Concat all the heads together now

val = val.reshape(batch_size, L, d_model)

# Linear layer to map back to original dimension

out = nn.Linear(d_model, in_dim)

##############################################################################################################
# Multi Head Attention all together in a class

class MultiHead_Attention(nn.Module):
    def __init__(self, in_dim, d_model, num_heads):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(in_dim, d_model * 3)
        self.out_layer = nn.Linear(d_model, in_dim)

    def attention(self, q, k, v, d_k, mask=False):
        attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
        if mask:
            m = torch.full(attn.size(), -np.inf)
            m = torch.triu(m, diagonal=1)
            attn = attn + m
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
    
input_dim = 1024
d_model = 512
num_heads = 8
batch_size = 30
# Batch size can accommodate parallelization =, ie.e for parallel sequences and stuff 
sequence_length = 5
x = torch.randn((batch_size, sequence_length, input_dim))

model = MultiHead_Attention(input_dim, d_model, num_heads)
out = model.forward(x)