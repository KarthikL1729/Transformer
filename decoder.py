# Transformer decoder

import encoder as en
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHead_CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model, d_model * 2)
        self.q_layer = nn.Linear(d_model, d_model)
        self.out_layer = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, d_k, mask=None):
        attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
        if mask != None:
            attn += mask
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        return output, attn
    
    def forward(self, x, y, mask = False):
        batch_size, L, d_model = x.size()
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        kv = kv.reshape(batch_size, L, self.num_heads, 2 * self.head_dim)
        q = q.reshape(batch_size, L, self.num_heads, self.head_dim)
        kv.permute(0, 2, 1, 3)
        q.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        d_k = q.size()[-1]
        val, attn = self.attention(q, k, v, d_k, mask)
        val = val.reshape(batch_size, L, self.d_model)
        out = self.out_layer(val)
        return out
    
class DecoderLayer(nn.module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attn = en.MultiHead_Attention(d_model = d_model, num_heads = num_heads)
        self.norm1 = en.LayerNorm(parameters_shape = [d_model])
        self.dropout1 = nn.Dropout(p = drop_prob)
        self.enc_dec_attn = MultiHead_CrossAttention(d_model = d_model, num_heads = num_heads)
        self.norm2 = en.LayerNorm(parameters_shape = [d_model])
        self.dropout2 = nn.Dropout(p = drop_prob)
        self.ffn = en.FeedForward(d_model = d_model, hidden = ffn_hidden, drop_prob = drop_prob)
        self.norm3 = en.LayerNorm(parameters_shape = [d_model])
        self.dropout3 = nn.Dropout(p = drop_prob)

    def forward(self, x, y, mask):
        # same flow as diagram of transformer decoder
        residual_branch = y
        y = self.self_attn(y, mask = mask)
        y = self.dropout1(y)
        y = self.norm1(y + residual_branch)
        residual_branch = y
        y = self.enc_dec_attn(x, y)
        y = self.dropout2(y)
        y = self.norm2(y + residual_branch)
        residual_branch = y
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.norm3(y + residual_branch)
        return y

class SequentialDecoder(nn.Sequential):
    # Because stock nn.Sequential only supports one input
    def forward(self, *inputs):
        x, y, mask = inputs
        for module in self._modules.values():
            y = module(x, y, mask)          # pass all inputs in every module
        return y

class Decoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers=1):
        super().__init__()
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])
    
    def forward(self, x, y, mask):
        y = self.layers(x, y, mask)
        return y