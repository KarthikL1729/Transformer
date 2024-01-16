# Self attention block
import numpy as np

L, d_k, d_v = 10, 20, 20
# L is length of input seq, d_k is dimension of key, d_v is dimension of value
q = np.random.randn(L, d_k)
k = np.random.randn(L, d_k)
v = np.random.randn(L, d_v)

def softmax(x):
    x = np.exp(x)
    return (x.T / np.sum(x, axis=-1)).T

# self attention = sfmax(q.k^T / sqrt(d_k) + M) . v
# M is the mask matrix, which is used to mask out the future values, only needed for decoder
# Mask is used to prevent the model from cheating by looking at the future values
# M is a lower triangular matrix, with upper triangle filled with -inf
# -inf is used because addition + softmax will result in 0

M = np.tril(np.ones((L, L)))
M[M == 0] = -np.inf
M[M == 1] = 0

def self_attention(q, k, v, M = None):
    attn = np.matmul(q, k.T) / np.sqrt(d_k)
    if M is not None:
        attn = attn + M
    attn = softmax(attn)
    output = np.matmul(attn, v)
    return output, attn

val, attn = self_attention(q, k, v)
print(val)
print(attn)