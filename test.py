import torch
from einops import rearrange
from model import feed_forward as ff
from model import attention as attn
from dataset import trainloader

BATCH_SIZE = 4
HEIGHT = 32
WIDTH = 32
DIM = 64
CHANNEL = 3

x = torch.rand((4, 12, 768))
x = rearrange(x, 'b n d -> b (n d)')
model = attn(x.shape[1])
model(x)