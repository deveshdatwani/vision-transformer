import torch
from einops import rearrange
from model import feed_forward as ff
from model import Attention as attn

BATCH_SIZE = 4
HEIGHT = 32
WIDTH = 32
DIM = 64
CHANNEL = 3

x = torch.rand((BATCH_SIZE, CHANNEL, HEIGHT, WIDTH))

y = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)

print(f'X shape: {x.shape}')
print(f'Y shape: {y.shape}')