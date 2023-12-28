from torch import nn as nn
from einops import rearrange

# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

class feed_forward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.linear(x) + x


class Attention(nn.Module):
    def __init__(self, heads: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.attn_dim = self.hidden_dim * heads
        self.to_qkv = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.attn_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        # x = rearrange(x, 'b h w c -> b n (h w)')
        qkv = self.to_qkv(x) #.chunk(2, dim=0)
        
        return qkv

    
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = None

class ViT(nn.Module):
    def __init__(self, num_heads=8, patch_width=16, patch_height=16):
        super().__init__()
        self.num_heads = num_heads
        # self.patch_dim = 

    def forward(self, x):
        flatten_x = rearrange(x, 'b h w c -> b n (p p c)', p=16)