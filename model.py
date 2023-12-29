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
        return self.linear(x) 


class attention(nn.Module):
    def __init__(self, embedded_dim):
        super().__init__()
        self.embedded_dim = embedded_dim
        # self.attn_dim = self.hidden_dim * 3
        self.to_qkv = nn.Sequential(
            nn.LayerNorm(self.embedded_dim),
            nn.Linear(self.embedded_dim, self.embedded_dim * 3),
            nn.GELU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        # b n d -> b n d
        q = self.to_qkv(x) #.chunk(3, dim=1)
        print(q.shape)
        
        return None

    
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = attention(1, )

    def forward(self, x):
        return self.block(x)


class ViT(nn.Module):
    def __init__(self, num_heads=8, patch_width=16, patch_height=16):
        super().__init__()
        self.num_heads = num_heads
        # self.patch_dim = 

    def forward(self, x):
        flatten_x = rearrange(x, 'b h w c -> b n (p p c)', p=16)