# src/gfm4mpm/models/mae_vit.py
import math
from typing import Tuple
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=8, embed_dim=256, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):  # (B,C,H,W) -> (B, N, D)
        x = self.proj(x)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1,2)
        return x, H, W

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, p=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=p, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)), nn.GELU(), nn.Dropout(p),
            nn.Linear(int(dim*mlp_ratio), dim), nn.Dropout(p)
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class MAEViT(nn.Module):
    """Masked Autoencoder with ViT encoder + tiny decoder."""
    def __init__(self, in_chans=8, embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0, patch_size=4, dec_dim=128, dec_depth=2):
        super().__init__()
        self.patch = PatchEmbed(in_chans, embed_dim, patch_size)
        self.pos_embed = None
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        # decoder
        self.dec_proj = nn.Linear(embed_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1,1,dec_dim))
        self.dec_pos = None
        self.dec_blocks = nn.ModuleList([TransformerBlock(dec_dim, num_heads, 2.0) for _ in range(dec_depth)])
        self.head = nn.Linear(dec_dim, patch_size*patch_size*in_chans)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.patch_size = patch_size
        self.in_chans = in_chans

    def _positional(self, H, W, dim):
        pe = self._build_2d_sincos_pos_embed(H, W, dim)
        return pe

    @staticmethod
    def _build_2d_sincos_pos_embed(H, W, dim):
        grid_h = torch.arange(H, dtype=torch.float32)
        grid_w = torch.arange(W, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(grid_h, grid_w, indexing='ij'), dim=0)  # (2, H, W)
        grid = grid.reshape(2, 1, H*W)
        emb_h = MAEViT._build_sincos(grid[0], dim//2)
        emb_w = MAEViT._build_sincos(grid[1], dim//2)
        pos = torch.cat([emb_h, emb_w], dim=1).transpose(0,2)  # (N, dim)
        return pos

    @staticmethod
    def _build_sincos(pos, dim):
        omega = torch.arange(dim//2, dtype=torch.float32) / (dim/2.)
        omega = 1.0 / (10000**omega)
        out = torch.einsum('np,d->ndp', pos, omega)
        sin = torch.sin(out)
        cos = torch.cos(out)
        return torch.cat([sin, cos], dim=1)  # (N, dim)

    def random_mask(self, x, keep_ratio=0.25):
        B, N, D = x.shape
        len_keep = int(N * keep_ratio)
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_keep = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1,-1,D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_keep, mask, ids_restore

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        # encoder
        x, H, W = self.patch(x)
        pos = self._positional(H, W, x.shape[-1]).to(x.device)  # (N,D)
        x = x + pos.unsqueeze(0)
        x_keep, mask, ids_restore = self.random_mask(x, keep_ratio=0.25)
        for blk in self.blocks:
            x_keep = blk(x_keep)
        latents = self.norm(x_keep)
        # decoder
        dec_tokens = self.dec_proj(latents)
        B, Nk, Dd = dec_tokens.shape
        N = H*W
        mask_tokens = self.mask_token.expand(B, N - Nk, Dd)
        x_ = torch.cat([dec_tokens, mask_tokens], dim=1)
        # unshuffle
        idx = ids_restore.unsqueeze(-1).expand(-1,-1,Dd)
        x_ = torch.gather(x_, 1, idx)
        dec_pos = self._positional(H, W, Dd).to(x.device)
        x_ = x_ + dec_pos.unsqueeze(0)
        for blk in self.dec_blocks:
            x_ = blk(x_)
        pred = self.head(x_)  # (B, N, p*p*C)
        pred = pred.view(B, H, W, self.patch_size, self.patch_size, self.in_chans)
        pred = pred.permute(0,5,1,3,2,4).contiguous().view(B, self.in_chans, H*self.patch_size, W*self.patch_size)
        return pred, mask

    @torch.no_grad()
    def encode(self, x):
        z, H, W = self.patch(x)
        pos = self._positional(H, W, z.shape[-1]).to(z.device)
        z = z + pos.unsqueeze(0)
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        # mean pool tokens
        return z.mean(dim=1)
