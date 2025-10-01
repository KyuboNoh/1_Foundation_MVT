# src/gfm4mpm/models/mae_vit.py
import math
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=8, embed_dim=256, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):  # (B,C,H,W) -> (B, N, D)
        B, C, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Input spatial dims {H}x{W} must be divisible by patch size {self.patch_size}"
            )
        x = self.proj(x)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
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

    def __init__(
        self,
        in_chans: int = 8,
        embed_dim: int = 256,
        depth: int = 6,
        encoder_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        mlp_ratio_dec: float = 2.0,
        patch_size: int = 4,
        dec_dim: int = 128,
        dec_depth: int = 2,
        decoder_num_heads: Optional[int] = None,
        mask_ratio: float = 0.75,
        image_size: Optional[Union[Tuple[int, int], int]] = None,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if encoder_num_heads <= 0:
            raise ValueError("encoder_num_heads must be positive")
        if embed_dim % encoder_num_heads != 0:
            raise ValueError("embed_dim must be divisible by encoder_num_heads")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if mlp_ratio_dec <= 0:
            raise ValueError("mlp_ratio must be positive")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if dec_dim <= 0:
            raise ValueError("dec_dim must be positive")
        if dec_depth < 0:
            raise ValueError("dec_depth must be non-negative")
        if decoder_num_heads is None:
            decoder_num_heads = encoder_num_heads
        if decoder_num_heads <= 0:
            raise ValueError("decoder_num_heads must be positive")
        if dec_dim % decoder_num_heads != 0:
            raise ValueError("dec_dim must be divisible by decoder_num_heads")
        if not (0.0 <= mask_ratio < 1.0):
            raise ValueError("mask_ratio must be in the range [0, 1)")
        self.patch = PatchEmbed(in_chans, embed_dim, patch_size)
        self.pos_embed = None
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, encoder_num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        # decoder
        self.dec_proj = nn.Linear(embed_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1,1,dec_dim))
        self.dec_pos = None
        self.dec_blocks = nn.ModuleList([TransformerBlock(dec_dim, decoder_num_heads, mlp_ratio_dec) for _ in range(dec_depth)])
        self.head = nn.Linear(dec_dim, patch_size*patch_size*in_chans)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = float(mask_ratio)
        self.image_size = self._normalize_image_size(image_size)
        self._pos_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}

    @staticmethod
    def _normalize_image_size(image_size: Optional[Union[Tuple[int, int], int]]) -> Optional[Tuple[int, int]]:
        if image_size is None:
            return None
        if isinstance(image_size, int):
            if image_size <= 0:
                raise ValueError("image_size must be positive")
            return (image_size, image_size)
        if len(image_size) != 2:
            raise ValueError("image_size must be an int or a tuple of two ints")
        h, w = image_size
        if h <= 0 or w <= 0:
            raise ValueError("image_size dimensions must be positive")
        return (int(h), int(w))

    def _positional(self, H, W, dim):
        key = (H, W, dim)
        if key not in self._pos_cache:
            self._pos_cache[key] = self._build_2d_sincos_pos_embed(H, W, dim)
        return self._pos_cache[key]

    @staticmethod
    def _build_2d_sincos_pos_embed(H, W, dim):
        grid_h = torch.arange(H, dtype=torch.float32)
        grid_w = torch.arange(W, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(grid_h, grid_w, indexing='ij'), dim=-1).reshape(-1, 2)  # (N, 2)
        emb_h = MAEViT._build_sincos_1d(grid[:, 0], dim // 2)
        emb_w = MAEViT._build_sincos_1d(grid[:, 1], dim // 2)
        return torch.cat([emb_h, emb_w], dim=1)

    @staticmethod
    def _build_sincos_1d(pos: torch.Tensor, dim: int) -> torch.Tensor:
        if dim % 2 != 0:
            raise ValueError("dim must be even for sin/cos positional encoding")
        omega = torch.arange(dim // 2, dtype=torch.float32) / (dim / 2.0)
        omega = 1.0 / (10000 ** omega)
        out = pos.unsqueeze(1) * omega.unsqueeze(0)
        sin = torch.sin(out)
        cos = torch.cos(out)
        return torch.cat([sin, cos], dim=1)  # (N, dim)

    def random_mask(self, x, keep_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        len_keep = max(1, min(N, int(round(N * keep_ratio))))
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
        if self.image_size is not None:
            expected_h, expected_w = self.image_size
            if x.shape[-2] != expected_h or x.shape[-1] != expected_w:
                raise ValueError(
                    f"Expected input spatial dims {expected_h}x{expected_w}, received {x.shape[-2:]}"
                )
        # encoder
        x, H, W = self.patch(x)
        pos = self._positional(H, W, x.shape[-1]).to(x.device)  # (N,D)
        x = x + pos.unsqueeze(0)
        keep_ratio = 1.0 - self.mask_ratio
        x_keep, mask, ids_restore = self.random_mask(x, keep_ratio=keep_ratio)
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
        if self.image_size is not None:
            expected_h, expected_w = self.image_size
            if x.shape[-2] != expected_h or x.shape[-1] != expected_w:
                raise ValueError(
                    f"Expected input spatial dims {expected_h}x{expected_w}, received {x.shape[-2:]}"
                )
        z, H, W = self.patch(x)
        pos = self._positional(H, W, z.shape[-1]).to(z.device)
        z = z + pos.unsqueeze(0)
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        # mean pool tokens
        return z.mean(dim=1)
