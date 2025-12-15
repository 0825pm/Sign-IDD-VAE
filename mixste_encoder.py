# coding: utf-8
"""
MixSTE Encoder for Sign-MixSTE-IDD
Based on HoT (Hourglass Tokenizer) MixSTE architecture
Adapted for Sign Language Production task
"""

import torch
import torch.nn as nn
from functools import partial
from einops import rearrange


class Mlp(nn.Module):
    """MLP block for transformer"""
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Block(nn.Module):
    """Transformer block with attention and MLP"""
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MixSTEEncoder(nn.Module):
    """
    Mixed Spatio-Temporal Encoder for Sign Language Production
    
    Takes 7D bone representation [B, T, J, 7] and applies alternating
    spatial attention (across joints) and temporal attention (across frames)
    
    Args:
        num_joints: Number of joints (default: 50 for sign language)
        in_chans: Input channels per joint (default: 7 for 4D bone rep)
        embed_dim: Embedding dimension (default: 512)
        depth: Number of transformer blocks (default: 4)
        num_heads: Number of attention heads (default: 8)
        mlp_ratio: MLP hidden dim ratio (default: 2)
        qkv_bias: Whether to use bias in qkv projection
        qk_scale: Scale for attention
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Stochastic depth rate
    """
    def __init__(self, 
                 num_joints=50,
                 num_frames=None,  # Will be set dynamically
                 in_chans=7,
                 embed_dim=512,
                 depth=4,
                 num_heads=8,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1):
        super().__init__()
        
        self.num_joints = num_joints
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.out_chans = 3
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        
        # Spatial patch embedding: 7D -> embed_dim
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim)
        
        # Positional embeddings
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        # Temporal pos embed will be created dynamically based on sequence length
        self.Temporal_pos_embed = None
        self.max_frames = 300  # Maximum expected frames
        self._temporal_pos_embed = nn.Parameter(torch.zeros(1, self.max_frames, embed_dim))
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Spatial Transformer Encoder blocks
        self.STEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        
        # Temporal Transformer Encoder blocks
        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        
        # Layer normalization
        self.Spatial_norm = norm_layer(embed_dim)
        self.Temporal_norm = norm_layer(embed_dim)
        
        # Output projection: embed_dim -> 7 (back to bone representation)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.out_chans),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.Spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self._temporal_pos_embed, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def get_temporal_pos_embed(self, num_frames):
        """Get temporal positional embedding for given number of frames"""
        return self._temporal_pos_embed[:, :num_frames, :]
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [B, T, J, 7] or [B, T, J*7]
               B: batch size
               T: number of frames
               J: number of joints (50)
               7: 7D bone representation (3D coord + 4D bone attr)
        
        Returns:
            Output tensor of shape [B, T, J*7] = [B, T, 350]
        """
        # Handle both [B, T, J, 7] and [B, T, J*7] input shapes
        if x.dim() == 3:
            B, T, D = x.shape
            assert D == self.num_joints * self.in_chans, \
                f"Expected D={self.num_joints * self.in_chans}, got {D}"
            x = x.view(B, T, self.num_joints, self.in_chans)
        
        B, T, J, C = x.shape
        
        # === First block: Initial Spatial + Temporal ===
        # Spatial attention: (B*T, J, C) -> embed_dim
        x = rearrange(x, 'b t j c -> (b t) j c')
        x = self.Spatial_patch_to_embedding(x)  # (B*T, J, embed_dim)
        x = x + self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = self.STEblocks[0](x)
        x = self.Spatial_norm(x)
        
        # Temporal attention: (B*J, T, embed_dim)
        x = rearrange(x, '(b t) j c -> (b j) t c', t=T)
        temporal_pos = self.get_temporal_pos_embed(T)
        x = x + temporal_pos
        x = self.pos_drop(x)
        x = self.TTEblocks[0](x)
        x = self.Temporal_norm(x)
        
        # Reshape back to (B, T, J, C)
        x = rearrange(x, '(b j) t c -> b t j c', j=J)
        
        # === Remaining blocks ===
        for i in range(1, self.depth):
            # Spatial attention
            x = rearrange(x, 'b t j c -> (b t) j c')
            x = self.STEblocks[i](x)
            x = self.Spatial_norm(x)
            
            # Temporal attention
            x = rearrange(x, '(b t) j c -> (b j) t c', t=T)
            x = self.TTEblocks[i](x)
            x = self.Temporal_norm(x)
            
            # Reshape back
            x = rearrange(x, '(b j) t c -> b t j c', j=J)
        
        # Output projection: embed_dim -> 7
        x = self.output_proj(x)  # (B, T, J, 7)
        
        # Flatten to (B, T, J*7)
        x = x.view(B, T, J * self.out_chans)
        
        return x


class MixSTEEncoderForACD(nn.Module):
    """
    MixSTE Encoder wrapper designed to work with Sign-IDD's ACD module
    
    This version keeps the output in embed_dim space for fusion with
    the original 7D representation before feeding into ACD_Denoiser
    """
    def __init__(self, 
                 num_joints=50,
                 in_chans=7,
                 embed_dim=512,
                 depth=4,
                 num_heads=8,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1):
        super().__init__()
        
        self.num_joints = num_joints
        self.in_chans = in_chans
        self.out_chans = 3
        self.embed_dim = embed_dim
        self.depth = depth
        self.out_dim = num_joints * self.out_chans  # 350
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        
        # Spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim)
        
        # Positional embeddings
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.max_frames = 512
        self._temporal_pos_embed = nn.Parameter(torch.zeros(1, self.max_frames, embed_dim))
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.STEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        
        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        
        self.Spatial_norm = norm_layer(embed_dim)
        self.Temporal_norm = norm_layer(embed_dim)
        
        # Project from embed_dim back to 350 (J*7)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.out_chans),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.Spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self._temporal_pos_embed, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def get_temporal_pos_embed(self, num_frames):
        return self._temporal_pos_embed[:, :num_frames, :]
    
    def forward(self, x):
        """
        Args:
            x: [B, T, 350] - 7D bone representation
        Returns:
            [B, T, 350] - Enhanced representation
        """
        B, T, D = x.shape
        J = self.num_joints
        C = self.in_chans
        
        x = x.view(B, T, J, C)
        
        # First block
        x = rearrange(x, 'b t j c -> (b t) j c')
        x = self.Spatial_patch_to_embedding(x)
        x = x + self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = self.STEblocks[0](x)
        x = self.Spatial_norm(x)
        
        x = rearrange(x, '(b t) j c -> (b j) t c', t=T)
        x = x + self.get_temporal_pos_embed(T)
        x = self.pos_drop(x)
        x = self.TTEblocks[0](x)
        x = self.Temporal_norm(x)
        
        x = rearrange(x, '(b j) t c -> b t j c', j=J)
        
        # Remaining blocks
        for i in range(1, self.depth):
            x = rearrange(x, 'b t j c -> (b t) j c')
            x = self.STEblocks[i](x)
            x = self.Spatial_norm(x)
            
            x = rearrange(x, '(b t) j c -> (b j) t c', t=T)
            x = self.TTEblocks[i](x)
            x = self.Temporal_norm(x)
            
            x = rearrange(x, '(b j) t c -> b t j c', j=J)
        
        # Project back to 7D per joint
        x = self.output_proj(x)  # (B, T, J, 7)
        x = x.view(B, T, J * self.out_chans)  # (B, T, 350)
        
        return x


if __name__ == '__main__':
    # Test the encoder
    B, T, J, C = 2, 64, 50, 7
    
    # Test MixSTEEncoder
    encoder = MixSTEEncoder(
        num_joints=J,
        in_chans=C,
        embed_dim=256,
        depth=4,
        num_heads=8
    )
    
    x = torch.randn(B, T, J * C)  # [B, T, 350]
    out = encoder(x)
    print(f"MixSTEEncoder: Input shape: {x.shape}, Output shape: {out.shape}")
    
    # Test MixSTEEncoderForACD
    encoder_acd = MixSTEEncoderForACD(
        num_joints=J,
        in_chans=C,
        embed_dim=256,
        depth=4,
        num_heads=8
    )
    
    out_acd = encoder_acd(x)
    print(f"MixSTEEncoderForACD: Input shape: {x.shape}, Output shape: {out_acd.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in encoder.parameters())
    print(f"MixSTEEncoder parameters: {params / 1e6:.2f}M")
    
    params_acd = sum(p.numel() for p in encoder_acd.parameters())
    print(f"MixSTEEncoderForACD parameters: {params_acd / 1e6:.2f}M")
