# coding: utf-8
"""
Flan-T5 기반 Non-Autoregressive Pose Generation Model
Sign-IDD의 기존 구조(Batch, Loss, data loading)와 완전 호환
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, Tuple
from transformers import T5EncoderModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from constants import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, TARGET_PAD
from vocabulary import Vocabulary
from embeddings import Embeddings
from batch import Batch


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class LearnableQueryEmbedding(nn.Module):
    """Learnable query embeddings for non-autoregressive decoding."""
    
    def __init__(self, num_queries: int, d_model: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, d_model))
        nn.init.xavier_uniform_(self.queries)
    
    def forward(self, batch_size: int) -> Tensor:
        return self.queries.unsqueeze(0).expand(batch_size, -1, -1)


class PoseDecoderLayer(nn.Module):
    """Custom Transformer Decoder Layer for Pose Generation."""
    
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
    
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        # Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention
        tgt2 = self.cross_attn(
            tgt, memory, memory,
            key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # FFN
        tgt2 = self.linear2(self.dropout3(F.gelu(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class PoseDecoder(nn.Module):
    """Custom Pose Decoder."""
    
    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 768,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            PoseDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, memory_key_padding_mask)
        return self.norm(output)


class Model(nn.Module):
    """
    Flan-T5 기반 Non-Autoregressive Pose Generation Model.
    Sign-IDD의 Model 클래스와 동일한 인터페이스 제공.
    """
    
    def __init__(
        self,
        cfg: dict,
        src_embed: Embeddings,
        src_vocab: Vocabulary,
        trg_vocab: Vocabulary,
        in_trg_size: int,
        out_trg_size: int
    ):
        super().__init__()
        
        model_cfg = cfg["model"]
        t5_cfg = model_cfg.get("t5", {})
        decoder_cfg = model_cfg.get("decoder", {})
        lora_cfg = model_cfg.get("lora", {})
        
        # Store vocab references (Sign-IDD compatibility)
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = src_vocab.stoi[BOS_TOKEN]
        self.pad_index = src_vocab.stoi[PAD_TOKEN]
        self.eos_index = src_vocab.stoi[EOS_TOKEN]
        self.target_pad = TARGET_PAD
        self.use_cuda = cfg["training"].get("use_cuda", True)
        
        self.in_trg_size = in_trg_size
        self.out_trg_size = out_trg_size
        
        # Model dimensions
        self.d_model = t5_cfg.get("hidden_size", 768)
        self.num_queries = model_cfg.get("num_frames", 300)
        self.pose_dim = model_cfg.get("trg_size", 150)
        
        # Source embedding (기존 Sign-IDD 방식 유지)
        self.src_embed = src_embed
        
        # Projection from embedding dim to T5 hidden size
        embed_dim = src_embed.embedding_dim
        self.embed_proj = nn.Linear(embed_dim, self.d_model) if embed_dim != self.d_model else nn.Identity()
        
        # T5 Encoder (transformer만 사용, embedding은 기존 것 사용)
        t5_model_path = t5_cfg.get("model_path", "google/flan-t5-base")
        
        # 로컬 경로가 존재하지 않으면 HuggingFace에서 로드
        import os
        if not os.path.exists(t5_model_path):
            print(f"Local path '{t5_model_path}' not found. Loading from HuggingFace: google/flan-t5-base")
            t5_model_path = "google/flan-t5-base"
        
        # T5 encoder layers만 로드
        from transformers import T5Config, T5EncoderModel
        self.t5_encoder = T5EncoderModel.from_pretrained(t5_model_path)
        
        # Apply LoRA
        if lora_cfg.get("use_lora", True):
            lora_config = LoraConfig(
                r=lora_cfg.get("r", 16),
                lora_alpha=lora_cfg.get("alpha", 32),
                target_modules=lora_cfg.get("target_modules", ["q", "v"]),
                lora_dropout=lora_cfg.get("dropout", 0.1),
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            self.t5_encoder = get_peft_model(self.t5_encoder, lora_config)
            self.t5_encoder.print_trainable_parameters()
        
        # Learnable Query Embedding
        self.query_embed = LearnableQueryEmbedding(self.num_queries, self.d_model)
        
        # Positional Encoding for queries
        self.query_pos_enc = SinusoidalPositionalEncoding(
            self.d_model, 
            max_len=self.num_queries,
            dropout=decoder_cfg.get("dropout", 0.1)
        )
        
        # Custom Pose Decoder
        self.decoder = PoseDecoder(
            num_layers=decoder_cfg.get("num_layers", 6),
            d_model=self.d_model,
            nhead=decoder_cfg.get("num_heads", 8),
            dim_feedforward=decoder_cfg.get("ff_size", 2048),
            dropout=decoder_cfg.get("dropout", 0.1)
        )
        
        # Output projection
        self.pose_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.pose_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize decoder and output head weights."""
        for module in [self.decoder, self.pose_head]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
    
    def encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor) -> Tensor:
        """
        Encode source sequence.
        
        Args:
            src: Source token ids [B, src_len]
            src_length: Source lengths [B]
            src_mask: Source mask [B, 1, src_len] (Sign-IDD format)
        
        Returns:
            Encoder output [B, src_len, d_model]
        """
        # Embed source using existing embeddings
        embedded = self.src_embed(src)  # [B, src_len, embed_dim]
        
        # Project to T5 dimension
        embedded = self.embed_proj(embedded)  # [B, src_len, d_model]
        
        # T5 encoder expects inputs_embeds and attention_mask
        # Convert mask: Sign-IDD uses [B, 1, src_len], T5 expects [B, src_len]
        if src_mask.dim() == 3:
            attention_mask = src_mask.squeeze(1)  # [B, src_len]
        else:
            attention_mask = src_mask
        
        # Forward through T5 encoder
        encoder_output = self.t5_encoder(
            inputs_embeds=embedded,
            attention_mask=attention_mask
        ).last_hidden_state
        
        return encoder_output
    
    def decode(self, encoder_output: Tensor, src_mask: Tensor) -> Tensor:
        """
        Decode pose sequence.
        
        Args:
            encoder_output: [B, src_len, d_model]
            src_mask: [B, 1, src_len] or [B, src_len]
        
        Returns:
            Pose output [B, num_queries, pose_dim]
        """
        batch_size = encoder_output.size(0)
        
        # Get learnable queries
        queries = self.query_embed(batch_size)
        queries = self.query_pos_enc(queries)
        
        # Convert mask for cross-attention
        if src_mask.dim() == 3:
            memory_mask = src_mask.squeeze(1)
        else:
            memory_mask = src_mask
        memory_key_padding_mask = (memory_mask == 0)
        
        # Decode
        decoder_output = self.decoder(
            tgt=queries,
            memory=encoder_output,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Project to pose
        pose_output = self.pose_head(decoder_output)
        
        return pose_output
    
    def forward(
        self,
        is_train: bool,
        src: Tensor,
        trg_input: Tensor,
        src_mask: Tensor,
        src_lengths: Tensor,
        trg_mask: Tensor
    ) -> Tensor:
        """
        Forward pass (Sign-IDD compatible interface).
        
        Args:
            is_train: Training mode flag
            src: Source token ids [B, src_len]
            trg_input: Target poses [B, trg_len, pose_dim]
            src_mask: Source mask [B, 1, src_len]
            src_lengths: Source lengths [B]
            trg_mask: Target mask (not used in NAR)
        
        Returns:
            Pose output [B, num_queries, pose_dim]
        """
        # Encode
        encoder_output = self.encode(src, src_lengths, src_mask)
        
        # Decode
        pose_output = self.decode(encoder_output, src_mask)
        
        return pose_output
    
    def get_loss_for_batch(self, is_train: bool, batch: Batch, loss_function: nn.Module) -> Tensor:
        """
        Compute loss for a batch (Sign-IDD compatible).
        
        Args:
            is_train: Training mode
            batch: Batch object
            loss_function: Loss function (Sign-IDD Loss class)
        
        Returns:
            Batch loss
        """
        # Forward pass
        pose_pred = self.forward(
            is_train=is_train,
            src=batch.src,
            trg_input=batch.trg_input[:, :, :self.pose_dim],
            src_mask=batch.src_mask,
            src_lengths=batch.src_lengths,
            trg_mask=batch.trg_mask
        )  # [B, num_queries, pose_dim]
        
        # Get target poses
        trg_poses = batch.trg_input[:, :, :self.pose_dim]  # [B, trg_len, pose_dim]
        
        # Pad or truncate target to match num_queries
        batch_size, trg_len, _ = trg_poses.shape
        if trg_len < self.num_queries:
            padding = torch.zeros(
                batch_size, self.num_queries - trg_len, self.pose_dim,
                device=trg_poses.device, dtype=trg_poses.dtype
            )
            trg_poses_padded = torch.cat([trg_poses, padding], dim=1)
        else:
            trg_poses_padded = trg_poses[:, :self.num_queries, :]
        
        # Compute loss using Sign-IDD Loss class
        batch_loss = loss_function(pose_pred, trg_poses_padded)
        
        return batch_loss


def build_model(cfg: dict, src_vocab: Vocabulary, trg_vocab: Vocabulary) -> Model:
    """
    Build Flan-T5 Pose model (Sign-IDD compatible build_model function).
    """
    model_cfg = cfg["model"]
    
    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    
    in_trg_size = model_cfg.get("trg_size", 150)
    out_trg_size = model_cfg.get("trg_size", 150)
    
    # Source embedding (same as Sign-IDD)
    src_embed = Embeddings(
        **model_cfg["encoder"]["embeddings"],
        vocab_size=len(src_vocab),
        padding_idx=src_padding_idx
    )
    
    # Build model
    model = Model(
        cfg=cfg,
        src_embed=src_embed,
        src_vocab=src_vocab,
        trg_vocab=trg_vocab,
        in_trg_size=in_trg_size,
        out_trg_size=out_trg_size
    )
    
    return model