# coding: utf-8
import torch
import torch.nn as nn

from torch import Tensor
from encoder import Encoder
from einops import rearrange, repeat
# from ACD import ACD
from ACD import ACD
from batch import Batch
from embeddings import Embeddings
from vocabulary import Vocabulary
from initialization import initialize_model
from constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, TARGET_PAD
from spl_ae import SPL_AE

class Model(nn.Module):
    def __init__(self,cfg: dict, 
                 encoder: Encoder, 
                 ACD: ACD, 
                 SPL_AE: SPL_AE,
                 src_embed: Embeddings, 
                 src_vocab: Vocabulary, 
                 trg_vocab: Vocabulary, 
                 in_trg_size: int, 
                 out_trg_size: int,
                 is_pretrain: bool):
        """
        Create Sign-IDD

        :param encoder: encoder
        :param ACD: ACD
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        super(Model, self).__init__()

        model_cfg = cfg["model"]
        self.src_embed = src_embed
        self.encoder = encoder
        self.ACD = ACD
        self.SPL_AE = SPL_AE
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = self.src_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.src_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.src_vocab.stoi[EOS_TOKEN]
        self.target_pad = TARGET_PAD

        self.use_cuda = cfg["training"]["use_cuda"]

        self.in_trg_size = in_trg_size
        self.out_trg_size = out_trg_size
        
        if not is_pretrain:
            for param in self.SPL_AE.parameters():
                param.requires_grad = False
            self.SPL_AE.eval()

    def forward(self, is_train: bool, is_pretrain: bool, src: Tensor, trg_input: Tensor, src_mask: Tensor, src_lengths: Tensor, trg_mask: Tensor):

        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param src_lengths: length of source inputs
        :param trg_mask: target mask
        :return: diffusion_output
        """
        pose_length = trg_mask[...,0].sum(dim=-1).ravel()
        if is_pretrain:
            pred_pose, mu, logvar = self.SPL_AE(trg_input, pose_length)
            
            return pred_pose, mu, logvar
        
        else:
            # Encode the source sequence
            encoder_output = self.encode(src=src,
                                        src_length=src_lengths,
                                        src_mask=src_mask)
            trg_input = rearrange(trg_input, "b f (n c) -> b f n c", c=3)
            trg_input = self.SPL_AE.encode_pose(trg_input, pose_length)
            h = self.SPL_AE.bottleneck(trg_input)
            mu = self.SPL_AE.fc_mu(h)  # [B, K, latent_dim]
            logvar = self.SPL_AE.fc_logvar(h)  # [B, K, latent_dim]
            
            # Reparameterization
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            B, T, C = z.shape
            trg_mask = torch.ones(B, 1, T, T, dtype=torch.bool, device=trg_input.device)
            # Diffusion the target sequence
            diffusion_output = self.diffusion(is_train=is_train,
                                            encoder_output=encoder_output,
                                            trg_input=z,
                                            src_mask=src_mask,
                                            trg_mask=trg_mask,
                                            pose_length=pose_length)
            # diffusion_output = self.SPL_AE.decode_pose(diffusion_output, pose_length)
            # diffusion_output = rearrange(diffusion_output, "b f n c -> b f (n c)")
            return diffusion_output

    def encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor):

        """
        Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs
        """

        # Encode an embedded source
        encode_output = self.encoder(embed_src=self.src_embed(src), 
                                     src_length=src_length, 
                                     mask=src_mask)

        return encode_output
    
    def diffusion(self, is_train: bool, encoder_output: Tensor, src_mask: Tensor, trg_input: Tensor, trg_mask: Tensor, pose_length: Tensor):
        
        """
        diffusion the target sentence.

        :param src: param encoder_output: encoder states for attention computation
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param trg_mask: mask for target steps
        :return: diffusion outputs
        """

        diffusion_output = self.ACD(is_train=is_train,
                                    encoder_output=encoder_output,
                                    input_3d=trg_input,
                                    src_mask=src_mask, 
                                    trg_mask=trg_mask)
        diffusion_output = self.SPL_AE.decode_pose(diffusion_output, pose_length)
        diffusion_output = rearrange(diffusion_output, "b f n c -> b f (n c)")
        
        return diffusion_output
    
    def get_loss_for_batch(self, is_train, is_pretrain, batch: Batch, loss_function: nn.Module) -> Tensor:
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param loss_function: loss function, computes for input and target
            a scalar loss for the complete batch
        :return: batch_loss: sum of losses over non-pad elements in the batch
        """
        # Forward through the batch input
        mu, logvar = None, None
        skel_out = self.forward(src=batch.src,
                                trg_input=batch.trg_input[:, :, :150],
                                src_mask=batch.src_mask,
                                src_lengths=batch.src_lengths,
                                trg_mask=batch.trg_mask,
                                is_train=is_train,
                                is_pretrain=is_pretrain)
        if is_pretrain:
            skel_out, mu, logvar = skel_out[0], skel_out[1], skel_out[2]
        # compute batch loss using skel_out and the batch target
        batch_loss = loss_function(skel_out, batch.trg_input[:, :, :150], is_pretrain, mu, logvar)
        
        # return batch loss = sum over all elements in batch that are not pad
        return batch_loss

def build_model(cfg: dict, src_vocab: Vocabulary, trg_vocab: Vocabulary, is_pretrain: bool):

    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """
    full_cfg = cfg
    cfg = cfg["model"]

    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = 0

    # Input target size is the joint vector length plus one for counter
    in_trg_size = cfg["trg_size"]
    # Output target size is the joint vector length plus one for counter
    out_trg_size = cfg["trg_size"]

    # Define source embedding
    src_embed = Embeddings(
        **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
        padding_idx=src_padding_idx)
    
    ## Encoder -------
    enc_dropout = cfg["encoder"].get("dropout", 0.) # Dropout
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
           cfg["encoder"]["hidden_size"], \
           "for transformer, emb_size must be hidden_size"
    
    # Transformer Encoder
    encoder = Encoder(**cfg["encoder"],
                      emb_size=src_embed.embedding_dim,
                      emb_dropout=enc_emb_dropout)
    
    # ACD
    diffusion = ACD(args=cfg, 
                    trg_vocab=trg_vocab)
    
    # SPL_AE
    spl_ae = SPL_AE(embed_dim=cfg["spl"].get('hidden_size', 64),
                    depth=cfg["spl"].get('depth', 1),
                    num_heads=cfg["spl"].get('num_heads', 8),
                    mlp_dim=cfg["spl"].get('ff_size', 64),
                    qkv_bias=cfg["spl"].get('qkv_bias', True),
                    qk_scale=cfg["spl"].get('qk_scale', None),
                    attn_drop_rate=cfg["spl"].get('attn_drop_rate', 0.),
                    drop_rate=cfg["spl"].get('drop_rate', 0.1),
                    max_len=cfg["spl"].get('max_len', 300))
    
    # Define the model
    model = Model(encoder=encoder,
                  ACD=diffusion,
                  SPL_AE=spl_ae,
                  src_embed=src_embed,
                  src_vocab=src_vocab,
                  trg_vocab=trg_vocab,
                  cfg=full_cfg,
                  in_trg_size=in_trg_size,
                  out_trg_size=out_trg_size,
                  is_pretrain=is_pretrain)

    # Custom initialization of model parameters
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx)

    return model