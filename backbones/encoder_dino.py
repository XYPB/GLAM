import os
import types
from typing import Callable
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from transformers import logging


from backbones.utils import (
    masked_only_prepare_tokens_with_masks,
    _parse_dinov2_model_name,
    _make_dinov2_model,
)

logging.set_verbosity_error()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MY_API_TOKEN = "<replace-with-your-hf-api-token>"


class GlobalEmbedding(nn.Module):
    def __init__(
        self, input_dim: int = 768, hidden_dim: int = 2048, output_dim: int = 512
    ) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False),  # output layer
        )

    def forward(self, x):
        return self.head(x)


class LocalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False),  # output layer
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)

        return x.permute(0, 2, 1)


class AttentionalPooler(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        n_head: int = 8,
        n_queries: int = 256,
        norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.n_head = n_head
        self.n_queries = n_queries
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(
            d_model, n_head, kdim=context_dim, vdim=context_dim
        )
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        if attn_mask is not None:
            # convert to float
            attn_mask[attn_mask == 0] = -1e9
            attn_mask[attn_mask > 0] = 0
            attn_mask = attn_mask.to(dtype=x.dtype)
            # (N, L) -> (N*num_heads, num_queries, L)
            attn_mask = (
                attn_mask.unsqueeze(1)
                .repeat(1, self.n_head, 1)
                .reshape(N * self.n_head, -1)
            )
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_queries, 1)
        out = self.attn(
            q.unsqueeze(1).expand(-1, N, -1),
            x,
            x,
            need_weights=False,
            attn_mask=attn_mask,
        )[0]
        return out.permute(1, 0, 2)  # LND -> NLD


class DinoEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "dinov2_vitb14_reg_lc",
        img_size: int = 224,
        text_feat_dim: int = 768,
        output_dim: int = 512,
        hidden_dim: int = 2048,
        img_mask_ratio: float = 0,
        freeze_vit: bool = False,
        pretrained: bool = True,
        linear_proj: bool = False,
        linear_local: bool = False,
        num_freeze_blocks: int = 0,
        vit_grad_ckpt: bool = False,
        **kwargs,
    ):
        super(DinoEncoder, self).__init__()

        self.model_name = model_name
        self.output_dim = output_dim
        self.text_feat_dim = text_feat_dim

        if "dinov2" in model_name:
            arch_name, pretrained, num_register_tokens, patch_size = (
                _parse_dinov2_model_name(model_name)
            )
            self.model = _make_dinov2_model(
                arch_name=arch_name,
                patch_size=patch_size,
                pretrained=pretrained,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=True,
                interpolate_offset=0.0,
                grad_ckpt=vit_grad_ckpt,
            )
        else:
            print(self.model_name)
            raise NotImplementedError
        if img_mask_ratio > 0:
            # self.model.random_masking = types.MethodType(random_masking, self.model)
            self.model.prepare_tokens_with_masks = types.MethodType(
                masked_only_prepare_tokens_with_masks, self.model
            )
            self.model.mask_ratio = img_mask_ratio

        self.model.mask_token.requires_grad = False  # never train the mask token

        self.feature_dim = self.model.embed_dim

        if linear_proj:
            self.global_embed = nn.Linear(self.feature_dim, output_dim)
        else:
            self.global_embed = GlobalEmbedding(
                self.feature_dim, hidden_dim, output_dim
            )

        if freeze_vit:
            print("Freezing vit model")
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.global_embed.parameters():
                param.requires_grad = False

        if num_freeze_blocks > 0:
            pass  # TODO

    def vit_forward(self, x):
        return self.model(x, is_training=True)

    def forward(self, x, get_local=False):
        ret = self.vit_forward(x)
        return (
            ret["x_norm_clstoken"].contiguous(),
            ret["x_norm_patchtokens"].contiguous(),
            ret["x_prenorm"].contiguous(),
        )
