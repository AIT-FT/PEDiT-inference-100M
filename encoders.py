from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import AutoencoderKL, AutoencoderTiny
from transformers import AutoTokenizer, MT5EncoderModel


class AttentionPooling(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()

        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"

        self.heads = heads
        self.head_dim = dim // heads

        self.probe = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.null_embedding = nn.Parameter(torch.randn(dim) * 0.02)

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, N, _ = x.shape

        q = self.q_proj(self.probe.expand(B, -1, -1))
        q = q.view(B, 1, self.heads, self.head_dim).transpose(1, 2)

        k = self.k_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)

        attn_mask = None

        if mask is not None:
            bool_mask = mask.view(B, 1, 1, -1).bool()
            has_valid = bool_mask.any(dim=-1, keepdim=True)
            attn_mask = torch.where(has_valid, bool_mask, torch.ones_like(bool_mask))

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, -1)
        out = self.out_proj(out)

        if mask is not None:
            has_valid = mask.view(B, -1).bool().any(dim=-1, keepdim=True)
            out = torch.where(
                has_valid,
                out,
                self.null_embedding.view(1, -1).to(dtype=out.dtype),
            )

        return out


class TextEncoder(nn.Module):
    """
    mT5-based text encoder.

    Modes:
        - include_encoder=True:
            for cache preparation / raw text inference.

        - include_encoder=False:
            for training with cached text features.
    """

    def __init__(
        self,
        model_name: str = "google/mt5-small",
        out_dim: int = 1024,
        include_encoder: bool = True,
        hidden_size: int = 512,
        max_length: int = 128,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.max_length = max_length
        self.include_encoder = include_encoder

        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, out_dim, bias=False),
        )

        self.pooler = AttentionPooling(out_dim, heads=4)

        if include_encoder:
            print(f"[TextEncoder] Loading/downloading tokenizer and encoder '{model_name}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = MT5EncoderModel.from_pretrained(model_name)

            for param in self.encoder.parameters():
                param.requires_grad = False
            print(f"[TextEncoder] '{model_name}' loaded successfully.")
        else:
            self.tokenizer = None
            self.encoder = None

    def train(self, mode: bool = True):
        super().train(mode)

        if self.encoder is not None:
            self.encoder.eval()

        return self

    def encode_raw(self, texts, device: torch.device):
        assert self.include_encoder, "TextEncoder was created without encoder"

        tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self.encoder(**tokens)
            text_features = outputs.last_hidden_state

        return text_features, tokens["attention_mask"]

    def forward_cached(self, text_features: torch.Tensor, mask: torch.Tensor):
        projected_features = self.proj(text_features)
        pooled_features = self.pooler(projected_features, mask)

        return pooled_features, projected_features, mask

    def forward(self, texts, device: torch.device):
        assert self.include_encoder, "For cached training use forward_cached()"

        text_features, mask = self.encode_raw(texts, device)
        return self.forward_cached(text_features, mask)


class TAESD(nn.Module):
    def __init__(self, model_name: str = "madebyollin/taesd"):
        super().__init__()

        print(f"[TAESD] Loading/downloading VAE '{model_name}'...")
        self.vae = AutoencoderTiny.from_pretrained(model_name)

        for param in self.vae.parameters():
            param.requires_grad = False

        self.scale_factor = 1.0
        print(f"[TAESD] '{model_name}' loaded successfully.")

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 2.0 - 1.0
        return self.vae.encode(x).latents

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        image = self.vae.decode(latent).sample
        return (image / 2.0 + 0.5).clamp(0.0, 1.0)


class SDVAE(nn.Module):
    def __init__(self, model_name: str = "stabilityai/sd-vae-ft-mse"):
        super().__init__()

        print(f"[SDVAE] Loading/downloading VAE '{model_name}'...")
        self.vae = AutoencoderKL.from_pretrained(model_name)

        for param in self.vae.parameters():
            param.requires_grad = False

        self.scale_factor = 0.18215
        print(f"[SDVAE] '{model_name}' loaded successfully.")

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 2.0 - 1.0
        dist = self.vae.encode(x).latent_dist
        latent = dist.mode() if hasattr(dist, "mode") else dist.sample()
        return latent * self.scale_factor

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        image = self.vae.decode(latent / self.scale_factor).sample
        return (image / 2.0 + 0.5).clamp(0.0, 1.0)


class SDXLVAE(nn.Module):
    def __init__(self, model_name: str = "madebyollin/sdxl-vae-fp16-fix"):
        super().__init__()

        print(f"[SDXLVAE] Loading/downloading VAE '{model_name}'...")
        self.vae = AutoencoderKL.from_pretrained(model_name)

        for param in self.vae.parameters():
            param.requires_grad = False

        self.scale_factor = 0.13025
        print(f"[SDXLVAE] '{model_name}' loaded successfully.")

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 2.0 - 1.0
        dist = self.vae.encode(x).latent_dist
        latent = dist.mode() if hasattr(dist, "mode") else dist.sample()
        return latent * self.scale_factor

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        image = self.vae.decode(latent / self.scale_factor).sample
        return (image / 2.0 + 0.5).clamp(0.0, 1.0)