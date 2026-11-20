import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from diffusers import AutoencoderDC
from transformers import AutoModelForCausalLM, AutoTokenizer, MT5EncoderModel


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (self.weight.shape[0],), self.weight.to(x.dtype), self.eps)
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class AttentionPooling(nn.Module):
    def __init__(self, dim: int, heads: int = 8, inner_dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim if inner_dim is not None else min(dim, 512)
        self.heads = heads if (self.inner_dim % heads == 0) else 4
        self.head_dim = self.inner_dim // self.heads

        self.probe = nn.Parameter(torch.randn(1, 1, self.inner_dim) * 0.02)
        self.null_embedding = nn.Parameter(torch.randn(dim) * 0.02)

        self.q_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        target_device = self.k_proj.weight.device
        if x.device != target_device:
            x = x.to(device=target_device)
        if not torch.is_autocast_enabled():
            target_dtype = self.k_proj.weight.dtype
            if x.dtype != target_dtype:
                x = x.to(dtype=target_dtype)

        B, N, _ = x.shape
        probe_in = self.probe.to(device=x.device, dtype=x.dtype).expand(B, -1, -1)

        q = self.q_proj(probe_in).view(B, 1, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if mask is not None:
            bool_mask = mask.view(B, 1, 1, -1).bool()
            has_valid = bool_mask.any(dim=-1, keepdim=True)
            attn_mask = torch.where(has_valid, bool_mask, torch.ones_like(bool_mask))

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, self.inner_dim)
        out = self.out_proj(out)

        if mask is not None:
            has_valid = mask.view(B, -1).bool().any(dim=-1, keepdim=True)
            out = torch.where(
                has_valid,
                out,
                self.null_embedding.view(1, -1).to(device=out.device, dtype=out.dtype),
            )

        return out


class TextEncoder(nn.Module):

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
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = MT5EncoderModel.from_pretrained(model_name)

            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            self.tokenizer = None
            self.encoder = None

    def train(self, mode: bool = True):
        super().train(mode)

        if self.encoder is not None:
            self.encoder.eval()

        return self

    def encode_raw(self, texts, device: torch.device):
        if isinstance(texts, str):
            texts = [texts]
        dev = torch.device(device) if isinstance(device, str) else device

        if not self.include_encoder or self.encoder is None or self.tokenizer is None:
            B = len(texts)
            if self.tokenizer is not None:
                tokens = self.tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokens = {k: v.to(dev) for k, v in tokens.items()}
                feat = torch.zeros(B, self.max_length, self.hidden_size, device=dev)
                return feat, tokens["attention_mask"]
            feat = torch.zeros(B, self.max_length, self.hidden_size, device=dev)
            mask = torch.zeros(B, self.max_length, dtype=torch.long, device=dev)
            for i, t in enumerate(texts):
                if t and len(t.strip()) > 0:
                    num_tok = min(len(t.strip().split()), self.max_length)
                    mask[i, :max(num_tok, 1)] = 1
            return feat, mask

        tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        tokens = {k: v.to(dev) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self.encoder(**tokens)
            text_features = outputs.last_hidden_state

        return text_features, tokens["attention_mask"]

    def forward_cached(self, text_features: torch.Tensor, mask: torch.Tensor):
        projected_features = self.proj(text_features)
        pooled_features = self.pooler(projected_features, mask)

        return pooled_features, projected_features, mask

    def forward(self, texts, device: torch.device):
        text_features, mask = self.encode_raw(texts, device)
        return self.forward_cached(text_features, mask)


class Qwen35TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-4B",
        out_dim: int = 2560,
        include_encoder: bool = True,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        target_layer: int = 28,
        target_layers: Tuple[int, ...] = (8, 16, 28),
        max_length: int = 1024,
        pooler_heads: int = 8,
    ):
        super().__init__()

        self.out_dim = out_dim
        self.include_encoder = include_encoder
        self.target_layer = target_layer
        self.target_layers = tuple(target_layers)
        self.max_length = max_length
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit

        self.proj = nn.Identity()
        self.norm = RMSNorm(out_dim)
        self.pooler = AttentionPooling(out_dim, heads=pooler_heads)

        self.ref_transform = transforms.Compose([
            transforms.RandomResizedCrop(512, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        ])

        self.model = None
        self.tokenizer = None
        self.processor = None

        if include_encoder:
            import gc
            max_req_layer = max(max(self.target_layers) if self.target_layers else 0, self.target_layer)
            use_8bit = load_in_8bit or (os.environ.get("QWEN_8BIT", "0") == "1")

            local_candidates = [
                os.environ.get("QWEN_PATH", ""),
                r"C:\ai_proj\edgeflow_phase_dit_test_atenshen\qwen35weights",
                "qwen35weights",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen35weights"),
                model_name,
            ]
            local_weights_dir = None
            for cand in local_candidates:
                if cand and os.path.isdir(cand):
                    sfs = [f for f in os.listdir(cand) if f.endswith(".safetensors")]
                    if sfs:
                        local_weights_dir = cand
                        break

            target_dev = "cuda" if torch.cuda.is_available() else "cpu"

            if (use_8bit or load_in_4bit) and target_dev == "cuda":
                load_src = local_weights_dir if local_weights_dir is not None else model_name
                q_mode = "8-bit" if use_8bit else "4-bit"
                print(f"[Qwen35TextEncoder] Loading {q_mode} quantized Qwen model from {load_src}...")
                try:
                    from transformers import BitsAndBytesConfig
                    q_cfg = BitsAndBytesConfig(load_in_8bit=True) if use_8bit else BitsAndBytesConfig(load_in_4bit=True)
                    if self.tokenizer is None:
                        try:
                            self.tokenizer = AutoTokenizer.from_pretrained(load_src, trust_remote_code=True)
                        except Exception as e:
                            print(f"[Qwen35TextEncoder] Warning: Failed to load tokenizer from {load_src}: {e}")
                    dev_map = {"": "cuda:0"} if str(target_dev).startswith("cuda") else "auto"
                    self.model = AutoModelForCausalLM.from_pretrained(
                        load_src,
                        quantization_config=q_cfg,
                        device_map=dev_map,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                    )
                    alloc_gb = torch.cuda.memory_allocated() / (1024**3)
                    print(f"[Qwen35TextEncoder] Successfully loaded {q_mode} model into CUDA (VRAM: {alloc_gb:.2f} GB)!")
                except Exception as e:
                    print(f"[Qwen35TextEncoder] Quantized load failed: {e}. Falling back to standard loader.")
                    self.model = None

            if self.model is None and local_weights_dir is not None:
                print(f"[Qwen35TextEncoder] Found local weights in {local_weights_dir}")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(local_weights_dir, trust_remote_code=True)
                except Exception as e:
                    print(f"[Qwen35TextEncoder] Warning: Failed to load local tokenizer: {e}")

                try:
                    from safetensors import safe_open
                    from transformers import AutoConfig
                    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

                    cfg = AutoConfig.from_pretrained(local_weights_dir, trust_remote_code=True)
                    text_cfg = getattr(cfg, "text_config", cfg)
                    if hasattr(text_cfg, "num_hidden_layers") and text_cfg.num_hidden_layers > max_req_layer:
                        text_cfg.num_hidden_layers = max_req_layer

                    print(f"[Qwen35TextEncoder] Instantiating Qwen3_5TextModel ({max_req_layer} layers) on meta device...")
                    with torch.device("meta"):
                        text_model = Qwen3_5TextModel(text_cfg)
                    if hasattr(text_model, "layers") and len(text_model.layers) > max_req_layer:
                        text_model.layers = nn.ModuleList(list(text_model.layers)[:max_req_layer])
                    if hasattr(text_model, "norm"):
                        text_model.norm = nn.Identity()
                    text_model.rotary_emb = type(text_model.rotary_emb)(text_cfg, device=target_dev)

                    def set_param_direct(mod, path, t, dev=target_dev):
                        parts = path.split(".")
                        if parts[0] == "layers" and len(parts) > 1 and parts[1].isdigit():
                            if int(parts[1]) >= max_req_layer:
                                return False
                        if parts[0] in ("norm", "lm_head"):
                            return False
                        curr = mod
                        for p in parts[:-1]:
                            curr = curr[int(p)] if p.isdigit() else getattr(curr, p)
                        name = parts[-1]
                        param = torch.nn.Parameter(t.to(device=dev, dtype=torch.bfloat16), requires_grad=False)
                        if name in curr._buffers:
                            curr._buffers[name] = param.data
                        else:
                            setattr(curr, name, param)
                        return True

                    safetensor_files = sorted([
                        os.path.join(local_weights_dir, f) for f in os.listdir(local_weights_dir) if f.endswith(".safetensors")
                    ])
                    print(f"[Qwen35TextEncoder] Streaming text model weights from {len(safetensor_files)} shard(s) directly to {target_dev}...")
                    loaded_count = 0
                    for shard in safetensor_files:
                        with safe_open(shard, framework="pt", device="cpu") as sf:
                            for k in sf.keys():
                                sub_key = None
                                if k.startswith("model.language_model."):
                                    sub_key = k[len("model.language_model."):]
                                elif k.startswith("model.") and not k.startswith("model.visual"):
                                    sub_key = k[len("model."):]
                                if sub_key is not None:
                                    val = sf.get_tensor(k)
                                    if set_param_direct(text_model, sub_key, val, dev=target_dev):
                                        loaded_count += 1
                                    del val
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    if loaded_count == 0:
                        raise RuntimeError("0 matching text parameters found in safetensors shards.")

                    self.model = text_model
                    alloc_gb = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
                    print(f"[Qwen35TextEncoder] Successfully loaded {loaded_count} parameters into {target_dev} (VRAM: {alloc_gb:.2f} GB)!")
                except Exception as e:
                    print(f"[Qwen35TextEncoder] Direct streaming load failed: {e}. Falling back to standard loader.")
                    self.model = None

            if self.model is None:
                try:
                    if self.tokenizer is None:
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
                except Exception:
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    except Exception as e:
                        print(f"[Qwen35TextEncoder] Warning: Failed to load tokenizer for {model_name}: {e}")
                        self.tokenizer = None

                offline = (os.environ.get("HF_HUB_OFFLINE", "0") == "1")
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        load_in_8bit=use_8bit if not load_in_4bit else False,
                        load_in_4bit=load_in_4bit,
                        device_map="auto" if (load_in_4bit or use_8bit) else None,
                        torch_dtype=torch.bfloat16,
                        local_files_only=True,
                        trust_remote_code=True,
                    )
                except Exception:
                    if not offline:
                        try:
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                load_in_8bit=use_8bit if not load_in_4bit else False,
                                load_in_4bit=load_in_4bit,
                                device_map="auto" if (load_in_4bit or use_8bit) else None,
                                torch_dtype=torch.bfloat16,
                                trust_remote_code=True,
                            )
                        except Exception:
                            try:
                                from transformers import AutoModel
                                self.model = AutoModel.from_pretrained(
                                    model_name,
                                    torch_dtype=torch.bfloat16,
                                    trust_remote_code=True,
                                )
                            except Exception as e:
                                print(f"[Qwen35TextEncoder] Warning: Pretrained weights not available offline ({e}). Running in fallback mode.")
                                self.model = None
                    else:
                        self.model = None

            if self.processor is None:
                try:
                    from transformers import AutoProcessor
                    load_src = local_weights_dir if local_weights_dir is not None else model_name
                    self.processor = AutoProcessor.from_pretrained(load_src, trust_remote_code=True)
                except Exception:
                    self.processor = None

            if self.tokenizer is not None:
                self.tokenizer.padding_side = "right"
                if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

            if self.model is not None:
                for param in self.model.parameters():
                    param.requires_grad = False

                base_m = getattr(self.model, "model", self.model)
                if hasattr(base_m, "layers") and len(base_m.layers) > max_req_layer:
                    base_m.layers = nn.ModuleList(list(base_m.layers)[:max_req_layer])
                if hasattr(base_m, "config") and hasattr(base_m.config, "num_hidden_layers"):
                    base_m.config.num_hidden_layers = min(base_m.config.num_hidden_layers, max_req_layer)
                if hasattr(base_m, "norm"):
                    base_m.norm = nn.Identity()
                if hasattr(self.model, "lm_head"):
                    self.model.lm_head = nn.Identity()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                text_cfg = getattr(self.model.config, "text_config", self.model.config)
                real_dim = getattr(text_cfg, "hidden_size", self.out_dim)
                if real_dim != self.out_dim:
                    self.out_dim = real_dim
                    self.norm = RMSNorm(real_dim)
                    eff_heads = min(pooler_heads, max(1, real_dim // 128))
                    self.pooler = AttentionPooling(real_dim, heads=eff_heads)
                m_dev = next(self.model.parameters()).device
                self.norm = self.norm.to(m_dev)
                self.pooler = self.pooler.to(m_dev)
        else:
            self.tokenizer = None
            self.processor = None
            self.model = None

    def train(self, mode: bool = True):
        super().train(mode)
        if self.model is not None:
            self.model.eval()
        return self

    def to(self, *args, **kwargs):
        device = None
        for a in args:
            if isinstance(a, (torch.device, str)):
                device = torch.device(a)
                break
        if "device" in kwargs:
            device = torch.device(kwargs["device"])
        if device is not None:
            if hasattr(self, "norm") and self.norm is not None:
                self.norm = self.norm.to(device)
            if hasattr(self, "pooler") and self.pooler is not None:
                self.pooler = self.pooler.to(device)
            if hasattr(self, "proj") and self.proj is not None:
                self.proj = self.proj.to(device)
        if self.model is not None:
            try:
                self.model = self.model.to(*args, **kwargs)
            except Exception:
                pass
        return self

    def augment_image(self, img: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
        if isinstance(img, Image.Image):
            return self.ref_transform(img)
        return img

    def encode_raw(self, texts, device: torch.device, pad_to_max: bool = False):
        if isinstance(texts, str):
            texts = [texts]
        dev = torch.device(device) if isinstance(device, str) else device
        pad_mode = "max_length" if (pad_to_max or self.training) else "longest"

        if not self.include_encoder or self.model is None or self.tokenizer is None:
            B = len(texts)
            if self.tokenizer is not None:
                tokens = self.tokenizer(
                    texts,
                    padding=pad_mode,
                    pad_to_multiple_of=8 if pad_mode == "longest" else None,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = tokens["input_ids"].to(dev)
                mask = tokens["attention_mask"].to(dev)
                tok_freq = (input_ids.unsqueeze(-1).float() * torch.arange(1, self.out_dim + 1, device=dev).float() * 0.001)
                feat = torch.sin(tok_freq).to(dtype=torch.float32)
                feat = self.norm(feat)
                return feat, mask
            max_w = max((len(t.strip().split()) if (t and t.strip()) else 1) for t in texts)
            eff_len = self.max_length if (pad_to_max or self.training) else min(self.max_length, max(8, ((max_w + 7) // 8) * 8))
            feat = torch.zeros(B, eff_len, self.out_dim, device=dev)
            mask = torch.zeros(B, eff_len, dtype=torch.long, device=dev)
            d_range = torch.arange(1, self.out_dim + 1, device=dev).float()
            for i, t in enumerate(texts):
                words = t.strip().split() if (t and t.strip()) else []
                num_tok = min(len(words), eff_len)
                if num_tok > 0:
                    mask[i, :num_tok] = 1
                for w_idx, word in enumerate(words[:num_tok]):
                    h_val = sum(ord(c) for c in word) + (w_idx + 1) * 31
                    feat[i, w_idx] = torch.sin(h_val * d_range * 0.001)
            feat = self.norm(feat)
            return feat, mask

        tokens = self.tokenizer(
            texts,
            padding=pad_mode,
            pad_to_multiple_of=8 if pad_mode == "longest" else None,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        model_dev = next(self.model.parameters()).device if self.model is not None else dev
        tokens = {k: v.to(model_dev) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self.model(**tokens, output_hidden_states=True)
            num_states = len(outputs.hidden_states)
            eff_layer = min(self.target_layer, num_states - 1)
            h = outputs.hidden_states[eff_layer]
            if self.norm.weight.device != h.device:
                self.norm = self.norm.to(h.device)
            text_features = self.norm(h.to(torch.float32)).to(h.dtype)

        return text_features.to(dev), tokens["attention_mask"].to(dev)

    def encode_multimodal(
        self,
        texts: Optional[List[str]] = None,
        images: Optional[List[Union[Image.Image, str]]] = None,
        device: Optional[torch.device] = None,
        augment: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(images, (Image.Image, str)):
            images = [images]
        if texts is None and images is None:
            texts = [""]

        if device is None:
            param = next(self.parameters(), None)
            device = param.device if param is not None else (next(self.model.parameters()).device if self.model is not None else torch.device("cpu"))

        if not self.include_encoder or self.model is None:
            prompts = texts if texts is not None else ["" for _ in (images or [""])]
            return self.encode_raw(prompts, device)

        if images is None or len(images) == 0:
            if texts is None:
                texts = [""]
            return self.encode_raw(texts, device)

        processed_images = []
        for img in images:
            if img is None:
                processed_images.append(Image.new("RGB", (512, 512), (0, 0, 0)))
                continue
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            if augment:
                img = self.augment_image(img)
            processed_images.append(img)

        supports_vision = hasattr(self.model, "visual") or "CausalLM" in type(self.model).__name__ or "ConditionalGeneration" in type(self.model).__name__
        if self.processor is not None and self.model is not None and supports_vision:
            try:
                formatted_texts = []
                for i in range(len(processed_images)):
                    t = texts[i] if (texts is not None and i < len(texts)) else ""
                    is_text_only = (images is not None and i < len(images) and images[i] is None)
                    if is_text_only:
                        formatted_texts.append(t if t else "")
                    else:
                        formatted_texts.append(f"<image>\n{t}" if t else "<image>")

                inputs = self.processor(
                    text=formatted_texts,
                    images=processed_images,
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                model_dev = next(self.model.parameters()).device
                inputs = {k: v.to(model_dev) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    num_states = len(outputs.hidden_states)
                    eff_layer = min(self.target_layer, num_states - 1)
                    h = outputs.hidden_states[eff_layer]
                    if self.norm.weight.device != h.device:
                        self.norm = self.norm.to(h.device)
                    features = self.norm(h.to(torch.float32)).to(h.dtype).to(device)
                    mask = inputs.get("attention_mask", torch.ones((h.shape[0], h.shape[1]), device=device)).to(device)

                if features.shape[1] < self.max_length:
                    pad_len = self.max_length - features.shape[1]
                    pad_feats = torch.zeros((features.shape[0], pad_len, features.shape[2]), device=device, dtype=features.dtype)
                    features = torch.cat([features, pad_feats], dim=1)
                    pad_mask = torch.zeros((mask.shape[0], pad_len), device=device, dtype=mask.dtype)
                    mask = torch.cat([mask, pad_mask], dim=1)
                elif features.shape[1] > self.max_length:
                    features = features[:, :self.max_length]
                    mask = mask[:, :self.max_length]

                return features, mask
            except Exception as e:
                print(f"[Qwen35TextEncoder] Multimodal forward fallback to encode_raw ({e})")

        prompts = texts if texts is not None else ["" for _ in (images or [""])]
        return self.encode_raw(prompts, device)

    def encode_phase_layers(
        self,
        texts: Union[str, List[str]],
        device: Optional[torch.device] = None,
        layers: Optional[Tuple[int, ...]] = None,
        pad_to_max: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layers is None:
            layers = self.target_layers

        if isinstance(texts, str):
            texts = [texts]

        if device is None:
            param = next(self.parameters(), None)
            device = param.device if param is not None else (
                next(self.model.parameters()).device if self.model is not None else torch.device("cpu")
            )
        else:
            device = torch.device(device) if isinstance(device, str) else device

        pad_mode = "max_length" if (pad_to_max or self.training) else "longest"

        if not self.include_encoder or self.model is None or self.tokenizer is None:
            feat, mask = self.encode_raw(texts, device, pad_to_max=pad_to_max)
            dim = self.out_dim
            d_vec = torch.arange(dim, device=device).float()
            phase_feats = []
            for lyr in layers:
                layer_freq = (float(lyr) / 28.0) * 0.5 + 0.5
                harmonic = torch.sin(d_vec * layer_freq * 0.1).view(1, 1, dim)
                phase_feats.append(self.norm(feat + harmonic))
            phase_features = torch.stack(phase_feats, dim=1)
            return phase_features, mask

        tokens = self.tokenizer(
            texts,
            padding=pad_mode,
            pad_to_multiple_of=8 if pad_mode == "longest" else None,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        model_dev = next(self.model.parameters()).device if self.model is not None else device
        tokens = {k: v.to(model_dev) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self.model(**tokens, output_hidden_states=True)
            num_states = len(outputs.hidden_states)
            if self.norm.weight.device != model_dev:
                self.norm = self.norm.to(model_dev)

            extracted = []
            for lyr in layers:
                eff_layer = min(lyr, num_states - 1)
                h = outputs.hidden_states[eff_layer]
                h_norm = self.norm(h.to(torch.float32)).to(h.dtype)
                extracted.append(h_norm)

            phase_features = torch.stack(extracted, dim=1)

        return phase_features.to(device), tokens["attention_mask"].to(device)

    def forward_cached(
        self,
        text_features: torch.Tensor,
        mask: torch.Tensor,
        pool_layer_idx: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if text_features.dim() == 4:
            seq_for_pooling = text_features[:, pool_layer_idx]
            projected_pool = self.proj(seq_for_pooling)
            pooled_features = self.pooler(projected_pool, mask)
            return pooled_features, text_features, mask

        projected = self.proj(text_features)
        pooled_features = self.pooler(projected, mask)
        return pooled_features, projected, mask

    def forward(self, texts, device: torch.device, extract_phase_layers: bool = True, pad_to_max: bool = False):
        if extract_phase_layers and self.target_layers is not None and len(self.target_layers) > 1:
            phase_features, mask = self.encode_phase_layers(texts, device, pad_to_max=pad_to_max)
            return self.forward_cached(phase_features, mask)
        text_features, mask = self.encode_raw(texts, device, pad_to_max=pad_to_max)
        return self.forward_cached(text_features, mask)


DEFAULT_DCAE_CONFIG = {
    "_class_name": "AutoencoderDC",
    "attention_head_dim": 32,
    "decoder_act_fns": "silu",
    "decoder_block_out_channels": [128, 256, 512, 512, 1024, 1024],
    "decoder_block_types": [
        "ResBlock",
        "ResBlock",
        "ResBlock",
        "EfficientViTBlock",
        "EfficientViTBlock",
        "EfficientViTBlock",
    ],
    "decoder_layers_per_block": [3, 3, 3, 3, 3, 3],
    "decoder_norm_types": "rms_norm",
    "decoder_qkv_multiscales": [[], [], [], [5], [5], [5]],
    "downsample_block_type": "Conv",
    "encoder_block_out_channels": [128, 256, 512, 512, 1024, 1024],
    "encoder_block_types": [
        "ResBlock",
        "ResBlock",
        "ResBlock",
        "EfficientViTBlock",
        "EfficientViTBlock",
        "EfficientViTBlock",
    ],
    "encoder_layers_per_block": [2, 2, 2, 3, 3, 3],
    "encoder_qkv_multiscales": [[], [], [], [5], [5], [5]],
    "in_channels": 3,
    "latent_channels": 32,
    "scaling_factor": 0.41407,
    "upsample_block_type": "interpolate",
}


class DCAE(nn.Module):
    def __init__(
        self,
        model_name: str = "mit-han-lab/dc-ae-f32c32-sana-1.0",
        subfolder: Optional[str] = None,
        local_files_only: bool = False,
        torch_dtype: Optional[torch.dtype] = torch.float16,
        decoder_only: bool = False,
    ):
        super().__init__()

        candidates = []
        local_dcae_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "dcae"))
        if os.path.exists(os.path.join(local_dcae_dir, "diffusion_pytorch_model.safetensors")):
            candidates.append(local_dcae_dir)
        if os.path.exists("models/dcae/diffusion_pytorch_model.safetensors"):
            candidates.append(os.path.abspath("models/dcae"))

        if not model_name.endswith("-diffusers") and not os.path.exists(model_name):
            candidates.append(f"{model_name}-diffusers")
        candidates.append(model_name)

        loaded = False
        last_error = None

        offline_mode = local_files_only or (os.environ.get("DCAE_OFFLINE", "0") == "1") or (os.environ.get("HF_HUB_OFFLINE", "0") == "1")
        extra_kwargs = {}
        if torch_dtype is not None:
            extra_kwargs["torch_dtype"] = torch_dtype

        for cand in candidates:
            try:
                if subfolder:
                    self.vae = AutoencoderDC.from_pretrained(
                        cand, subfolder=subfolder, local_files_only=True, **extra_kwargs
                    )
                else:
                    self.vae = AutoencoderDC.from_pretrained(
                        cand, local_files_only=True, **extra_kwargs
                    )
                loaded = True
                break
            except Exception as e:
                last_error = e

        if not loaded and not offline_mode:
            for cand in candidates:
                try:
                    if subfolder:
                        self.vae = AutoencoderDC.from_pretrained(
                            cand, subfolder=subfolder, **extra_kwargs
                        )
                    else:
                        self.vae = AutoencoderDC.from_pretrained(cand, **extra_kwargs)
                    loaded = True
                    break
                except Exception as e:
                    last_error = e
                    if not subfolder:
                        try:
                            self.vae = AutoencoderDC.from_pretrained(
                                cand, subfolder="vae", **extra_kwargs
                            )
                            loaded = True
                            break
                        except Exception as e_sub:
                            last_error = e_sub

        if not loaded:
            raise RuntimeError(
                f"[DCAE] CRITICAL: Failed to load pretrained DCAE weights from candidates: {candidates}. "
                f"Ensure models/dcae/diffusion_pytorch_model.safetensors exists or HF network is accessible. "
                f"Last error: {last_error}"
            )

        self.decoder_only = decoder_only
        if decoder_only and hasattr(self.vae, "encoder"):
            self.vae.encoder = None

        for param in self.vae.parameters():
            param.requires_grad = False

        try:
            self.vae.to(memory_format=torch.channels_last)
        except Exception:
            pass

        self.scaling_factor = getattr(self.vae.config, "scaling_factor", 0.41407)
        self.shift_factor = getattr(self.vae.config, "shift_factor", 0.0)
        self.latent_channels = getattr(self.vae.config, "latent_channels", 32)
        self.spatial_scale_factor = getattr(self.vae, "spatial_compression_ratio", 32)
        self.scale_factor = self.scaling_factor
        self.use_tiling = False
        self._compiled = False

    def compile(self, mode: str = "default"):
        if hasattr(self.vae, "decoder") and self.vae.decoder is not None:
            try:
                self.vae.decoder = torch.compile(self.vae.decoder, mode=mode)
                self._compiled = True
                print("[DCAE] Decoder successfully compiled with torch.compile.")
            except Exception as e:
                print(f"[DCAE] Note: torch.compile not applied ({e}), proceeding in eager mode.")
        return self

    def enable_tiling(self, *args, **kwargs):
        self.use_tiling = True
        if hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling(*args, **kwargs)

    def disable_tiling(self):
        self.use_tiling = False
        if hasattr(self.vae, "disable_tiling"):
            self.vae.disable_tiling()

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "decoder_only", False) or getattr(self.vae, "encoder", None) is None:
            raise RuntimeError("DCAE was initialized with decoder_only=True and has no encoder.")
        vae_param = next(self.vae.parameters(), None)
        if vae_param is not None:
            if x.device != vae_param.device:
                x = x.to(vae_param.device)
            if x.dtype != vae_param.dtype:
                x = x.to(vae_param.dtype)
        x = x * 2.0 - 1.0
        out = self.vae.encode(x)
        latent = out.latent if hasattr(out, "latent") else (out[0] if isinstance(out, (tuple, list)) else out)
        if self.shift_factor != 0.0:
            latent = latent - self.shift_factor
        return latent * self.scaling_factor

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor, _from_tiler: bool = False) -> torch.Tensor:
        if getattr(self, "use_tiling", False) and not _from_tiler:
            return self.tiled_decode(latent)

        vae_param = next(self.vae.parameters(), None)
        target_device = vae_param.device if vae_param is not None else latent.device
        target_dtype = vae_param.dtype if vae_param is not None else latent.dtype

        if latent.device != target_device or latent.dtype != target_dtype:
            latent = latent.to(device=target_device, dtype=target_dtype)

        b, c, h, w = latent.shape
        if c < self.latent_channels:
            if self.latent_channels % c == 0:
                reps = self.latent_channels // c
                latent = latent.repeat(1, reps, 1, 1)
            else:
                pad = torch.zeros(
                    b, self.latent_channels - c, h, w,
                    device=latent.device, dtype=latent.dtype
                )
                latent = torch.cat([latent, pad], dim=1)
        elif c > self.latent_channels:
            latent = latent[:, :self.latent_channels]

        inv_scale = 1.0 / self.scaling_factor
        unscaled_latent = (latent * inv_scale) if self.shift_factor == 0.0 else (latent * inv_scale + self.shift_factor)
        unscaled_latent = unscaled_latent.contiguous(memory_format=torch.channels_last)

        if hasattr(self.vae, "decoder") and self.vae.decoder is not None:
            out = self.vae.decoder(unscaled_latent)
        else:
            out = self.vae.decode(unscaled_latent)

        image = out.sample if hasattr(out, "sample") else (out[0] if isinstance(out, (tuple, list)) else out)
        return (image.mul(0.5).add_(0.5)).clamp_(0.0, 1.0)

    @torch.inference_mode()
    def tiled_decode(
        self,
        latent: torch.Tensor,
        tile_size: int = 48,
        tile_overlap: int = 12,
        batch_chunk_size: int = 1,
    ) -> torch.Tensor:
        return tiled_decode_latents(
            self,
            latent,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            batch_chunk_size=batch_chunk_size,
        )


TAESD = DCAE
SDVAE = DCAE
SDXLVAE = DCAE
FluxVAE = DCAE


def tiled_decode_latents(
    vae: nn.Module,
    latent: torch.Tensor,
    tile_size: int = 64,
    tile_overlap: int = 16,
    batch_chunk_size: int = 1,
) -> torch.Tensor:
    if not isinstance(latent, torch.Tensor) or latent.dim() != 4:
        raise ValueError(f"Expected 4D latent tensor [B, C, H, W], got {type(latent)}")

    B, C, H, W = latent.shape

    if batch_chunk_size > 0 and B > batch_chunk_size:
        chunks = []
        for b_start in range(0, B, batch_chunk_size):
            b_chunk = latent[b_start : b_start + batch_chunk_size]
            chunks.append(
                tiled_decode_latents(
                    vae=vae,
                    latent=b_chunk,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    batch_chunk_size=batch_chunk_size,
                )
            )
        return torch.cat(chunks, dim=0)

    spatial_factor = getattr(vae, "spatial_scale_factor", 32)

    min_tile = 32 if spatial_factor == 32 else 16
    min_overlap = 8 if spatial_factor == 32 else 4
    tile_size = max(min_tile, int(tile_size))
    tile_overlap = max(min_overlap, min(int(tile_overlap), tile_size // 2))

    if H <= tile_size and W <= tile_size:
        if hasattr(vae, "decode"):
            try:
                return vae.decode(latent, _from_tiler=True)
            except TypeError:
                return vae.decode(latent)
        return vae(latent)
    if spatial_factor is None:
        with torch.no_grad():
            probe_h = min(H, 16)
            probe_w = min(W, 16)
            if hasattr(vae, "decode"):
                try:
                    probe_out = vae.decode(latent[:, :, :probe_h, :probe_w], _from_tiler=True)
                except TypeError:
                    probe_out = vae.decode(latent[:, :, :probe_h, :probe_w])
            else:
                probe_out = vae(latent[:, :, :probe_h, :probe_w])
            spatial_factor = max(1, probe_out.shape[-1] // probe_w)

    stride_h = tile_size - tile_overlap
    stride_w = tile_size - tile_overlap

    y_starts = list(range(0, max(1, H - tile_size + 1), stride_h))
    if not y_starts or y_starts[-1] + tile_size < H:
        y_starts.append(max(0, H - tile_size))
    y_starts = sorted(list(set(y_starts)))

    x_starts = list(range(0, max(1, W - tile_size + 1), stride_w))
    if not x_starts or x_starts[-1] + tile_size < W:
        x_starts.append(max(0, W - tile_size))
    x_starts = sorted(list(set(x_starts)))

    H_out = H * spatial_factor
    W_out = W * spatial_factor

    dev = latent.device
    out_dtype = torch.float32
    out_img = torch.zeros((B, 3, H_out, W_out), dtype=out_dtype, device=dev)
    weight_map = torch.zeros((1, 1, H_out, W_out), dtype=out_dtype, device=dev)

    for yi, y in enumerate(y_starts):
        y_end = min(y + tile_size, H)
        out_y = y * spatial_factor
        out_y_end = y_end * spatial_factor
        img_th = out_y_end - out_y

        for xi, x in enumerate(x_starts):
            x_end = min(x + tile_size, W)
            out_x = x * spatial_factor
            out_x_end = x_end * spatial_factor
            img_tw = out_x_end - out_x

            tile_latent = latent[:, :, y:y_end, x:x_end]
            if hasattr(vae, "decode"):
                try:
                    img_tile = vae.decode(tile_latent, _from_tiler=True).to(device=dev, dtype=out_dtype)
                except TypeError:
                    img_tile = vae.decode(tile_latent).to(device=dev, dtype=out_dtype)
            else:
                img_tile = vae(tile_latent).to(device=dev, dtype=out_dtype)

            if img_tile.shape[-2:] != (img_th, img_tw):
                img_tile = F.interpolate(img_tile, size=(img_th, img_tw), mode="bilinear", align_corners=False)

            m_y = torch.ones((img_th,), dtype=out_dtype, device=dev)
            if yi > 0:
                ov_top = (y_starts[yi - 1] + tile_size) - y
                ov_top_px = min(ov_top * spatial_factor, img_th)
                if ov_top_px > 0:
                    ramp = torch.sin(0.5 * torch.pi * torch.linspace(0.0, 1.0, ov_top_px, device=dev)) ** 2
                    m_y[:ov_top_px] *= ramp
            if yi < len(y_starts) - 1:
                ov_bottom = (y + tile_size) - y_starts[yi + 1]
                ov_bottom_px = min(ov_bottom * spatial_factor, img_th)
                if ov_bottom_px > 0:
                    ramp = torch.cos(0.5 * torch.pi * torch.linspace(0.0, 1.0, ov_bottom_px, device=dev)) ** 2
                    m_y[-ov_bottom_px:] *= ramp

            m_x = torch.ones((img_tw,), dtype=out_dtype, device=dev)
            if xi > 0:
                ov_left = (x_starts[xi - 1] + tile_size) - x
                ov_left_px = min(ov_left * spatial_factor, img_tw)
                if ov_left_px > 0:
                    ramp = torch.sin(0.5 * torch.pi * torch.linspace(0.0, 1.0, ov_left_px, device=dev)) ** 2
                    m_x[:ov_left_px] *= ramp
            if xi < len(x_starts) - 1:
                ov_right = (x + tile_size) - x_starts[xi + 1]
                ov_right_px = min(ov_right * spatial_factor, img_tw)
                if ov_right_px > 0:
                    ramp = torch.cos(0.5 * torch.pi * torch.linspace(0.0, 1.0, ov_right_px, device=dev)) ** 2
                    m_x[-ov_right_px:] *= ramp

            w_tile = (m_y.unsqueeze(1) * m_x.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

            out_img[:, :, out_y:out_y_end, out_x:out_x_end] += img_tile * w_tile
            weight_map[:, :, out_y:out_y_end, out_x:out_x_end] += w_tile

    out_img = out_img / weight_map.clamp(min=1e-6)
    return out_img.clamp(0.0, 1.0)
