"""
Adapted from:
https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/model.py
"""
import math
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from crystallm import CIFTokenizer


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 371
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    # KG (optional; off by default for vanilla CrystaLLM)
    use_kg_cross_attn: bool = False
    use_kg_prefix: bool = False  # soft prefix: add projected HGT to the first N token embeddings
    kg_dim: int = 0  # HGT embedding dim before projection (e.g. 32 or 64)
    kg_dropout: float = 0.1  # dropout on the KG path; 0 at eval


class LayerNorm(nn.Module):

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


def gelu(x: Tensor) -> Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention: CIF hidden states (Query) attend to KG vectors (Key/Value).

    Inputs
    ------
    x  : [B, T, n_embd]    CIF token hidden states (after self-attention)
    kg : [B, N_kg, n_embd] projected HGT node vectors

    Output
    ------
    [B, T, n_embd]  KG-conditioned update (same shape as x)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        dropout = float(getattr(config, "kg_dropout", config.dropout) or 0.0)
        self.dropout = dropout
        self.wq = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.wk = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.wv = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, kg: Tensor) -> Tensor:
        B, T, C = x.size()
        _, N, _ = kg.size()

        q = self.wq(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.wk(kg).view(B, N, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.wv(kg).view(B, N, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.use_kg_cross_attn = bool(config.use_kg_cross_attn)
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        if self.use_kg_cross_attn:
            self.ln_x = LayerNorm(config.n_embd, bias=config.bias)
            self.cross_attn = CrossAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: Tensor, kg: Tensor | None = None) -> Tensor:
        """
        Self-attention → (optional) ungated cross-attention to KG → MLP.

        :param x: CIF hidden states [B, T, n_embd]
        :param kg: projected KG context [B, N_kg, n_embd], or None to skip cross-attn
        """
        x = x + self.attn(self.ln_1(x))
        if self.use_kg_cross_attn and kg is not None:
            x = x + self.cross_attn(self.ln_x(x), kg)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.use_kg_cross_attn = bool(config.use_kg_cross_attn)
        self.use_kg_prefix = bool(getattr(config, "use_kg_prefix", False))
        if self.use_kg_cross_attn and self.use_kg_prefix:
            raise ValueError("use_kg_cross_attn and use_kg_prefix cannot both be True")

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.kg_proj = None
        self.kg_null = None
        if self.use_kg_cross_attn or self.use_kg_prefix:
            assert config.kg_dim > 0, "kg_dim must be > 0 when KG injection is on"
            self.kg_proj = nn.Linear(config.kg_dim, config.n_embd, bias=config.bias)
            self.kg_null = nn.Parameter(torch.zeros(1, 1, config.kg_dim))

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Zero kg_proj so step 0 equals vanilla (additive prefix of zeros).
        if self.use_kg_prefix and self.kg_proj is not None:
            nn.init.zeros_(self.kg_proj.weight)
            if self.kg_proj.bias is not None:
                nn.init.zeros_(self.kg_proj.bias)

        if self.use_kg_cross_attn:
            print(f"KG cross-attn: ungated, all {config.n_layer} layers (kg_dim={config.kg_dim})")
        elif self.use_kg_prefix:
            print(f"KG soft prefix: add projected HGT to first N token embeddings (kg_dim={config.kg_dim})")

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def project_kg(self, kg_raw: Tensor | None, kg_types: Tensor | None = None) -> Tensor | None:
        """Map raw HGT vectors [B, N_kg, kg_dim] → [B, N_kg, n_embd]."""
        if kg_raw is None or self.kg_proj is None:
            return None
        if not (self.use_kg_cross_attn or self.use_kg_prefix):
            return None
        return self.kg_proj(kg_raw)

    def forward(self, idx, targets=None, kg_context=None, kg_types=None, kg_mask=None,
                apply_kg_prefix=True):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        kg = self.project_kg(kg_context, kg_types=kg_types)
        if kg is not None and kg.size(0) == 1 and b > 1:
            kg = kg.expand(b, -1, -1)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        # Soft prefix: mix HGT into the first N CIF embeddings. Sequence length
        # and positions stay vanilla. Zero-init kg_proj ⇒ x unchanged at step 0.
        if self.use_kg_prefix and kg is not None and apply_kg_prefix:
            n = min(kg.size(1), t)
            x = x.clone()
            x[:, :n, :] = x[:, :n, :] + kg[:, :n, :]
        x = self.transformer.drop(x)

        ca_kg = kg if self.use_kg_cross_attn else None
        for block in self.transformer.h:
            x = block(x, kg=ca_kg)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size: int):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pn in ("kg_null", "kg_prefix_scale"):
                    no_decay.add(fpn)

        decay.remove("lm_head.weight")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters in both sets: {inter_params}"
        assert len(param_dict.keys() - union_params) == 0, (
            f"parameters not in either set: {param_dict.keys() - union_params}"
        )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 kg_context=None, kg_types=None, kg_mask=None):
        """
        Generate up to max_new_tokens tokens. Stops early on a double-newline.
        kg_context is reused at every generation step.
        """
        tokenizer = CIFTokenizer()
        newline_id = tokenizer.token_to_id["\n"]
        prev_id = None
        for _ in range(max_new_tokens):
            idx_from_start = idx.size(1) <= self.config.block_size
            idx_cond = idx if idx_from_start else idx[:, -self.config.block_size:]
            logits, _ = self(
                idx_cond, kg_context=kg_context, kg_types=kg_types, kg_mask=kg_mask,
                apply_kg_prefix=idx_from_start,
            )
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if prev_id is not None and prev_id == newline_id and idx_next.item() == newline_id:
                break
            prev_id = idx_next.item()
        return idx
