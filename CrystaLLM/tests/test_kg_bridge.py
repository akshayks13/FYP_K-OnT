import unittest

import torch

from crystallm import GPT, GPTConfig


def _tiny_cfg(**kwargs):
    base = dict(
        block_size=32,
        vocab_size=40,
        n_layer=4,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        kg_dropout=0.0,
    )
    base.update(kwargs)
    return GPTConfig(**base)


def _copy_shared_weights(src: GPT, dst: GPT) -> None:
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    for k, v in src_sd.items():
        if k in dst_sd and dst_sd[k].shape == v.shape:
            dst_sd[k] = v.clone()
    dst.load_state_dict(dst_sd)


class TestKgBridge(unittest.TestCase):

    def test_all_layers_have_cross_attn(self):
        model = GPT(_tiny_cfg(use_kg_cross_attn=True, kg_dim=4, n_layer=4))
        self.assertTrue(all(b.use_kg_cross_attn for b in model.transformer.h))
        self.assertIsNotNone(model.kg_proj)
        self.assertIsNotNone(model.kg_null)

    def test_vanilla_has_no_cross_attn(self):
        model = GPT(_tiny_cfg())
        self.assertFalse(any(getattr(b, "use_kg_cross_attn", False) for b in model.transformer.h))
        self.assertIsNone(model.kg_proj)

    def test_none_kg_skips_ca(self):
        torch.manual_seed(2)
        vanilla = GPT(_tiny_cfg())
        kg = GPT(_tiny_cfg(use_kg_cross_attn=True, kg_dim=8))
        _copy_shared_weights(vanilla, kg)

        vanilla.eval()
        kg.eval()
        idx = torch.randint(0, 40, (1, 6))
        logits_a, _ = vanilla(idx)
        logits_b, _ = kg(idx, kg_context=None)
        self.assertTrue(torch.allclose(logits_a, logits_b, atol=1e-5, rtol=1e-5))

    def test_kg_context_changes_logits(self):
        torch.manual_seed(1)
        vanilla = GPT(_tiny_cfg())
        kg = GPT(_tiny_cfg(use_kg_cross_attn=True, kg_dim=8))
        _copy_shared_weights(vanilla, kg)

        vanilla.eval()
        kg.eval()
        idx = torch.randint(0, 40, (2, 8))
        kg_raw = torch.randn(2, 3, 8)
        logits_a, _ = vanilla(idx)
        logits_b, _ = kg(idx, kg_context=kg_raw)
        self.assertFalse(torch.allclose(logits_a, logits_b, atol=1e-4))

    def test_configure_optimizers_covers_all_params(self):
        model = GPT(_tiny_cfg(use_kg_cross_attn=True, kg_dim=4))
        opt = model.configure_optimizers(0.1, 1e-4, (0.9, 0.95))
        n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
        n_model = sum(p.numel() for p in model.parameters())
        self.assertEqual(n_opt, n_model)

    def test_prefix_none_kg_matches_vanilla(self):
        torch.manual_seed(0)
        vanilla = GPT(_tiny_cfg())
        kg = GPT(_tiny_cfg(use_kg_prefix=True, kg_dim=8))
        _copy_shared_weights(vanilla, kg)
        torch.nn.init.normal_(kg.kg_proj.weight, mean=0.0, std=0.1)
        vanilla.eval()
        kg.eval()
        idx = torch.randint(0, 40, (2, 8))
        logits_a, _ = vanilla(idx)
        logits_b, _ = kg(idx, kg_context=None)
        self.assertTrue(torch.allclose(logits_a, logits_b, atol=1e-5, rtol=1e-5))

    def test_prefix_optimizer_covers_kg_null(self):
        model = GPT(_tiny_cfg(use_kg_prefix=True, kg_dim=4))
        opt = model.configure_optimizers(0.1, 1e-4, (0.9, 0.95))
        n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
        n_model = sum(p.numel() for p in model.parameters())
        self.assertEqual(n_opt, n_model)
        self.assertTrue(any("kg_null" in n for n, _ in model.named_parameters()))

    def test_prefix_zero_init_matches_vanilla(self):
        torch.manual_seed(0)
        vanilla = GPT(_tiny_cfg())
        kg = GPT(_tiny_cfg(use_kg_prefix=True, kg_dim=8))
        _copy_shared_weights(vanilla, kg)
        vanilla.eval()
        kg.eval()
        idx = torch.randint(0, 40, (2, 8))
        kg_raw = torch.randn(2, 3, 8)
        logits_a, _ = vanilla(idx)
        logits_b, _ = kg(idx, kg_context=kg_raw)
        self.assertTrue(torch.allclose(logits_a, logits_b, atol=1e-5, rtol=1e-5))

    def test_prefix_trained_proj_changes_logits(self):
        torch.manual_seed(1)
        vanilla = GPT(_tiny_cfg())
        kg = GPT(_tiny_cfg(use_kg_prefix=True, kg_dim=8))
        _copy_shared_weights(vanilla, kg)
        torch.nn.init.normal_(kg.kg_proj.weight, mean=0.0, std=0.1)
        vanilla.eval()
        kg.eval()
        idx = torch.randint(0, 40, (2, 8))
        kg_raw = torch.randn(2, 3, 8)
        logits_a, _ = vanilla(idx)
        logits_b, _ = kg(idx, kg_context=kg_raw)
        self.assertFalse(torch.allclose(logits_a, logits_b, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
