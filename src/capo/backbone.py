import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from capo.config import ROOT, local_path
from capo.lora import inject_expert_lora, merge_expert_lora


def use_vendored_openpi():
    paths = [
        ROOT / "third_party/openpi/src",
        ROOT / "third_party/openpi/packages/openpi-client/src",
    ]
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import openpi

    if not Path(openpi.__file__).resolve().is_relative_to(ROOT):
        raise RuntimeError("An external openpi was imported first. Start a clean Python process.")
    import transformers

    if transformers.__version__ != "4.53.2":
        raise RuntimeError("Install transformers==4.53.2 and run scripts/setup_transformers.py")
    patches = ROOT / "third_party/openpi/src/openpi/models_pytorch/transformers_replace"
    installed = Path(transformers.__file__).parent
    for patch in patches.rglob("*.py"):
        target = installed / patch.relative_to(patches)
        if not target.is_file() or target.read_bytes() != patch.read_bytes():
            raise RuntimeError("OpenPI patches are missing; run scripts/setup_transformers.py")


class Pi05(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        use_vendored_openpi()
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

        self.cfg = cfg
        model_cfg = Pi0Config(
            pi05=True,
            action_dim=cfg.action_dim,
            action_horizon=cfg.action_horizon,
            max_token_len=cfg.max_token_len,
            dtype=cfg.precision,
            image_keys=tuple(cfg.image_keys),
            pytorch_compile_mode=None,
        )

        with torch.device("meta"):
            self.model = PI0Pytorch(model_cfg)
        from safetensors import safe_open
        from capo.identity import file_sha256

        path = local_path(cfg.checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Copy a converted pi_05 SFT model to {path}")
        digest = file_sha256(path)
        if cfg.base_sha256 and cfg.base_sha256 != digest:
            raise ValueError("Base checkpoint content differs from configured SHA256")
        cfg.base_sha256 = digest
        expected = self.model.state_dict()
        aliases = {}
        for name, parameter in self.model.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(parameter), []).append(name)
        with safe_open(str(path), framework="pt", device="cpu") as source:
            keys = set(source.keys())
            if keys - expected.keys():
                raise ValueError(f"Unexpected base weights: {keys - expected.keys()}")
            mapping = {name: name for name in keys}
            for group in aliases.values():
                saved = next((name for name in group if name in keys), None)
                if saved is not None:
                    for name in group:
                        mapping.setdefault(name, saved)
            if expected.keys() - mapping.keys():
                raise ValueError(f"Missing base weights: {expected.keys() - mapping.keys()}")
            tensors = {}
            cache = {}
            for name, target in expected.items():
                key = (mapping[name], target.dtype)
                if key not in cache:
                    cache[key] = source.get_tensor(mapping[name]).to(target.dtype)
                tensors[name] = cache[key]
            self.model.load_state_dict(tensors, strict=True, assign=True)

        for group in aliases.values():
            first = self.model.get_parameter(group[0])
            for name in group[1:]:
                parent, leaf = name.rsplit(".", 1)
                setattr(self.model.get_submodule(parent), leaf, first)
        for module in self.model.modules():
            for name, buffer in list(module.named_buffers(recurse=False)):
                if not buffer.is_meta:
                    continue
                if name == "position_ids":
                    module.register_buffer(
                        name, torch.arange(buffer.shape[-1]).expand(buffer.shape), persistent=False
                    )
                elif name == "inv_freq" and hasattr(module, "rope_init_fn"):
                    value, module.attention_scaling = module.rope_init_fn(
                        module.config, torch.device("cpu")
                    )
                    module.register_buffer(name, value, persistent=False)
                    module.original_inv_freq = value
                else:
                    raise ValueError(
                        f"Unsupported nonpersistent meta buffer: {type(module).__name__}.{name}"
                    )
        del expected, tensors, cache
        self.model.paligemma_with_expert.paligemma.requires_grad_(False)
        if cfg.actor_tuning == "lora":
            inject_expert_lora(self.model.paligemma_with_expert.gemma_expert, cfg.lora_rank, cfg.lora_alpha)
        else:
            self.model.paligemma_with_expert.gemma_expert.float()
        for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
            getattr(self.model, name).float().requires_grad_(cfg.train_action_time_projections)
        self.model.eval()

    def merge_lora_for_export(self):
        merge_expert_lora(self.model.paligemma_with_expert.gemma_expert)

    @torch.no_grad()
    def prepare(self, obs):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch

        images = {
            k: (v.float() / 127.5 - 1 if v.dtype == torch.uint8 else v.float())
            for k, v in obs["images"].items()
        }
        images = {k: v.permute(0, 3, 1, 2) if v.shape[-1] == 3 else v for k, v in images.items()}
        observation = SimpleNamespace(
            **{"token_ar_mask": None, "token_loss_mask": None, **obs, "images": images}
        )
        observation = preprocess_observation_pytorch(observation, train=False, image_keys=self.cfg.image_keys)
        prefix, mask, ar = self.model.embed_prefix(
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
        )
        vlm = self.model.paligemma_with_expert.paligemma
        prefix = prefix.to(vlm.language_model.layers[0].self_attn.q_proj.weight.dtype)
        vlm.language_model.config._attn_implementation = "eager"
        _, kv = self.model.paligemma_with_expert(
            attention_mask=self.model._prepare_attention_masks_4d(make_att_2d_masks(mask, ar)),
            position_ids=mask.long().cumsum(1) - 1,
            inputs_embeds=[prefix, None],
            use_cache=True,
        )

        from transformers.cache_utils import DynamicCache

        kv = DynamicCache.from_legacy_cache(tuple(tuple(t.detach() for t in layer) for layer in kv))
        return (observation.state.detach(), mask.detach(), kv)

    def velocity(self, context, x, tau):
        state, mask, kv = context

        def evaluate(x, tau):
            return -self.model.denoise_step(state, mask, kv, x, 1 - tau)

        if self.cfg.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(evaluate, x, tau, use_reentrant=False)
        return evaluate(x, tau)

    def reference(self):
        ref = copy.copy(self)
        ref._modules = self._modules.copy()
        ref.model = copy.copy(self.model)
        ref.model._modules = self.model._modules.copy()
        combined = copy.copy(self.model.paligemma_with_expert)
        combined._modules = self.model.paligemma_with_expert._modules.copy()
        combined.gemma_expert = copy.deepcopy(combined.gemma_expert)
        ref.model.paligemma_with_expert = combined
        for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
            setattr(ref.model, name, copy.deepcopy(getattr(self.model, name)))
        ref.requires_grad_(False)
        return ref

    def trainable_state(self):
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {n: t.detach().cpu().clone() for n, t in self.state_dict().items() if n in names}


def make_backbone(cfg, **kwargs):
    if cfg.backend != "pi05":
        raise ValueError("backend must be pi05")
    return Pi05(cfg, **kwargs).to(cfg.device)
