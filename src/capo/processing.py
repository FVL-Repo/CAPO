import json

import numpy as np
import torch

from capo.config import local_path


class Processor:
    def __init__(self, cfg, norm_stats, tokenizer="assets/paligemma_tokenizer.model", *, quantile=True):
        import sentencepiece

        self.cfg, self.quantile = cfg, quantile
        stats = json.loads(local_path(norm_stats).read_text())
        self.stats = stats.get("norm_stats", stats)
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(local_path(tokenizer)))
        from capo.identity import file_sha256

        spec = dict(cfg.data_spec or {})
        spec.update(
            norm_stats_sha256=file_sha256(local_path(norm_stats)),
            normalization_mode="quantile" if quantile else "mean_std",
            tokenizer_sha256=file_sha256(local_path(tokenizer)),
            image_keys=list(cfg.image_keys),
            max_token_len=cfg.max_token_len,
            state_dim=cfg.state_dim,
            robot_action_dim=cfg.robot_action_dim,
            action_semantics=cfg.action_semantics,
            inactive_action_dims=list(cfg.inactive_action_dims),
            preprocessing="rgb_letterbox224_bilinear_v1_pi05_state_tokens",
        )

        for key, value in (cfg.data_spec or {}).items():
            if key in spec and spec[key] != value:
                raise ValueError(f"Processor data specification mismatch: {key}")
        cfg.data_spec = spec
        self.spec = spec

    def normalize(self, values, key, *, inverse=False):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim < 1 or not np.isfinite(values).all():
            raise ValueError(f"{key} must be finite and have a coordinate dimension")
        stat = self.stats[key]
        names = ("q01", "q99") if self.quantile else ("mean", "std")
        arrays = [np.asarray(stat[name], np.float32) for name in names]
        if any(a.ndim != 1 or not np.isfinite(a).all() for a in arrays):
            raise ValueError(f"Invalid normalization statistics for {key}")
        if arrays[0].shape != arrays[1].shape or len(arrays[0]) < values.shape[-1]:
            raise ValueError(f"Normalization statistics do not cover all {key} coordinates")
        if (self.quantile and np.any(arrays[1] < arrays[0])) or (not self.quantile and np.any(arrays[1] < 0)):
            raise ValueError(f"Invalid normalization range for {key}")
        if self.quantile:
            low, high = np.asarray(stat["q01"], np.float32), np.asarray(stat["q99"], np.float32)
            size = min(values.shape[-1], len(low))
            x = values[..., :size]
            result = (
                (x + 1) / 2 * (high[:size] - low[:size] + 1e-6) + low[:size]
                if inverse
                else ((x - low[:size]) / (high[:size] - low[:size] + 1e-6) * 2 - 1)
            )
        else:
            mean, std = np.asarray(stat["mean"], np.float32), np.asarray(stat["std"], np.float32)
            size = min(values.shape[-1], len(mean))
            x = values[..., :size]
            result = (
                x * (std[:size] + 1e-6) + mean[:size] if inverse else (x - mean[:size]) / (std[:size] + 1e-6)
            )
        if size < values.shape[-1]:
            result = np.concatenate([result, values[..., size:]], -1)
        return result.astype(np.float32)

    def observation(self, raw):
        for key in ("robot_config_sha256", "state_format", "action_format"):
            if key in self.spec and raw.get("robot_metadata", {}).get(key) != self.spec[key]:
                raise ValueError(f"Observation/data specification mismatch: {key}")
        if "task" in self.spec and str(raw["prompt"]) != self.spec["task"]:
            raise ValueError("Critic is single-task: observation prompt differs from data specification")
        state = self.normalize(raw["state"], "state")
        if len(state) > self.cfg.state_dim:
            raise ValueError("State exceeds configured dimension")
        prompt = str(raw["prompt"]).strip().replace("_", " ").replace("\n", " ")

        bins = np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1
        text = f"Task: {prompt}, State: {' '.join(map(str, bins))};\nAction: "
        tokens = self.tokenizer.encode(text, add_bos=True)
        if len(tokens) > self.cfg.max_token_len:
            raise ValueError("Prompt exceeds max_token_len; increase config or shorten the prompt")
        valid = len(tokens)
        tokens += [0] * (self.cfg.max_token_len - valid)
        state = np.pad(state, (0, self.cfg.state_dim - len(state)))
        images, masks = {}, {}
        for key in self.cfg.image_keys:
            image = raw["images"].get(key)
            masks[key] = torch.tensor(image is not None)
            if image is None:
                images[key] = torch.zeros(224, 224, 3, dtype=torch.uint8)
            else:
                from PIL import Image

                image = np.asarray(image)
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError("Raw camera images must be RGB HWC uint8")
                height, width = image.shape[:2]
                ratio = max(width / 224, height / 224)
                new_height, new_width = int(height / ratio), int(width / ratio)
                resized = Image.fromarray(image).resize((new_width, new_height), Image.Resampling.BILINEAR)
                padded = Image.new("RGB", (224, 224), 0)
                padded.paste(resized, ((224 - new_width) // 2, (224 - new_height) // 2))
                images[key] = torch.from_numpy(np.array(padded, copy=True))
        return {
            "state": torch.from_numpy(state),
            "images": images,
            "image_masks": masks,
            "tokenized_prompt": torch.tensor(tokens, dtype=torch.long),
            "tokenized_prompt_mask": torch.arange(self.cfg.max_token_len) < valid,
        }

    def actions(self, actions):
        actions = np.asarray(actions, np.float32)
        if (
            actions.ndim != 2
            or actions.shape[1] != self.cfg.robot_action_dim
            or len(actions) > self.cfg.action_horizon
        ):
            raise ValueError("Expected actions [executed_steps, robot_action_dim]")
        padded = np.zeros((self.cfg.action_horizon, self.cfg.action_dim), np.float32)
        padded[: len(actions), : actions.shape[1]] = self.normalize(actions, "actions")
        return torch.from_numpy(padded)

    def unnormalize_actions(self, actions):
        return self.normalize(np.asarray(actions)[..., : self.cfg.robot_action_dim], "actions", inverse=True)
