import time
import uuid

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from capo.config import local_path, load_mapping
from capo.resnet import ResNet10


def load_reward_config(path):
    cfg = load_mapping(path)
    allowed = {
        "enabled",
        "interactive",
        "init_checkpoint",
        "dataset",
        "validation",
        "output",
        "checkpoint",
        "image_keys",
        "device",
        "encoder",
        "hidden_dim",
        "dropout",
        "batch_size",
        "steps",
        "learning_rate",
        "threshold",
        "success_reward",
        "step_reward",
        "seed",
    }
    if set(cfg) - allowed:
        raise ValueError(f"Unknown reward fields: {set(cfg) - allowed}")
    if not 0 < cfg.get("threshold", 0.9) < 1:
        raise ValueError("Reward threshold must be between zero and one")
    if cfg.get("step_reward", -0.05) != -0.05 or cfg.get("success_reward", 10.0) != 10.0:
        raise ValueError("Online rewards are fixed at step=-0.05, success=10.0")
    return cfg


class SuccessClassifier(nn.Module):
    architecture = "resnet10_success_224_v1"

    def __init__(self, cfg):
        super().__init__()
        self.keys = cfg["image_keys"]
        if not self.keys or len(set(self.keys)) != len(self.keys):
            raise ValueError("Specify unique reward camera keys")
        self.encoder = ResNet10().load_pretrained(cfg["encoder"])
        self.head = nn.Sequential(
            nn.Linear(512 * len(self.keys), cfg["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["hidden_dim"], 2),
        )

    def forward(self, images):
        features = []
        for key in self.keys:
            image = images[key]
            if image.dtype != torch.uint8 or image.ndim != 4 or image.shape[-1] != 3:
                raise ValueError("Reward expects batched RGB HWC uint8")
            x = image.permute(0, 3, 1, 2).float() / 255
            x = F.interpolate(x, (224, 224), mode="bilinear", align_corners=False, antialias=True)
            mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            with torch.no_grad():
                features.append(self.encoder((x - mean) / std).mean((-2, -1)))
        return self.head(torch.cat(features, -1))


class RewardAdapter:
    """Reward decisions are provisional until the operator resolves success."""

    def __init__(self, robot, cfg):
        self.robot, self.cfg = robot, cfg
        self.interactive = cfg.get("interactive", True)
        self.model = None
        self.last_inference = None
        if cfg.get("enabled", False):
            self.device = cfg["device"]
            payload = torch.load(local_path(cfg["checkpoint"]), map_location="cpu", weights_only=True)
            if payload.get("architecture") != SuccessClassifier.architecture:
                raise ValueError("Reward checkpoint requires training with 224x224 inputs")
            for key in ("image_keys", "encoder", "hidden_dim", "dropout"):
                if cfg[key] != payload["config"][key]:
                    raise ValueError(f"Reward checkpoint/config mismatch: {key}")
            if "dataset" in payload["config"] and local_path(cfg["dataset"]) != local_path(
                payload["config"]["dataset"]
            ):
                raise ValueError("Hard examples must be appended to the classifier training dataset")
            self.model = SuccessClassifier(cfg).to(self.device)
            self.model.load_state_dict(payload["model"], strict=True)
            self.model.eval()
        if self.interactive:
            if not hasattr(robot, "configure_review"):
                raise ValueError("Interactive reward needs robot pause/reset/confirmation support")
            robot.configure_review()

    def __getattr__(self, name):
        return getattr(self.robot, name)

    @property
    def demonstration(self):
        return self.robot.demonstration

    @demonstration.setter
    def demonstration(self, value):
        self.robot.demonstration = value

    def reset(self):
        self.last_inference = None
        return self.robot.reset()

    def probability(self, observation):
        images = {
            k: torch.as_tensor(observation["images"][k], device=self.device)[None] for k in self.model.keys
        }
        with torch.inference_mode():
            return self.model(images).softmax(-1)[0, 1].item()

    def execute(self, selected):
        result = self.robot.execute(selected)
        if result.get("quarantined") or result.get("retry") or result.get("safety_abort"):
            return result
        manual = bool(result.get("manual_success"))
        if (result.get("boundary") or result.get("terminated")) and not manual:
            return result
        observation = result["next_obs"]
        previous_inference = self.last_inference
        probability = self.probability(observation) if self.model is not None else None
        self.last_inference = (observation, probability)
        predicted = probability is not None and probability >= self.cfg["threshold"]
        rewards = np.full(len(result.get("actions", [])), -0.05, np.float32)
        result = {**result, "rewards": rewards, "reward_probability": probability}
        if not manual and not predicted:
            return result
        decision_id = result.get("decision_id", observation.get("decision_id", uuid.uuid4().hex))
        if self.interactive:
            token = self.robot.begin_review("manual" if manual else "model", decision_id)
            feedback_observation, feedback_probability = observation, probability
            if manual and result.get("boundary") and previous_inference is not None:
                old_observation, old_probability = previous_inference
                if (
                    old_observation.get("decision_id") == decision_id
                    and old_probability is not None
                    and old_probability < self.cfg["threshold"]
                ):
                    feedback_observation, feedback_probability = old_observation, old_probability
            if manual and self.model is not None and feedback_probability < self.cfg["threshold"]:
                from capo.reward_feedback import save_feedback

                result["reward_feedback"] = save_feedback(
                    self.cfg,
                    feedback_observation,
                    success=True,
                    probability=feedback_probability,
                    decision_id=decision_id,
                )
            while True:
                phase = self.robot.poll_review(token)
                if phase == "rejected":
                    from capo.reward_feedback import save_feedback

                    result["reward_feedback"] = save_feedback(
                        self.cfg, observation, success=False, probability=probability, decision_id=decision_id
                    )

                    result["next_obs"] = self.robot.resume_attempt(token)
                    return {
                        **result,
                        "terminated": False,
                        "success": False,
                        "reward_override": -0.05 if result.get("boundary") else None,
                    }
                if phase == "confirmed":
                    break
                if phase in ("error", "aborted"):
                    raise RuntimeError("Success review/reset aborted; collection remains stopped")
                time.sleep(0.1)
        if result.get("boundary"):
            result["reward_override"] = 10.0
        else:
            result["rewards"][-1] = 10.0

        return {**result, "terminated": True, "truncated": False, "success": True}


def wrap_reward(robot, config):
    cfg = load_reward_config(config) if config else {"enabled": False, "interactive": True}
    if not cfg.get("enabled", False) and not cfg.get("interactive", True):
        return robot
    return RewardAdapter(robot, cfg)
