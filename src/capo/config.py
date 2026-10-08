from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def local_path(path: str | Path) -> Path:
    p = Path(path)
    p = (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError(f"Path must be inside {ROOT}; copy the asset here first: {p}")
    return p


@dataclass
class Config:
    backend: str = "pi05"
    seed: int = 42
    device: str = "cuda"
    checkpoint: str = "assets/openpi_pi05_base/model.safetensors"
    demonstrations: str = ""
    replay: str = "data/replay"
    output: str = "runs/pi05_capo"
    base_sha256: str = ""
    data_spec: dict | None = None
    replay_capacity: int = 100000
    replay_cache_shards: int = 8
    inactive_action_dims: tuple = ()
    action_semantics: str = "command"
    action_horizon: int = 50
    execution_horizon: int = 10
    action_dim: int = 32
    robot_action_dim: int = 7
    state_dim: int = 32
    image_keys: tuple = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    max_token_len: int = 200
    precision: str = "bfloat16"
    gradient_checkpointing: bool = True
    actor_tuning: str = "lora"
    lora_rank: int = 32
    lora_alpha: float = 16.0
    train_action_time_projections: bool = True
    critic_heads: int = 2
    critic_hidden: int = 256
    critic_resnet_weights: str = "assets/resnet10.safetensors"
    critic_visual_dim: int = 256
    critic_proprio_dim: int = 64
    critic_spatial_blocks: int = 8
    gamma: float = 0.99
    target_retention: float = 0.995
    td_samples: int = 1
    margin: float = 0.1
    rank_weight: float = 10.0
    eta: float = 1.0
    alpha_q: float = 1.0
    bc_weight: float = 0.5
    flow_steps: int = 10
    cutoff: float = 0.5
    cutoff_rule: str = "ceil"
    apo_time_samples: int = 5
    batch_size: int = 256
    microbatch_size: int = 1
    replay_mixture: tuple = (0.4, 0.4, 0.2)
    bc_correction_fraction: float = 0.5
    critic_warmup_steps: int = 200
    critic_steps_per_round: int = 100
    actor_steps_per_round: int = 100
    rounds: int = 10
    sft_steps: int = 10000
    critic_lr: float = 3e-5
    actor_lr: float = 2e-5
    sft_lr: float = 5e-5
    critic_min_lr: float = 3e-6
    actor_min_lr: float = 2.5e-6
    critic_lr_warmup: int = 100
    actor_lr_warmup: int = 1000
    sft_lr_warmup: int = 10000
    critic_schedule_steps: int = 20000
    actor_schedule_steps: int = 10000
    weight_decay: float = 1e-10
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float | None = None

    def validate(self):
        if self.replay_capacity < 1 or self.replay_cache_shards < 1:
            raise ValueError("Replay capacity/cache must be positive")
        if self.action_semantics != "command":
            raise ValueError("Only command-space Q is supported")
        if any(i < 0 or i >= self.robot_action_dim for i in self.inactive_action_dims):
            raise ValueError("Invalid inactive action dimension")
        if self.actor_tuning not in ("lora", "full"):
            raise ValueError("actor_tuning must be lora or full")
        if (
            self.lora_rank < 1
            or not isinstance(self.lora_rank, int)
            or not 0 < self.lora_alpha < float("inf")
        ):
            raise ValueError("LoRA requires positive integer rank and finite positive alpha")
        if not isinstance(self.train_action_time_projections, bool):
            raise ValueError("train_action_time_projections must be boolean")
        if self.backend != "pi05":
            raise ValueError("backend must be pi05")
        for name in (
            "action_horizon",
            "execution_horizon",
            "action_dim",
            "robot_action_dim",
            "state_dim",
            "flow_steps",
            "apo_time_samples",
            "batch_size",
            "microbatch_size",
            "critic_heads",
            "critic_hidden",
            "critic_visual_dim",
            "critic_proprio_dim",
            "critic_spatial_blocks",
            "td_samples",
            "rounds",
            "critic_steps_per_round",
            "actor_steps_per_round",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.execution_horizon <= self.action_horizon:
            raise ValueError("execution_horizon must be within the generated chunk")
        if not 0 < self.robot_action_dim <= self.action_dim:
            raise ValueError("robot_action_dim must be within action_dim")
        if not 0 <= self.cutoff < 1 or self.cutoff_rule not in ("ceil", "floor", "error"):
            raise ValueError("invalid cutoff/cutoff_rule")
        if not 0 < self.gamma <= 1 or not 0 <= self.target_retention < 1:
            raise ValueError("invalid discount or target retention")
        if (
            len(self.replay_mixture) != 3
            or min(self.replay_mixture) < 0
            or abs(sum(self.replay_mixture) - 1) > 1e-6
        ):
            raise ValueError("replay_mixture must be three nonnegative weights summing to one")
        if (
            self.eta <= 0
            or self.alpha_q <= 0
            or self.bc_weight < 0
            or self.rank_weight < 0
            or self.margin < 0
        ):
            raise ValueError("invalid loss scale")
        if not 0 <= self.bc_correction_fraction <= 1:
            raise ValueError("invalid BC mixture")
        return self

    @classmethod
    def load(cls, path):
        data = load_mapping(path)
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config fields: {unknown}")
        return cls(**data).validate()

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


def load_mapping(path):
    path = local_path(path)
    if path.suffix in (".yaml", ".yml"):
        import yaml

        value = yaml.safe_load(path.read_text())
    else:
        value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value
