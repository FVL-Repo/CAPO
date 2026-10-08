import hashlib
import json

import numpy as np


def load_config(path):
    from capo.config import load_mapping

    cfg = load_mapping(path)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if cfg["state_format"] != "tcp_xyz_quaternion_wxyz_gripper_width_m":
        raise ValueError("Unsupported Flexiv state convention")
    if cfg["action_format"] != "delta_xyz_m_euler_xyz_rad_base_absolute_gripper_width_m":
        raise ValueError("Unsupported Flexiv action convention")
    if cfg["command_gripper"]:
        raise ValueError("Flexiv adapter preserves gripper width; gripper commands are disabled")
    for key in ("workspace_min", "workspace_max"):
        value = np.asarray(cfg[key])
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key}")
    if np.any(np.asarray(cfg["workspace_min"]) >= cfg["workspace_max"]):
        raise ValueError("Invalid workspace")
    for key in (
        "control_hz",
        "execution_horizon",
        "max_episode_steps",
        "max_translation_step_m",
        "max_rotation_step_rad",
        "camera_max_age_s",
        "decision_timeout_s",
        "idle_stop_s",
        "local_xy_radius",
        "success_events",
    ):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not 0 < cfg["stiffness_ratio"] <= 1:
        raise ValueError("Invalid stiffness ratio")
    if len(cfg["reset_joint_deg"]) != 7 or not np.isfinite(cfg["reset_joint_deg"]).all():
        raise ValueError("Invalid reset joints")
    names = [c["name"] for c in cfg["cameras"]]
    serials = [c["serial_number"] for c in cfg["cameras"]]
    if len(set(names)) != len(names) or len(set(serials)) != len(serials):
        raise ValueError("Duplicate cameras")
    if not set(names).issubset(cfg["image_keys"]):
        raise ValueError("Camera names must map to policy image keys")
    if cfg["teleop"]["translation_frame"] != "base":
        raise ValueError("This adapter uses base-frame deltas")


def config_digest(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
