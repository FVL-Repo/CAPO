import hashlib
import json


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def bind_spec(cfg, spec):
    if cfg.data_spec is not None and canonical(cfg.data_spec) != canonical(spec):
        raise ValueError(
            "Data specification mismatch: normalization, task, cameras or action semantics differ"
        )
    cfg.data_spec = spec


def check_spec(cfg, spec):
    if canonical(cfg.data_spec) != canonical(spec):
        raise ValueError("Replay data specification mismatch; reimport with the matching processor")


def bind_robot(cfg, robot_cfg):
    from capo.robots.flexiv_config import config_digest

    spec = dict(cfg.data_spec or {})
    addition = {
        "robot_config_sha256": config_digest(robot_cfg),
        "state_format": robot_cfg["state_format"],
        "action_format": robot_cfg["action_format"],
        "control_hz": robot_cfg["control_hz"],
        "task": robot_cfg["prompt"],
    }
    for key, value in addition.items():
        if key in spec and spec[key] != value:
            raise ValueError(f"Robot/data contract mismatch: {key}")
    cfg.data_spec = {**spec, **addition}
    if not robot_cfg["command_gripper"]:
        cfg.inactive_action_dims = tuple(sorted(set(cfg.inactive_action_dims) | {6}))
