def check_actor_config(actor, saved):
    from capo.identity import canonical

    for key in (
        "backend",
        "action_horizon",
        "execution_horizon",
        "action_dim",
        "robot_action_dim",
        "state_dim",
        "image_keys",
        "max_token_len",
        "inactive_action_dims",
        "action_semantics",
        "data_spec",
    ):
        old = saved.get(key, getattr(actor.cfg, key))
        if canonical(old) != canonical(getattr(actor.cfg, key)):
            raise ValueError(f"Actor checkpoint mismatch for {key}")
    if actor.cfg.backend != "pi05":
        return
    if saved.get("checkpoint") != actor.cfg.checkpoint:
        raise ValueError("Actor checkpoint mismatch for base checkpoint path")
    if not actor.cfg.base_sha256 or saved.get("base_sha256") != actor.cfg.base_sha256:
        raise ValueError("Actor base content hash mismatch or missing; migrate legacy checkpoint explicitly")
    defaults = {
        "actor_tuning": "full",
        "lora_rank": 32,
        "lora_alpha": 16.0,
        "train_action_time_projections": True,
    }
    for key, default in defaults.items():
        if key.startswith("lora_") and actor.cfg.actor_tuning != "lora":
            continue
        if saved.get(key, default) != getattr(actor.cfg, key):
            raise ValueError(
                f"Actor checkpoint mismatch for {key}; use matching config. "
                "For full-expert to LoRA migration, export the old actor as a new base checkpoint first."
            )


def load_actor_state(actor, state, checkpoint_config=None):
    if checkpoint_config is not None:
        check_actor_config(actor, checkpoint_config)
    expected = {n for n, p in actor.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError(
            f"Actor checkpoint keys mismatch: missing={expected - set(state)}, extra={set(state) - expected}"
        )
    actor.load_state_dict(state, strict=False)
