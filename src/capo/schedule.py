"""Persistent optimizer schedule, independent of collection episode boundaries."""

TRAINING_SCHEDULE = "warmup_then_c_ca_reference_v1"


def next_stage(stage, critic_step, warmup):
    if stage == "warmup":
        return "warmup" if critic_step < warmup else "critic_only"
    return {"critic_only": "joint_critic", "joint_critic": "joint_actor",
            "joint_actor": "critic_only"}[stage]


def validate_schedule_state(trainer):
    stage, c, a = trainer.schedule_stage, trainer.critic_step, trainer.actor_step
    warmup = trainer.cfg.critic_warmup_steps
    if stage == "warmup":
        valid = a == 0 and 0 <= c < warmup
    else:
        offsets = {"critic_only": 0, "joint_critic": 1, "joint_actor": 2}
        valid = stage in offsets and c == warmup + 2 * a + offsets.get(stage, 0)
    if not valid:
        raise ValueError(f"Invalid update schedule state: {stage=}, {c=}, {a=}")


def training_complete(payload, cfg):
    return (payload.get("training_schedule") == TRAINING_SCHEDULE
            and payload.get("schedule_stage") == "critic_only"
            and payload["actor_step"] == cfg.max_actor_steps
            and payload["critic_step"] == cfg.critic_warmup_steps + 2 * cfg.max_actor_steps)
