import os

from .flexiv_config import config_digest, load_config
from .flexiv_rpc import Client


class RemoteFlexiv:
    def __init__(self, cfg, robot_config, url):
        if cfg.robot_action_dim != 7 or cfg.state_dim < 8:
            raise ValueError("Flexiv adapter requires 7 action coordinates and at least 8 state slots")
        if list(cfg.image_keys) != robot_config["image_keys"]:
            raise ValueError("Robot/policy image keys differ")
        if cfg.execution_horizon != robot_config["execution_horizon"]:
            raise ValueError("Robot/policy execution horizons differ")
        if cfg.backend == "pi05" and 6 not in cfg.inactive_action_dims:
            raise ValueError("Fixed Flexiv gripper must be masked with inactive_action_dims=[6]")
        self.client = Client(url)
        health = self.client.call("health")
        if health["robot_config_sha256"] != config_digest(robot_config):
            raise ValueError("Robot host config differs from local config; redeploy the matching file")
        self.demonstration = False

    def reset(self):
        return self.client.call("reset")

    def choose_action(self, obs, proposal):
        result = self.client.call(
            "choose", decision_id=obs["decision_id"], proposal=proposal, demonstration=self.demonstration
        )
        self.decision_id = obs["decision_id"]
        return result["selected"], result["human"], result["aligned"]

    def execute(self, selected):
        import time

        try:
            return self.client.call("execute", selected=selected, decision_id=self.decision_id)
        except (OSError, TimeoutError):
            for _ in range(20):
                result = self.client.call("result", decision_id=self.decision_id)
                if result["status"] == "complete":
                    return result["result"]
                if result["status"] == "failed":
                    return {**result, "quarantined": True, "next_obs": None}
                time.sleep(0.1)
            raise RuntimeError("Execution outcome unresolved; inspect server journal before resetting")

    def configure_review(self):
        return self.client.call("review_configure")

    def begin_review(self, kind, decision_id):
        return self.client.call("review_begin", kind=kind, decision_id=decision_id)["token"]

    def poll_review(self, token):
        return self.client.call("review_poll", token=token)["phase"]

    def resume_attempt(self, token):
        return self.client.call("review_resume", token=token)

    def close(self):
        self.client.call("stop")


def create(cfg):
    from capo.config import local_path

    path = local_path(os.environ.get("CAPO_FLEXIV_CONFIG", "configs/robots/flexiv.yaml"))
    return RemoteFlexiv(cfg, load_config(path), os.environ.get("CAPO_FLEXIV_URL", "http://127.0.0.1:8766"))
