from typing import Protocol
import importlib
from pathlib import Path
import numpy as np
from capo.config import ROOT


class RobotAdapter(Protocol):
    """Actions use the EXACT coordinate convention expected by the SFT checkpoint.

    reset/observe return raw {state, images, prompt}. choose_action receives the
    pre-execution observation and proposed action prefix, returning (selected,
    is_human, pair_aligned). execute returns actual actions/rewards, next_obs,
    terminated, truncated, human_steps, success, safety_abort. If a human takes
    over mid-chunk, split at that boundary and obtain a new proposal at the new
    state before creating a preference pair. Never pair across decision states.
    """

    def reset(self) -> dict: ...
    def choose_action(self, obs: dict, proposal: np.ndarray) -> tuple[np.ndarray, bool, bool]: ...
    def execute(self, selected: np.ndarray) -> dict: ...


class ReviewableRobot(Protocol):
    """Optional interactive reward contract. begin_review must physically stop first.

    poll_review returns model/manual/resetting/ready/confirmed/rejected/error/aborted.
    A rejected review stays stopped until resume_attempt; a confirmed reset is
    consumed by the next reset() without performing the physical reset twice.
    """

    def configure_review(self) -> dict: ...
    def begin_review(self, kind: str, decision_id: str) -> str: ...
    def poll_review(self, token: str) -> str: ...
    def resume_attempt(self, token: str) -> dict: ...


def create_adapter(spec, cfg):
    module, name = spec.split(":")
    imported = importlib.import_module(module)
    if not Path(imported.__file__).resolve().is_relative_to(ROOT):
        raise ValueError("Place your adapter code inside the CAPO project")
    return getattr(imported, name)(cfg)
