import threading
import time
import uuid
import os
from pathlib import Path

import numpy as np

from .flexiv_config import config_digest, validate_config
from .flexiv_math import clip_norm, euler_delta_to_matrix, matrix_to_quat_wxyz, tcp_pose_to_matrix


class OperatorEvents:
    def __init__(self):
        self.lock = threading.Lock()
        self.events = []
        self.human = False

    def put(self, key):
        if key not in ("s", "f", "x", "h", "p", "enter"):
            raise ValueError("Events: s=success event, f=failure, x=abort, h=human hold, p=policy")
        with self.lock:
            if key in ("h", "p"):
                self.human = key == "h"
            else:
                self.events.append(key)

    def take(self):
        with self.lock:
            events, self.events = self.events, []
            return events


class FlexivAdapter:
    def __init__(self, cfg, hardware, cameras, teleop, events=None, *, sleep=time.sleep, journal_dir=None):
        validate_config(cfg)
        self.cfg, self.hardware, self.cameras, self.teleop = cfg, hardware, cameras, teleop
        self.events = events or OperatorEvents()
        from .reward_review import RewardReview

        self.review = RewardReview(self)
        self.sleep = sleep
        self.cancel = threading.Event()
        self.hardware.cancel = self.cancel
        self.journal_dir = Path(journal_dir) if journal_dir else None
        if self.journal_dir:
            self.journal_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.active_execution = None
        self.pending = None
        self.current = None
        self.episode = None
        self.steps = self.success_count = 0
        self.done = True
        self.last_activity = time.monotonic()
        self.lock = threading.RLock()
        self.config_sha256 = config_digest(cfg)
        self.local_min, self.local_max = np.array(cfg["workspace_min"]), np.array(cfg["workspace_max"])

    def observe(self):
        state = self.hardware.state()
        images = self.cameras.latest()
        obs = {
            "state": np.r_[state["tcp"], state["gripper"]].astype(np.float32),
            "images": images,
            "prompt": self.cfg["prompt"],
            "decision_id": uuid.uuid4().hex,
            "robot_metadata": {
                "robot_sn": self.cfg["robot_sn"],
                "episode_id": self.episode,
                "timestamp_unix_s": time.time(),
                "robot_config_sha256": self.config_sha256,
                "state_format": self.cfg["state_format"],
                "action_format": self.cfg["action_format"],
            },
        }
        self.current = (obs, time.monotonic(), state)
        return obs

    def reset(self, *, _review=False):
        with self.lock:
            if not _review and self.review.ready:
                return self.review.consume_ready()
            if not _review and self.review.phase != "running":
                raise RuntimeError("Finish the success/reset/continue gates before collection")
            with self.review.lock:
                if _review and self.review.phase != "resetting":
                    raise RuntimeError("Review reset cancelled")
                self.cancel.clear()
            self.cameras.latest()
            self.teleop.sample()
            try:
                if not getattr(self.hardware, "allow_reset", False):
                    xyz = self.hardware.state()["tcp"][:3]
                    if np.any(xyz < np.asarray(self.cfg["workspace_min"]) - 1e-4) or np.any(
                        xyz > np.asarray(self.cfg["workspace_max"]) + 1e-4
                    ):
                        raise RuntimeError(
                            "Current TCP is outside configured workspace; align locally before reset"
                        )
                self.hardware.reset()
                if self.cancel.is_set():
                    raise RuntimeError("Reset cancelled")
                self.episode = uuid.uuid4().hex
                self.steps = self.success_count = 0
                self.events.take()
                self.pending = None
                state = self.hardware.state()
                xyz = state["tcp"][:3]
                low, high = np.array(self.cfg["workspace_min"]), np.array(self.cfg["workspace_max"])
                if np.any(xyz < low - 1e-4) or np.any(xyz > high + 1e-4):
                    raise RuntimeError(
                        "Current TCP is outside configured workspace; align locally before reset"
                    )
                radius = np.array([self.cfg["local_xy_radius"], self.cfg["local_xy_radius"], np.inf])
                self.local_min, self.local_max = np.maximum(low, xyz - radius), np.minimum(high, xyz + radius)
                self.done = False
                self.last_activity = time.monotonic()
                return self.observe()
            except BaseException:
                self.hardware.stop()
                self.done = True
                raise

    def _check_decision(self, decision_id):
        if self.done or self.current is None or self.current[0]["decision_id"] != decision_id:
            raise ValueError("Expired/mismatched decision; obtain a new observation")
        if time.monotonic() - self.current[1] > self.cfg["decision_timeout_s"]:
            raise TimeoutError("Policy decision expired")
        before, now = self.current[2], self.hardware.state()
        rotation_error = 2 * np.arccos(np.clip(abs(np.dot(before["tcp"][3:], now["tcp"][3:])), 0, 1))
        if np.linalg.norm(before["tcp"][:3] - now["tcp"][:3]) > 0.005 or rotation_error > 0.02:
            raise RuntimeError("Robot moved during inference; discard proposal and observe again")

    def choose_action(self, obs, proposal, *, demonstration=False):
        with self.lock:
            self._check_decision(obs["decision_id"])
            if self.pending is not None:
                raise RuntimeError("An action is already pending")
            proposal = np.asarray(proposal, dtype=np.float32)
            if (
                proposal.ndim != 2
                or proposal.shape[1] != 7
                or not 1 <= len(proposal) <= self.cfg["execution_horizon"]
                or not np.isfinite(proposal).all()
            ):
                raise ValueError("Expected finite physical actions [1..execution_horizon,7]")
            sample = self.teleop.sample()
            if sample["abort"]:
                self.stop()
                raise RuntimeError("SpaceMouse two-button abort")
            human = bool(demonstration or self.events.human or sample["active"])
            selected = (
                np.r_[sample["delta"], self.current[2]["gripper"]][None].astype(np.float32)
                if human
                else proposal.copy()
            )
            self.pending = {"selected": selected.copy(), "human": human, "decision_id": obs["decision_id"]}
            self.last_activity = time.monotonic()
            return selected, human, human and not demonstration

    def _target(self, action, state):

        delta = clip_norm(action[:3], self.cfg["max_translation_step_m"])
        matrix = tcp_pose_to_matrix(state["tcp"])
        xyz = np.clip(matrix[:3, 3] + delta, self.local_min, self.local_max)
        executed_xyz = xyz - matrix[:3, 3]
        if np.linalg.norm(executed_xyz) > self.cfg["max_translation_step_m"] + 1e-7:
            raise RuntimeError("Measured TCP left workspace; refusing a jump back to the boundary")
        euler = np.asarray(action[3:6], dtype=float).copy()

        euler = clip_norm(euler, self.cfg["max_rotation_step_rad"])
        for _ in range(3):
            delta_rot = euler_delta_to_matrix(*euler)
            angle = np.arccos(np.clip((np.trace(delta_rot) - 1) / 2, -1, 1))
            if angle <= self.cfg["max_rotation_step_rad"] + 1e-10:
                break
            euler *= self.cfg["max_rotation_step_rad"] / angle
        rot = euler_delta_to_matrix(*euler) @ matrix[:3, :3]
        tcp = np.r_[xyz, matrix_to_quat_wxyz(rot)]

        executed = np.r_[executed_xyz, euler, state["gripper"]].astype(np.float32)
        return tcp, executed

    def _journal(self, decision_id, result):
        from .flexiv_rpc import pack

        self.results = {decision_id: result}
        if self.journal_dir:
            path = self.journal_dir / (decision_id + ".msgpack")
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                stream.write(pack(result))
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)

    def execution_result(self, decision_id):
        from .flexiv_rpc import unpack

        if (
            not isinstance(decision_id, str)
            or len(decision_id) != 32
            or any(c not in "0123456789abcdef" for c in decision_id)
        ):
            raise ValueError("Invalid decision ID")
        if self.journal_dir:
            path = self.journal_dir / (decision_id + ".msgpack")
            if path.exists():
                return unpack(path.read_bytes())
        return self.results.get(decision_id, {"status": "unknown"})

    def execute(self, selected, decision_id=None):
        with self.lock:
            if self.pending is None:
                raise ValueError("No pending decision (already consumed or expired)")
            token = self.pending["decision_id"]
            if decision_id is not None and token != decision_id:
                raise ValueError("Execute decision ID mismatch")
            self.active_execution = token
            self._journal(token, {"status": "started", "decision_id": token})
            try:
                result = self._execute(selected)
                result["decision_id"] = token
                self._journal(token, {"status": "complete", "result": result})
                return result
            except Exception as exc:
                previous = self.execution_result(token)
                self._journal(token, {**previous, "status": "failed", "error": str(exc)})
                self.stop()
                raise
            finally:
                self.active_execution = None

    def _execute(self, selected):
        with self.lock:
            pending, self.pending = self.pending, None
            if pending is None:
                raise ValueError("No pending decision (already consumed or expired)")
            selected = np.asarray(selected, np.float32)
            if selected.shape != pending["selected"].shape or not np.array_equal(
                selected, pending["selected"]
            ):
                raise ValueError("Execute must use the exact selected action")
            self._check_decision(pending["decision_id"])
            human = pending["human"]
            actions, commands, rewards = [], [], []
            terminated = truncated = success = safety_abort = False
            reason = None
            manual_success = False
            for action in selected:
                started = time.monotonic()
                try:
                    if not human:
                        sample = self.teleop.sample()
                        if sample["abort"]:
                            raise RuntimeError("SpaceMouse two-button abort")
                        if sample["active"] or self.events.human:
                            break
                    else:
                        if self.teleop.sample()["abort"]:
                            raise RuntimeError("SpaceMouse two-button abort")
                    events = self.events.take()
                    if self.cancel.is_set() or "x" in events:
                        raise RuntimeError("Operator abort")
                    if self.review.enabled:
                        events = [e for e in events if e != "s"]
                    if "enter" in events:
                        self.hardware.stop()
                        if not actions:
                            self.done = True
                            return {
                                "boundary": True,
                                "terminated": True,
                                "truncated": False,
                                "success": True,
                                "manual_success": True,
                                "safety_abort": False,
                                "reward_override": 10.0,
                                "next_obs": self.observe(),
                            }
                        rewards[-1] = 10.0
                        terminated = success = manual_success = True
                        break

                    if "s" in events and "f" not in events:
                        self.success_count += events.count("s")
                        success = self.success_count >= self.cfg["success_events"]
                    if "f" in events or "s" in events:
                        terminated = "f" in events or success
                        label_reward = (
                            (self.cfg["success_reward"] if success else self.cfg["intermediate_reward"])
                            if "s" in events
                            else None
                        )
                        if "f" in events:
                            success = False
                            label_reward = None
                        if not actions:
                            self.done = terminated
                            if self.done:
                                self.hardware.stop()
                            return {
                                "boundary": True,
                                "terminated": terminated,
                                "truncated": False,
                                "success": success,
                                "safety_abort": False,
                                "reward_override": label_reward,
                                "next_obs": self.observe(),
                            }
                        if label_reward is not None:
                            rewards[-1] = label_reward
                        if terminated:
                            break
                    self.cameras.latest()
                    state = self.hardware.state()
                    tcp, executed = self._target(action, state)
                    if self.cancel.is_set():
                        raise RuntimeError("Execution cancelled")
                    self._journal(
                        self.active_execution,
                        {
                            "status": "inflight",
                            "decision_id": self.active_execution,
                            "actions": np.asarray(actions),
                            "commands": np.asarray(commands),
                            "rewards": np.asarray(rewards),
                            "next_command": action,
                            "note": "next_command may have executed if process/SDK fails before acknowledgement",
                        },
                    )
                    self.hardware.send(tcp)
                    actions.append(executed)
                    commands.append(action.copy())
                    rewards.append(-0.05 if self.review.enabled else self.cfg["step_reward"])
                    self._journal(
                        self.active_execution,
                        {
                            "status": "partial",
                            "decision_id": self.active_execution,
                            "actions": np.asarray(actions),
                            "commands": np.asarray(commands),
                            "rewards": np.asarray(rewards),
                        },
                    )
                    self.steps += 1
                    self.last_activity = time.monotonic()
                    self.sleep(max(0, 1 / self.cfg["control_hz"] - (time.monotonic() - started)))
                    events = self.events.take()
                    if self.cancel.is_set() or "x" in events:
                        raise RuntimeError("Operator abort")
                    if self.review.enabled:
                        events = [e for e in events if e != "s"]
                    if "enter" in events:
                        terminated = success = manual_success = True
                        rewards[-1] = 10.0
                    elif "f" in events:
                        terminated = True
                    elif "s" in events:
                        self.success_count += 1
                        success = self.success_count >= self.cfg["success_events"]
                        terminated = success
                        rewards[-1] = (
                            self.cfg["success_reward"] if success else self.cfg["intermediate_reward"]
                        )
                    truncated = self.steps >= self.cfg["max_episode_steps"] and not terminated
                    if terminated or truncated:
                        break
                except Exception as exc:
                    self.hardware.stop()
                    self.done = True
                    if not actions:
                        raise
                    truncated = safety_abort = True
                    reason = str(exc)
                    break
            self.last_activity = time.monotonic()
            if not actions:
                return {"retry": True, "next_obs": self.observe()}
            self.done = terminated or truncated
            if self.done:
                self.hardware.stop()
            try:
                next_obs = self.observe()
            except Exception as exc:
                self.hardware.stop()
                self.done = truncated = safety_abort = True
                next_obs = None
                reason = str(exc)
            return {
                "commands": np.asarray(commands),
                "actions": np.asarray(actions),
                "rewards": np.asarray(rewards, np.float32),
                "next_obs": next_obs,
                "quarantined": next_obs is None,
                "terminated": terminated,
                "truncated": truncated,
                "human_steps": len(actions) if human else 0,
                "success": success,
                "safety_abort": safety_abort,
                "abort_reason": reason,
                "success_count": self.success_count,
                "manual_success": manual_success,
            }

    def stop(self):

        self.cancel.set()
        self.done = True
        self.hardware.stop()

    def watchdog(self):
        if not self.done:
            try:
                if self.teleop.sample()["abort"]:
                    self.stop()
                self.cameras.latest()
                if time.monotonic() - self.last_activity > self.cfg["idle_stop_s"]:
                    self.stop()
            except Exception:
                self.stop()

    def close(self):
        try:
            self.stop()
        finally:
            try:
                self.teleop.close()
            finally:
                try:
                    self.cameras.close()
                finally:
                    self.hardware.close()
