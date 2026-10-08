import threading
import time

MODEL_PROMPT = "Reward model detected success. Press Space to keep trying, or Enter to reset."
MANUAL_PROMPT = "Operator confirmed success. Press Enter to reset."
READY_PROMPT = "Reset complete. Press Enter to resume collection."


class RewardReview:
    def __init__(self, adapter):
        self.adapter = adapter
        self.lock = threading.Lock()
        self.phase = "running"
        self.token = None
        self.key = None
        self.ready = False
        self.enabled = False

    def configure(self):
        hardware = self.adapter.hardware
        if not getattr(hardware, "simulated", False) and not getattr(hardware, "allow_reset", False):
            raise RuntimeError("Interactive success/reset requires robot server --allow-motion --allow-reset")
        if not getattr(hardware, "allow_motion", False):
            raise RuntimeError("Interactive review requires motion permission")
        self.enabled = True
        return {"ok": True}

    def operator_key(self, key):
        with self.lock:
            if self.phase in ("model", "manual", "ready"):
                if self.key is None and (key == "enter" or (key == "space" and self.phase == "model")):
                    self.key = key
            elif self.phase == "running" and key == "enter":
                self.adapter.events.put("enter")

    def begin(self, kind, decision_id):
        if not self.enabled or kind not in ("model", "manual"):
            raise ValueError("Interactive review is not configured")

        if self.token == decision_id:
            return {"token": self.token, "phase": self.phase}
        if self.phase != "running":
            raise RuntimeError("Another success review is pending")
        self.adapter.stop()
        self.adapter.events.take()
        with self.lock:
            self.token, self.phase, self.key = decision_id, kind, None
        print(MODEL_PROMPT if kind == "model" else MANUAL_PROMPT, flush=True)
        return {"token": self.token, "phase": self.phase}

    def poll(self, token):
        self._check(token)
        with self.lock:
            phase, key = self.phase, self.key
            self.key = None
            if key == "space" and phase == "model":
                self.phase = "rejected"
            elif key == "enter" and phase in ("model", "manual"):
                self.phase = "resetting"
            elif key == "enter" and phase == "ready":
                self.phase, self.ready = "confirmed", True
        if self.phase == "resetting":
            try:
                self.adapter.reset(_review=True)
                self.adapter.stop()
            except BaseException:
                self.phase = "error"
                self.adapter.stop()
                raise
            with self.lock:
                if self.phase == "aborted":
                    raise RuntimeError("Reset aborted")
                self.phase, self.key = "ready", None
            print(READY_PROMPT, flush=True)
        return {"token": token, "phase": self.phase}

    def resume_attempt(self, token):
        self._check(token)
        if self.phase == "running":
            return self.adapter.current[0]
        if self.phase != "rejected":
            raise RuntimeError("Only a rejected model success can resume without reset")
        obs = self._activate()
        with self.lock:
            if self.phase == "aborted":
                raise RuntimeError("Operator aborted review")
            self.phase, self.key = "running", None
        return obs

    def consume_ready(self):
        if not self.ready:
            raise RuntimeError("Reset/continue confirmation has not completed")
        obs = self._activate()
        with self.lock:
            if self.phase == "aborted":
                raise RuntimeError("Operator aborted review")
            self.ready, self.phase, self.key = False, "running", None
        return obs

    def _activate(self):
        adapter = self.adapter
        adapter.cameras.latest()
        with self.lock:
            if self.phase == "aborted":
                raise RuntimeError("Operator aborted review")
            adapter.cancel.clear()
        adapter.hardware.activate()
        if adapter.cancel.is_set() or self.phase == "aborted":
            adapter.stop()
            raise RuntimeError("Activation cancelled")
        adapter.done = False
        adapter.last_activity = time.monotonic()
        adapter.pending = None
        adapter.events.take()
        return adapter.observe()

    def abort(self):
        with self.lock:
            self.phase, self.ready, self.key = "aborted", False, None

    def _check(self, token):
        if token != self.token:
            raise ValueError("Stale reward review token")
