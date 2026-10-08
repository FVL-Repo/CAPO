import os
from pathlib import Path
import threading
import time

import numpy as np


class FlexivHardware:
    def __init__(self, cfg, *, allow_motion=False, allow_reset=False):
        self.cfg = cfg
        self.allow_motion, self.allow_reset = allow_motion, allow_reset
        self.robot = self.gripper = None
        self.activated = False
        self.cancel = threading.Event()

    def connect(self):
        import flexivrdk

        self.robot = flexivrdk.Robot(self.cfg["robot_sn"], self.cfg["network_interfaces"])
        deadline = time.monotonic() + 10
        while not self.robot.connected() or len(self.robot.states().q) != 7:
            if time.monotonic() >= deadline:
                raise TimeoutError("Flexiv telemetry did not become ready")
            time.sleep(0.05)
        self.gripper = flexivrdk.Gripper(self.robot)

    def status(self):
        return {
            "connected": self.robot.connected(),
            "fault": self.robot.fault(),
            "operational": self.robot.operational(),
            "mode": str(self.robot.mode()),
            "lite": self.robot.lite(),
            "activated": self.activated,
        }

    def state(self):
        state = self.robot.states()
        width = float(self.gripper.states().width)

        if not np.isfinite(width) or width < 0:
            raise RuntimeError("No valid gripper width telemetry")
        tcp = np.asarray(state.tcp_pose, dtype=np.float64)
        if tcp.shape != (7,) or not np.isfinite(tcp).all() or abs(np.linalg.norm(tcp[3:]) - 1) > 0.01:
            raise RuntimeError("Invalid robot TCP telemetry")
        return {"tcp": tcp, "joints": np.asarray(state.q), "gripper": width}

    def activate(self):
        if not self.allow_motion:
            raise RuntimeError("Read-only server: start with --allow-motion to execute commands")
        if self.robot.fault() or not self.robot.estop_released():
            raise RuntimeError("Robot fault/e-stop: resolve locally; CAPO does not clear it automatically")
        import flexivrdk

        tool = flexivrdk.Tool(self.robot)
        if tool.name() != self.cfg["gripper_name"]:
            raise RuntimeError("Active tool differs from configured GN01; configure it locally first")
        self._check_cancel()
        self.robot.Enable()
        self.activated = True
        self._check_cancel()
        deadline = time.monotonic() + 30
        while not self.robot.operational():
            if self.cancel.is_set():
                self.stop()
                raise RuntimeError("Activation cancelled")
            if time.monotonic() > deadline:
                self.stop()
                raise TimeoutError("Flexiv did not become operational")
            time.sleep(0.1)
        self._check_cancel()
        self.robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
        stiffness = [k * self.cfg["stiffness_ratio"] for k in self.robot.info().K_x_nom]
        self.robot.SetCartesianImpedance(stiffness, [0.8] * len(stiffness))

    def reset(self):
        if not self.activated:
            self.activate()
        if not self.allow_reset:
            return
        import flexivrdk

        self._check_cancel()
        self.robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
        self._check_cancel()
        self.robot.ExecutePrimitive(
            "MoveJ",
            {
                "target": flexivrdk.JPos(self.cfg["reset_joint_deg"]),
                "jntVelScale": self.cfg["reset_velocity_percent"],
            },
        )
        deadline = time.monotonic() + 30
        while not self.robot.primitive_states().get("reachedTarget", False):
            if self.cancel.is_set():
                self.stop()
                raise RuntimeError("MoveJ reset cancelled")
            if self.robot.fault() or time.monotonic() > deadline:
                self.stop()
                raise RuntimeError("Flexiv MoveJ reset failed or timed out")
            time.sleep(0.05)
        self.robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
        stiffness = [k * self.cfg["stiffness_ratio"] for k in self.robot.info().K_x_nom]
        self.robot.SetCartesianImpedance(stiffness, [0.8] * len(stiffness))

    def send(self, tcp):
        if not self.activated or not self.allow_motion:
            raise RuntimeError("Motion is not enabled")
        if self.robot.fault() or not self.robot.operational():
            raise RuntimeError("Robot became non-operational")
        if self.cancel.is_set():
            raise RuntimeError("Command cancelled")
        self.robot.SendCartesianMotionForce(np.asarray(tcp, dtype=float).tolist())

    def stop(self):
        self.cancel.set()
        if self.activated:
            try:
                self.robot.Stop()
            finally:
                self.activated = False

    def _check_cancel(self):
        if self.cancel.is_set():
            self.stop()
            raise RuntimeError("Control operation cancelled")

    def close(self):
        self.stop()
        self.gripper = self.robot = None


class RealSenseCameras:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipelines, self.threads, self.frames = [], [], {}
        self.lock, self.quit = threading.Lock(), threading.Event()

    def start(self):
        import pyrealsense2 as rs

        try:
            for camera in self.cfg["cameras"]:
                pipeline, config = rs.pipeline(), rs.config()
                config.enable_device(camera["serial_number"])
                config.enable_stream(
                    rs.stream.color, camera["width"], camera["height"], rs.format.rgb8, camera["fps"]
                )
                pipeline.start(config)
                self.pipelines.append(pipeline)
                thread = threading.Thread(target=self._read, args=(camera, pipeline), daemon=True)
                self.threads.append(thread)
                thread.start()
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                try:
                    self.latest()
                    return
                except RuntimeError:
                    time.sleep(0.05)
            raise TimeoutError("Configured RealSense cameras did not produce fresh frames")
        except BaseException:
            self.close()
            raise

    def _read(self, camera, pipeline):
        import cv2

        warmup = 5
        while not self.quit.is_set():
            try:
                frame = pipeline.wait_for_frames(timeout_ms=1000).get_color_frame()
                if not frame:
                    continue
                if warmup:
                    warmup -= 1
                    continue
                image = np.asarray(frame.get_data())
                (y0, y1), (x0, x1) = camera["crop"]
                image = image[y0 : None if y1 == -1 else y1, x0 : None if x1 == -1 else x1]
                if image.size == 0:
                    raise ValueError("Camera crop is empty")
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
                with self.lock:
                    self.frames[camera["name"]] = (image.copy(), time.monotonic())
            except Exception:
                self.quit.wait(0.02)

    def latest(self):
        with self.lock:
            result = {key: None for key in self.cfg["image_keys"]}
            for camera in self.cfg["cameras"]:
                name = camera["name"]
                if (
                    name not in self.frames
                    or time.monotonic() - self.frames[name][1] > self.cfg["camera_max_age_s"]
                ):
                    raise RuntimeError(f"Missing or stale camera: {name}")
                result[name] = self.frames[name][0].copy()
            return result

    def close(self):
        self.quit.set()
        for pipeline in self.pipelines:
            pipeline.stop()
        for thread in self.threads:
            thread.join(timeout=2)
        self.pipelines.clear()
        self.threads.clear()


def hid_devices():
    result = []
    for path in sorted(Path("/sys/class/hidraw").glob("hidraw*")):
        entries = dict(
            line.split("=", 1) for line in (path / "device/uevent").read_text().splitlines() if "=" in line
        )
        parts = entries.get("HID_ID", "").split(":")
        if len(parts) == 3 and int(parts[1], 16) == 0x256F:
            node = f"/dev/{path.name}"
            result.append(
                {
                    "path": node,
                    "vendor_id": int(parts[1], 16),
                    "product_id": int(parts[2], 16),
                    "name": entries.get("HID_NAME"),
                    "readable": os.access(node, os.R_OK),
                }
            )
    return result


class SpaceMouse:
    """Raw-axis mapping with separate button events and motion commands."""

    def __init__(self, cfg):
        self.cfg, self.fd = cfg, None
        self.axes, self.buttons = np.zeros(6), 0
        self.failed = None
        self.lock, self.quit = threading.Lock(), threading.Event()
        self.thread = None

    def start(self):
        matches = [
            d
            for d in hid_devices()
            if d["path"] == self.cfg["device_path"]
            and d["vendor_id"] == self.cfg["vendor_id"]
            and d["product_id"] == self.cfg["product_id"]
        ]
        if not matches:
            raise RuntimeError("Configured SpaceMouse path does not match USB identity; run --inventory")
        self.fd = os.open(self.cfg["device_path"], os.O_RDONLY | os.O_NONBLOCK)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while not self.quit.is_set():
            try:
                data = os.read(self.fd, 13)
                if not data:
                    raise OSError("SpaceMouse disconnected")
                self.feed(data)
            except BlockingIOError:
                self.quit.wait(0.001)
            except OSError as exc:
                self.failed = str(exc)
                return

    def feed(self, data):
        def axis(offset):
            return np.clip(int.from_bytes(data[offset : offset + 2], "little", signed=True) / 350, -1, 1)

        with self.lock:
            if data[0] in (1, 2) and len(data) >= 7:
                offset = 0 if data[0] == 1 else 3
                self.axes[offset : offset + 3] = [axis(3), axis(1), -axis(5)]
                if data[0] == 1 and len(data) >= 13:
                    self.axes[3:] = [axis(9), axis(7), -axis(11)]
            elif data[0] == 3 and len(data) >= 2:
                self.buttons = data[1]

    def sample(self):
        if self.failed:
            raise RuntimeError(self.failed)
        with self.lock:
            axes, buttons = self.axes.copy(), self.buttons
        axes[np.abs(axes) < self.cfg["deadzone"]] = 0
        active = bool(np.linalg.norm(axes) >= self.cfg["motion_threshold"])
        axes = np.sign(axes) * np.abs(axes) ** self.cfg["response_exponent"]

        delta = np.r_[axes[:3] * self.cfg["pos_scale"], axes[3:] * self.cfg["rot_scale"]]
        return {"delta": delta if active else np.zeros(6), "active": active, "abort": bool(buttons & 3 == 3)}

    def close(self):
        self.quit.set()
        if self.thread:
            self.thread.join(timeout=2)
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
