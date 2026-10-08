import argparse
import json
import os
from pathlib import Path
import select
import signal
import sys
import threading

from .flexiv import FlexivAdapter, OperatorEvents
from .flexiv_config import load_config
from .flexiv_devices import FlexivHardware, RealSenseCameras, SpaceMouse, hid_devices
from .flexiv_rpc import Client, make_server


def inventory():
    import importlib.metadata
    import pyrealsense2 as rs

    return {
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "flexivrdk", "pyrealsense2", "msgpack")
        },
        "cameras": [
            {"serial": d.get_info(rs.camera_info.serial_number), "name": d.get_info(rs.camera_info.name)}
            for d in rs.context().query_devices()
        ],
        "spacemice": hid_devices(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/robots/flexiv.yaml")
    parser.add_argument("--journal", default="data/flexiv_journal")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument(
        "--allow-reset", action="store_true", help="Also enable configured MoveJ on episode reset"
    )
    parser.add_argument(
        "--inventory", action="store_true", help="Enumerate USB devices; do not connect robot"
    )
    parser.add_argument(
        "--probe", action="store_true", help="Read robot state and camera frames once, then exit"
    )
    parser.add_argument(
        "--event",
        choices=("s", "f", "x", "h", "p", "enter", "space"),
        help="Send operator event to running server",
    )
    parser.add_argument("--probe-output", help="Save probe JSON and RGB PNGs in this directory")
    args = parser.parse_args()
    if args.inventory:
        print(json.dumps(inventory(), indent=2))
        return
    if args.event:
        print(Client(f"http://127.0.0.1:{args.port}").call("event", key=args.event))
        return
    if args.probe and (args.allow_motion or args.allow_reset):
        parser.error("--probe only supports read-only mode")
    if args.allow_reset and not args.allow_motion:
        parser.error("--allow-reset requires --allow-motion")
    cfg = load_config(args.config)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    hardware = FlexivHardware(cfg, allow_motion=args.allow_motion, allow_reset=args.allow_reset)
    cameras, teleop = RealSenseCameras(cfg), SpaceMouse(cfg["teleop"])
    events = OperatorEvents()
    from capo.config import local_path

    adapter = FlexivAdapter(cfg, hardware, cameras, teleop, events, journal_dir=local_path(args.journal))
    server = None
    quit_event = threading.Event()
    terminal_settings = None
    try:
        hardware.connect()
        cameras.start()
        if args.probe:
            obs = adapter.observe()
            report = {
                "status": hardware.status(),
                "state": obs["state"].tolist(),
                "images": {
                    key: None if value is None else list(value.shape) for key, value in obs["images"].items()
                },
                "config_sha256": adapter.config_sha256,
                "spacemice": hid_devices(),
            }
            report["configured_spacemouse_present"] = any(
                d["path"] == cfg["teleop"]["device_path"] for d in report["spacemice"]
            )
            report["tcp_inside_workspace"] = bool(
                all(
                    a <= x <= b
                    for x, a, b in zip(obs["state"][:3], cfg["workspace_min"], cfg["workspace_max"])
                )
            )
            print(json.dumps(report, indent=2))
            if args.probe_output:
                import cv2

                output = Path(args.probe_output)
                output.mkdir(parents=True, exist_ok=True)
                (output / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
                for name, image in obs["images"].items():
                    if image is not None:
                        cv2.imwrite(str(output / f"{name}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            return
        if args.allow_motion:
            teleop.start()
        server = make_server(adapter, args.port)

        def watch():
            while not quit_event.wait(0.1):
                adapter.watchdog()

        def keyboard():
            while not quit_event.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                if ready:
                    key = os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
                    if not key:
                        return
                    if key in ("\r", "\n", " "):
                        adapter.review.operator_key("space" if key == " " else "enter")
                    elif key in ("s", "f", "x", "h", "p"):
                        events.put(key)
                        if key == "x":
                            adapter.review.abort()
                            adapter.stop()

        threading.Thread(target=watch, daemon=True).start()
        if sys.stdin.isatty():
            import termios
            import tty

            terminal_settings = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
            threading.Thread(target=keyboard, daemon=True).start()
        print(
            json.dumps(
                {
                    "listening": server.server_address,
                    "allow_motion": args.allow_motion,
                    "config_sha256": adapter.config_sha256,
                    "operator": "Enter success/reset/continue; Space reject model success; x abort; h human; p policy",
                }
            ),
            flush=True,
        )
        server.serve_forever(poll_interval=0.2)
    finally:
        quit_event.set()
        if terminal_settings is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, terminal_settings)
        if server is not None:
            server.server_close()
        adapter.close()


if __name__ == "__main__":
    main()
