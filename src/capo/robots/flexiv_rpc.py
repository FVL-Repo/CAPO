from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import msgpack
import numpy as np

MAX_MESSAGE_BYTES = 8 * 1024 * 1024


def pack(value):
    def encode(item):
        if isinstance(item, np.ndarray):
            return {"__array__": True, "dtype": item.dtype.str, "shape": item.shape, "data": item.tobytes()}
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    return msgpack.packb(value, default=encode, use_bin_type=True)


def unpack(value):
    if len(value) > MAX_MESSAGE_BYTES:
        raise ValueError("Message too large")

    def decode(item):
        if item.get("__array__") is True:
            dtype = np.dtype(item["dtype"])
            if dtype.kind not in "buif" or dtype.itemsize > 8:
                raise ValueError("Unsupported array dtype")
            return np.frombuffer(item["data"], dtype=dtype).reshape(item["shape"]).copy()
        return item

    return msgpack.unpackb(value, raw=False, object_hook=decode)


class Client:
    def __init__(self, url, timeout=45):
        if not url.startswith(("http://127.0.0.1:", "http://localhost:")):
            raise ValueError("Use a loopback URL via SSH local forwarding")
        self.url, self.timeout = url.rstrip("/"), timeout

    def call(self, method, **payload):
        request = Request(
            self.url + "/" + method,
            data=pack(payload),
            headers={"Content-Type": "application/msgpack"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return unpack(response.read(MAX_MESSAGE_BYTES + 1))
        except HTTPError as exc:
            raise RuntimeError(unpack(exc.read(MAX_MESSAGE_BYTES + 1)).get("error", str(exc))) from exc


def make_server(adapter, port=8766):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_MESSAGE_BYTES:
                    raise ValueError("Invalid request size")
                payload = unpack(self.rfile.read(size))
                method = self.path.removeprefix("/")
                if method == "event":
                    if payload["key"] in ("enter", "space"):
                        adapter.review.operator_key(payload["key"])
                    else:
                        adapter.events.put(payload["key"])
                    if payload["key"] == "x":
                        adapter.review.abort()
                        adapter.stop()
                    result = {"ok": True}
                elif method == "stop":
                    adapter.stop()
                    result = {"ok": True}
                elif method == "result":
                    result = adapter.execution_result(payload["decision_id"])
                else:
                    with adapter.lock:
                        if method == "health":
                            result = {
                                "robot_config_sha256": adapter.config_sha256,
                                "robot_sn": adapter.cfg["robot_sn"],
                                "allow_motion": adapter.hardware.allow_motion,
                                "status": adapter.hardware.status(),
                            }
                        elif method == "review_configure":
                            result = adapter.review.configure()
                        elif method == "review_begin":
                            result = adapter.review.begin(payload["kind"], payload["decision_id"])
                        elif method == "review_poll":
                            result = adapter.review.poll(payload["token"])
                        elif method == "review_resume":
                            result = adapter.review.resume_attempt(payload["token"])
                        elif method == "observe":
                            if adapter.pending is not None:
                                raise RuntimeError("Cannot replace an observation while an action is pending")
                            result = adapter.observe()
                        elif method == "reset":
                            result = adapter.reset()
                        elif method == "choose":
                            selected, human, aligned = adapter.choose_action(
                                {"decision_id": payload["decision_id"]},
                                payload["proposal"],
                                demonstration=payload.get("demonstration", False),
                            )
                            result = {"selected": selected, "human": human, "aligned": aligned}
                        elif method == "execute":
                            result = adapter.execute(payload["selected"], payload.get("decision_id"))
                        elif method == "stop":
                            adapter.stop()
                            result = {"ok": True}
                        else:
                            raise ValueError("Unknown method")
                body, status = pack(result), 200
            except Exception as exc:
                if self.path in ("/reset", "/choose", "/execute", "/review_resume", "/review_poll"):
                    adapter.stop()
                body, status = pack({"error": f"{type(exc).__name__}: {exc}"}), 400
            self.send_response(status)
            self.send_header("Content-Type", "application/msgpack")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                adapter.stop()

        def log_message(self, fmt, *args):
            if args and str(args[1]) != "200":
                print(json.dumps({"http": fmt % args}), flush=True)

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
