import json
import os
import uuid

import numpy as np

from capo.config import local_path


def save_feedback(cfg, observation, *, success, probability, decision_id):
    directory = local_path(cfg["dataset"])
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {"success": np.array([int(success)], dtype=np.int64)}
    for key in cfg["image_keys"]:
        image = np.asarray(observation["images"][key])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Hard example needs the exact RGB inference frame: {key}")
        arrays["image/" + key] = image[None]
    arrays["metadata"] = np.asarray(
        json.dumps(
            {
                "kind": "hard_positive" if success else "hard_negative",
                "decision_id": decision_id,
                "probability": probability,
                "checkpoint": cfg["checkpoint"],
                "label_source": "operator",
                "robot_metadata": observation.get("robot_metadata", {}),
            }
        )
    )
    path = directory / f"hard_{'positive' if success else 'negative'}_{uuid.uuid4().hex}.npz"
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    print(f"Saved hard {'positive' if success else 'negative'} example: {path}", flush=True)
    return str(path)
