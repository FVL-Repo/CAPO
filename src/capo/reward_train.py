import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from capo.config import local_path
from capo.identity import file_sha256
from capo.reward import SuccessClassifier, load_reward_config


def load_examples(directory, keys):
    images, labels, files = {k: [] for k in keys}, [], []
    for path in sorted(local_path(directory).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            label = np.asarray(data["success"])
            if label.ndim != 1 or not np.isin(label, [0, 1]).all():
                raise ValueError("success must contain per-frame binary labels")
            labels.append(torch.from_numpy(label.astype(np.int64)))
            for key in keys:
                value = data["image/" + key]
                if (
                    value.dtype != np.uint8
                    or value.ndim != 4
                    or value.shape[-1] != 3
                    or len(value) != len(label)
                ):
                    raise ValueError("Reward images must be matching RGB uint8 frames")

                from PIL import Image

                resized = np.stack(
                    [
                        np.asarray(Image.fromarray(frame).resize((224, 224), Image.Resampling.BILINEAR))
                        for frame in value
                    ]
                )
                images[key].append(torch.from_numpy(resized.copy()))
        files.append({"path": str(path), "sha256": file_sha256(path)})
    if not labels:
        raise ValueError("No labeled reward data")
    labels = torch.cat(labels)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Reward dataset must contain both positive and negative frames")
    return {k: torch.cat(v) for k, v in images.items()}, labels, files


def train(cfg):
    output = local_path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "classifier.pt").exists():
        raise FileExistsError("Reward output already exists; choose a new directory")
    torch.manual_seed(cfg["seed"])
    if cfg["steps"] < 1 or cfg["batch_size"] < 2:
        raise ValueError("Reward steps must be positive and batch size at least two")
    images, labels, files = load_examples(cfg["dataset"], cfg["image_keys"])
    validation, targets, heldout = load_examples(cfg["validation"], cfg["image_keys"])
    if {x["sha256"] for x in files} & {x["sha256"] for x in heldout}:
        raise ValueError("Reward train/validation episode overlap")
    model = SuccessClassifier(cfg).to(cfg["device"])
    if cfg.get("init_checkpoint"):
        initial = torch.load(local_path(cfg["init_checkpoint"]), map_location="cpu", weights_only=True)
        if initial.get("architecture") != SuccessClassifier.architecture:
            raise ValueError("Reward fine-tuning requires a 224x224 checkpoint")
        for key in ("image_keys", "encoder", "hidden_dim", "dropout"):
            if initial["config"][key] != cfg[key]:
                raise ValueError(f"Reward fine-tuning architecture mismatch: {key}")
        model.load_state_dict(initial["model"], strict=True)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=cfg["learning_rate"])
    groups = [(labels == k).nonzero().flatten() for k in (0, 1)]
    for _ in range(cfg["steps"]):
        indices = torch.cat([group[torch.randint(len(group), (cfg["batch_size"] // 2,))] for group in groups])
        optimizer.zero_grad(set_to_none=True)
        logits = model({k: v[indices].to(cfg["device"]) for k, v in images.items()})
        loss = F.cross_entropy(logits, labels[indices].to(cfg["device"]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), 10, error_if_nonfinite=True)
        optimizer.step()
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(targets), cfg["batch_size"]):
            logits = model(
                {k: v[start : start + cfg["batch_size"]].to(cfg["device"]) for k, v in validation.items()}
            )
            predictions.append((logits.softmax(-1)[:, 1] >= cfg["threshold"]).cpu())
    prediction = torch.cat(predictions)
    metrics = {
        "loss": loss.item(),
        "validation_accuracy": (prediction == targets).float().mean().item(),
        "false_positive_rate": prediction[targets == 0].float().mean().item(),
        "false_negative_rate": (~prediction[targets == 1]).float().mean().item(),
    }
    torch.save(
        {"architecture": SuccessClassifier.architecture, "config": cfg, "model": model.state_dict()},
        output / "classifier.pt",
    )
    (output / "metrics.json").write_text(
        json.dumps({**metrics, "train": files, "validation": heldout}, indent=2) + "\n"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(train(load_reward_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
