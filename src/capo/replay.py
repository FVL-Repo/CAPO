from pathlib import Path
import random
import warnings
from collections import OrderedDict

from capo.identity import file_sha256, check_spec, canonical

import torch

from capo.batching import stack_trees, tree_map

SOURCES = ("demonstration", "correction", "autonomous")


def validate_record(record, cfg):
    required = {"obs", "next_obs", "actions", "reward", "terminated", "truncated", "duration", "source"}
    if required - record.keys():
        raise ValueError(f"Missing transition fields: {required - record.keys()}")
    if record["source"] not in SOURCES:
        raise ValueError("Invalid source")
    duration = int(record["duration"])
    if not 1 <= duration <= cfg.execution_horizon:
        raise ValueError("duration must be the actual number of executed primitive steps")
    actions = record["actions"]
    if actions.shape != (cfg.action_horizon, cfg.action_dim) or not torch.isfinite(actions).all():
        raise ValueError("actions must be finite, normalized [action_horizon, action_dim]")
    for key in ("obs", "next_obs"):
        obs = record[key]
        if obs["state"].shape != (cfg.state_dim,) or not torch.isfinite(obs["state"]).all():
            raise ValueError("Invalid state")
        if cfg.backend == "pi05":
            if set(obs["images"]) != set(cfg.image_keys):
                raise ValueError("Replay camera keys must match config")
            if obs["tokenized_prompt"].shape != (cfg.max_token_len,):
                raise ValueError("Invalid prompt length")
            if obs["tokenized_prompt_mask"].shape != (cfg.max_token_len,):
                raise ValueError("Invalid prompt mask")
            for camera in cfg.image_keys:
                image = obs["images"][camera]
                if image.ndim != 3 or (image.shape[0] != 3 and image.shape[-1] != 3):
                    raise ValueError("Expected RGB CHW or HWC images")
                if image.dtype != torch.uint8 and (
                    not torch.isfinite(image).all() or image.abs().max() > 1.001
                ):
                    raise ValueError("Float images must be normalized to [-1,1]")
    if not torch.isfinite(torch.as_tensor(record["reward"])):
        raise ValueError("Nonfinite reward")
    if "rejected_actions" in record:
        if record["source"] != "correction":
            raise ValueError("Only corrections can supply an aligned rejected proposal")
        if not record.get("pair_aligned", False):
            raise ValueError("Explicit same-state, same-horizon alignment is required for preferences")
        rejected = record["rejected_actions"]
        if rejected.shape != actions.shape or not torch.isfinite(rejected).all():
            raise ValueError("Invalid rejected proposal")


def tensor_record(record, cfg):
    duration = int(record["duration"])
    mask = torch.zeros(cfg.action_horizon, cfg.action_dim, dtype=torch.bool)
    mask[:duration, : cfg.robot_action_dim] = True
    mask[:, list(cfg.inactive_action_dims)] = False
    return {
        "obs": record["obs"],
        "next_obs": record["next_obs"],
        "actions": record["actions"].float(),
        "mask": mask,
        "reward": torch.tensor(float(record["reward"])),
        "terminated": torch.tensor(bool(record["terminated"])),
        "truncated": torch.tensor(bool(record["truncated"])),
        "duration": torch.tensor(duration),
        "rejected_actions": record.get("rejected_actions", torch.zeros_like(record["actions"])).float(),
    }


class RecordStore:
    """Lazy immutable disk records and owned in-memory records for small callers."""

    def __init__(self, cache_size):
        self.entries = []
        self.cache = OrderedDict()
        self.cache_size = cache_size

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        entry = self.entries[index]
        if "memory" in entry:
            return entry["memory"]
        path = entry["path"]
        if path not in self.cache:
            self.cache[path] = torch.load(path, map_location="cpu", weights_only=True)["records"]
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(path)
        return self.cache[path][entry["offset"]]


class Replay:
    def __init__(self, cfg):
        self.cfg = cfg
        self.records = RecordStore(cfg.replay_cache_shards)
        self.manifest = []
        self.semantic_metadata = None
        self.groups = {s: [] for s in SOURCES}
        self.pairs = []
        self.loaded = set()
        self.rng = random.Random(cfg.seed)
        self.warned = set()

    def add(self, record, *, location=None):
        validate_record(record, self.cfg)
        metadata = record.get("metadata", {})
        semantic = {
            k: metadata.get(k)
            for k in (
                "robot_config_sha256",
                "norm_stats_sha256",
                "normalization_mode",
                "action_format",
                "state_format",
                "task",
            )
        }
        if self.semantic_metadata is None:
            self.semantic_metadata = semantic
        elif canonical(self.semantic_metadata) != canonical(semantic):
            raise ValueError("Conflicting record semantic metadata")
        if self.cfg.data_spec is not None:
            check_spec(self.cfg, metadata.get("data_spec"))
        if location is None:
            record = tree_map(
                lambda t: t.detach().cpu().clone() if isinstance(t, torch.Tensor) else t, record
            )
            location = {"memory": record}
        location = {**location, "source": record["source"], "pair": "rejected_actions" in record}
        i = len(self.records)
        self.records.entries.append(location)
        self.groups[record["source"]].append(i)
        if "rejected_actions" in record:
            self.pairs.append(i)
        if len(self.records) > self.cfg.replay_capacity:
            weights = dict(zip(SOURCES, self.cfg.replay_mixture))
            eligible = [s for s in SOURCES if len(self.groups[s]) > 1]
            source = max(
                eligible or [record["source"]],
                key=lambda s: len(self.groups[s]) - self.cfg.replay_capacity * weights[s],
            )
            self.records.entries.pop(self.groups[source][0])
            self._reindex()

    def _reindex(self):
        self.groups = {s: [] for s in SOURCES}
        self.pairs = []
        for i, entry in enumerate(self.records.entries):
            self.groups[entry["source"]].append(i)
            if entry["pair"]:
                self.pairs.append(i)

    def snapshot(self):
        return {
            "manifest": self.manifest,
            "entries": self.records.entries,
            "semantic_metadata": self.semantic_metadata,
        }

    def restore_snapshot(self, state):
        for item in state["manifest"]:
            if file_sha256(item["path"]) != item["sha256"]:
                raise ValueError(f"Replay shard changed since checkpoint: {item['path']}")
        self.manifest = state["manifest"]
        self.loaded = {item["path"] for item in self.manifest}
        self.records.entries = state["entries"]
        self.records.cache.clear()
        self.semantic_metadata = state["semantic_metadata"]
        self._reindex()

    def refresh(self, directory):
        from capo import distributed as parallel
        import torch.distributed as dist

        paths = sorted(Path(directory).glob("*.pt")) if parallel.rank() == 0 else None
        if parallel.active():
            message = [paths]
            dist.broadcast_object_list(message, src=0)
            paths = message[0]
        for path in paths:
            if str(path) in self.loaded:
                continue
            shard = torch.load(path, map_location="cpu", weights_only=True)
            if shard.get("format_version") != 1:
                raise ValueError(f"Unsupported replay format in {path}")
            check_spec(self.cfg, shard.get("data_spec"))
            metadata = shard.get("metadata", {})
            for key in ("action_horizon", "execution_horizon", "action_dim", "robot_action_dim", "gamma"):
                if metadata.get(key) != getattr(self.cfg, key):
                    raise ValueError(f"Replay/config mismatch for {key} in {path}")
            digest = file_sha256(path)
            for offset, record in enumerate(shard["records"]):
                self.add(record, location={"path": str(path), "offset": offset})
            self.loaded.add(str(path))
            self.manifest.append({"path": str(path), "sha256": digest})

    def sample(self, count, view="exec", *, cfg=None, start=0, end=None):
        cfg = cfg or self.cfg
        if view == "pair":
            if not self.pairs:
                return None
            indices = self.rng.choices(self.pairs, k=count)
        else:
            weights = (
                list(cfg.replay_mixture)
                if view == "exec"
                else [1 - cfg.bc_correction_fraction, cfg.bc_correction_fraction, 0]
            )
            available = [w if self.groups[s] else 0 for s, w in zip(SOURCES, weights)]
            if sum(available) == 0:
                raise ValueError(f"No examples with positive sampling weight in replay view {view}")
            if available != weights and view not in self.warned:
                warnings.warn(
                    f"{view}: empty source groups; renormalizing mixture to available data", stacklevel=2
                )
                self.warned.add(view)

            expected = [count * w / sum(available) for w in available]
            counts = [int(v) for v in expected]
            for _ in range(count - sum(counts)):
                fractional = [e - c for e, c in zip(expected, counts)]
                index = self.rng.choices(range(3), weights=[max(0, v) for v in fractional])[0]
                counts[index] += 1
            indices = []
            for source, n in zip(SOURCES, counts):
                indices.extend(self.rng.choices(self.groups[source], k=n))
            self.rng.shuffle(indices)
        return stack_trees([tensor_record(self.records[i], cfg) for i in indices[start:end]])


def save_shard(path, records, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for record in records:
        validate_record(record, cfg)
    shard = {
        "format_version": 1,
        "data_spec": cfg.data_spec,
        "metadata": {
            k: getattr(cfg, k)
            for k in ("action_horizon", "execution_horizon", "action_dim", "robot_action_dim", "gamma")
        },
        "records": records,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(shard, temporary)
    temporary.replace(path)
