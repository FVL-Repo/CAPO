from pathlib import Path
import random
import math
from collections import OrderedDict

from capo.identity import file_sha256, check_spec, canonical

import torch

from capo.batching import stack_trees, tree_map
from capo.imitation import bc_eligible

SOURCES = ("demonstration", "correction", "autonomous")
BUFFERS = ("demo", "exec", "inter")
REPLAY_LAYOUT = "demo_exec_inter_v1"


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
            self.cache[path] = torch.load(path, map_location="cpu", weights_only=True)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(path)
        return self.cache[path][entry.get("record_key", "records")][entry["offset"]]


class Replay:
    def __init__(self, cfg):
        self.cfg = cfg
        self.records = RecordStore(cfg.replay_cache_shards)
        self.manifest = []
        self.semantic_metadata = None
        self.groups = {s: [] for s in SOURCES}
        self.buffers = {s: [] for s in BUFFERS}
        self.loaded = set()
        self.rng = random.Random(cfg.seed)

    @property
    def pairs(self):
        return self.buffers["inter"]

    @staticmethod
    def membership(record):
        if record["source"] == "demonstration":
            return ("demo",)
        if record["source"] == "autonomous":
            return ("exec",)
        return ("demo", "exec", "inter") if "rejected_actions" in record else ("demo", "exec")

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
        evicted = False
        for name in self.membership(record):
            self.buffers[name].append(i)
            if len(self.buffers[name]) > self.cfg.replay_capacity:
                self.buffers[name].pop(0)
                evicted = True
        if evicted:
            # Independent FIFO capacities. Shared transitions remain while any buffer owns them.
            retained = sorted(set().union(*map(set, self.buffers.values())))
            remap = {old: new for new, old in enumerate(retained)}
            self.records.entries = [self.records.entries[j] for j in retained]
            self.buffers = {k: [remap[j] for j in ids] for k, ids in self.buffers.items()}
            self._reindex()

    def _reindex(self):
        self.groups = {s: [] for s in SOURCES}
        for i, entry in enumerate(self.records.entries):
            self.groups[entry["source"]].append(i)

    def snapshot(self):
        return {
            "layout": REPLAY_LAYOUT,
            "buffers": {k: list(v) for k, v in self.buffers.items()},
            "manifest": self.manifest,
            "entries": self.records.entries,
            "semantic_metadata": self.semantic_metadata,
        }

    def restore_snapshot(self, state):
        if state.get("layout") != REPLAY_LAYOUT:
            raise ValueError("Legacy source-mixture replay cannot resume as three buffers; use --actor-init")
        for item in state["manifest"]:
            if file_sha256(item["path"]) != item["sha256"]:
                raise ValueError(f"Replay shard changed since checkpoint: {item['path']}")
        self.manifest = state["manifest"]
        self.loaded = {item["path"] for item in self.manifest}
        self.records.entries = state["entries"]
        self.buffers = {k: list(state["buffers"][k]) for k in BUFFERS}
        self.records.cache.clear()
        self.semantic_metadata = state["semantic_metadata"]
        self._reindex()

    def load_demonstrations(self):
        """Load actual offline transitions; BC-only data cannot supply fabricated TD targets."""
        if not self.cfg.demonstrations:
            return
        from capo.config import local_path
        import json

        root = local_path(self.cfg.demonstrations)
        manifest = json.loads((root / "manifest.json").read_text())
        check_spec(self.cfg, manifest["data_spec"])
        if manifest.get("td_transitions", 0) != sum(s["count"] for s in manifest["shards"]):
            raise ValueError("Offline demo buffer needs rewards, next observations and boundaries; "
                             "re-prepare transition-complete demonstrations (legacy BC-only data is insufficient)")
        for key in ("action_horizon", "execution_horizon", "action_dim", "robot_action_dim", "gamma"):
            if manifest.get("transition_metadata", {}).get(key) != getattr(self.cfg, key):
                raise ValueError(f"Offline demo/config mismatch for {key}; re-prepare the dataset")
        for item in manifest["shards"]:
            path = local_path(root / item["file"])
            if str(path) in self.loaded:
                continue
            if file_sha256(path) != item["sha256"]:
                raise ValueError("Offline demonstration shard changed")
            shard = torch.load(path, map_location="cpu", weights_only=True)
            if len(shard.get("transitions", [])) != item["count"]:
                raise ValueError("Offline shard must supply one TD transition per demonstration")
            for offset, record in enumerate(shard["transitions"]):
                if record["source"] != "demonstration":
                    raise ValueError("Offline dataset contains a non-demonstration transition")
                self.add(record, location={"path": str(path), "offset": offset, "record_key": "transitions"})
            self.loaded.add(str(path))
            self.manifest.append({"path": str(path), "sha256": item["sha256"]})

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

    def sample(self, count, view="cvl", *, cfg=None, start=0, end=None):
        cfg = cfg or self.cfg
        if view in ("cvl", "actor"):
            weights = list(cfg.replay_mixture)
        else:
            name = {"bc": "demo", "pair": "inter"}.get(view, view)
            if name not in BUFFERS:
                raise ValueError(f"Unknown replay view: {view}")
            weights = [float(k == name) for k in BUFFERS]
        counts = [round(count * w) for w in weights]
        if sum(counts) != count or any(not math.isclose(count * w, n, abs_tol=1e-8)
                                       for w, n in zip(weights, counts)):
            raise ValueError("Batch size must realize the exact demo/exec/inter ratio; "
                             "use a multiple of 5 (and the GPU world size) for 40/40/20")
        choices = []
        for buffer_id, (name, n) in enumerate(zip(BUFFERS, counts)):
            if n and not self.buffers[name]:
                raise ValueError(f"{name} buffer is empty; collect/import data before sampling (no renormalization)")
            choices.extend((i, buffer_id) for i in self.rng.choices(self.buffers[name], k=n))
        self.rng.shuffle(choices)
        # Compute denominators before rank slicing: subsets can be uneven or empty
        # on a rank. Averaging rank gradients must equal the global subset mean.
        bc_count = sum(bc_eligible(b)
                       for i, b in choices)
        apo_count = sum(b != 2 for _, b in choices)
        samples = []
        for i, buffer_id in choices[start:end]:
            source_id = SOURCES.index(self.records.entries[i]["source"])
            use_bc = bc_eligible(buffer_id)
            sample = tensor_record(self.records[i], cfg)
            sample.update(
                buffer_id=torch.tensor(buffer_id),
                source_id=torch.tensor(source_id),
                bc_mask=torch.tensor(use_bc),
                bc_weight=torch.tensor(count / bc_count if use_bc else 0.0),
                apo_mask=torch.tensor(buffer_id != 2),
                apo_weight=torch.tensor(count / apo_count if buffer_id != 2 else 0.0),
                record_index=torch.tensor(i),
                ranking_mask=torch.tensor(buffer_id == 2),
                ranking_weight=torch.tensor(count / counts[2] if buffer_id == 2 else 0.0),
            )
            samples.append(sample)
        return stack_trees(samples)


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
