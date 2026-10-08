from capo.batching import stack_trees
from capo.demonstrations import DemonstrationDataset
from capo.replay import tensor_record


class ImitationSampler:
    def __init__(self, cfg, replay):
        self.cfg, self.replay = cfg, replay
        self.demonstrations = DemonstrationDataset(cfg) if cfg.demonstrations else None

    @property
    def identity(self):
        return self.demonstrations.identity if self.demonstrations else None

    def sample(self, count, start, end):
        cfg, replay, dataset = self.cfg, self.replay, self.demonstrations
        if dataset is None:
            return replay.sample(count, "bc", cfg=cfg, start=start, end=end)
        fraction = cfg.bc_correction_fraction if replay.groups["correction"] else 0.0
        choices = []

        for _ in range(count):
            correction = replay.rng.random() < fraction
            index = (
                replay.rng.choice(replay.groups["correction"])
                if correction
                else replay.rng.randrange(len(dataset))
            )
            choices.append((correction, index))
        samples = []
        for correction, index in choices[start:end]:
            sample = tensor_record(replay.records[index], cfg) if correction else dataset[index]
            samples.append({k: sample[k] for k in ("obs", "actions", "mask")})
        return stack_trees(samples)
