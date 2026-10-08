"""Atomic, immutable policy publications shared by a learner and collector."""
from dataclasses import asdict
import json
from pathlib import Path
import uuid

import torch

from capo.identity import file_sha256


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def publish_state(actor_state, config, output, actor_step):
    output = Path(output)
    directory = output / 'policies'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'actor_{actor_step:08d}_{uuid.uuid4().hex}.pt'
    artifact = {'format_version': 1, 'config': config, 'actor': actor_state,
                'actor_step': actor_step}
    temporary = path.with_suffix('.tmp')
    torch.save(artifact, temporary)
    temporary.replace(path)
    publication = {'checkpoint': str(path), 'sha256': file_sha256(path), 'actor_step': actor_step}
    atomic_json(output / 'policy.json', publication)
    with (output / 'publications.jsonl').open('a') as log:
        log.write(json.dumps(publication) + '\n')
    return publication


def publish_actor(actor, cfg, output, actor_step):
    return publish_state(actor.trainable_state(), asdict(cfg), output, actor_step)


def publish_checkpoint(checkpoint, output):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    return publish_state(payload['actor'], payload['config'], output, payload.get('actor_step', 0))
