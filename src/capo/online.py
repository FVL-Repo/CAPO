"""Continuous collection and a persistent learner; no episode-based learning rounds."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

from capo.config import Config, ROOT, local_path
from capo.identity import file_sha256, canonical
from capo.processing import Processor
from capo.replay import REPLAY_LAYOUT
from capo.imitation import TRAINING_OBJECTIVE
from capo.schedule import TRAINING_SCHEDULE, training_complete
from capo.exchange import atomic_json, publish_checkpoint


def stop_child(process, stop_file):
    if process is None or process.poll() is not None:
        return
    stop_file.touch()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run(cfg, **kwargs):
    # A second launcher must never create another collector for the same robot/run.
    import fcntl
    output = local_path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'online.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(cfg, **kwargs)


def _run(cfg, *, norm_stats, adapter, actor_init=None, reward_config=None,
         train_processes=1, collector_device='cuda:0', checkpoint_interval=100):
    cfg.validate()
    if train_processes < 1 or checkpoint_interval < 1:
        raise ValueError('Invalid process/checkpoint settings')
    output = local_path(cfg.output)
    processor = Processor(cfg, norm_stats,
                          quantile=(cfg.data_spec or {}).get('normalization_mode', 'quantile') == 'quantile')
    del processor
    digest = file_sha256(local_path(cfg.checkpoint))
    if cfg.base_sha256 and cfg.base_sha256 != digest:
        raise ValueError('Base checkpoint changed')
    cfg.base_sha256 = digest
    reward_identity = None
    if reward_config:
        from capo.reward import load_reward_config
        reward = load_reward_config(reward_config)
        reward_identity = {'config': reward,
                           'sha256': file_sha256(local_path(reward['checkpoint'])) if reward.get('enabled') else None}
    identity = json.loads(canonical({
        'mode': 'continuous_online_v1', 'replay_layout': REPLAY_LAYOUT,
        'training_objective': TRAINING_OBJECTIVE, 'training_schedule': TRAINING_SCHEDULE,
        'config': cfg.__dict__, 'norm_stats': file_sha256(local_path(norm_stats)),
        'adapter': adapter, 'train_processes': train_processes, 'collector_device': collector_device,
        'reward': reward_identity, 'checkpoint_interval': checkpoint_interval,
        'actor_init': str(local_path(actor_init)) if actor_init else None,
        'actor_init_sha256': file_sha256(local_path(actor_init)) if actor_init else None,
    }))
    state_path = output / 'online_state.json'
    target = output / 'latest.pt'
    state = {'identity': identity, 'phase': 'starting'}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        previous = json.loads(canonical(state['identity']))
        current = json.loads(canonical(identity))
        old_budget = previous['config'].pop('max_actor_steps', None)
        new_budget = current['config'].pop('max_actor_steps')
        if previous != current or old_budget is None or new_budget < old_budget:
            raise ValueError('Online run configuration changed; only increasing max_actor_steps is allowed')
        if state['phase'] == 'complete' and old_budget == new_budget:
            payload = torch.load(target, map_location='cpu', weights_only=True)
            if not training_complete(payload, cfg):
                raise ValueError('Completed learner checkpoint does not match requested schedule')
            if file_sha256(state['policy']) != state['policy_sha256']:
                raise ValueError('Published policy changed')
            return state
    state.update(identity=identity, phase='starting', training_checkpoint=str(target))
    atomic_json(state_path, state)
    initial = target if target.exists() else (local_path(actor_init) if actor_init else None)
    if initial:
        # Restore the collection policy from the same checkpoint as the learner.
        from capo.checkpoints import check_actor_config
        from types import SimpleNamespace
        payload = torch.load(initial, map_location='cpu', weights_only=True)
        check_actor_config(SimpleNamespace(cfg=cfg), payload['config'])
        if target.exists() and payload.get('training_schedule') != TRAINING_SCHEDULE:
            raise ValueError('Existing learner checkpoint uses a different schedule')
        publication = publish_checkpoint(initial, output)
        del payload
    else:
        # External full SFT bases are supported without an actor delta checkpoint.
        from capo.backbone import make_backbone
        from capo.exchange import publish_actor
        base_actor = make_backbone(replace(cfg, device='cpu'))
        publication = publish_actor(base_actor, cfg, output, 0)
        del base_actor
    state.update(policy=publication['checkpoint'], policy_sha256=publication['sha256'])
    cfg.save(output / 'train_config.json')
    replace(cfg, device=collector_device).save(output / 'collect_config.json')
    stop_collection, stop_training = output / 'STOP_COLLECTION', output / 'STOP_TRAINING'
    stop_collection.unlink(missing_ok=True)
    stop_training.unlink(missing_ok=True)
    environment = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    collect = [sys.executable, '-m', 'capo.collect', '--config', str(output / 'collect_config.json'),
               '--norm-stats', str(norm_stats), '--adapter', adapter, '--continuous',
               '--policy-pointer', str(output / 'policy.json'), '--stop-file', str(stop_collection)]
    if (cfg.data_spec or {}).get('normalization_mode') == 'mean_std':
        collect.append('--mean-std')
    if reward_config:
        collect += ['--reward-config', str(reward_config)]
    train = [sys.executable, '-m']
    if train_processes > 1:
        from capo.distributed import local_launch_args
        train += ['torch.distributed.run', *local_launch_args(train_processes), '-m']
    train += ['capo.train', '--config', str(output / 'train_config.json'), '--streaming',
              '--save-every', str(checkpoint_interval), '--publish-dir', str(output),
              '--stop-file', str(stop_training)]
    if initial:
        train += ['--resume' if target.exists() else '--actor-init', str(initial)]
    collector = learner = None
    try:
        with (output / 'collector.log').open('a') as c_log, (output / 'learner.log').open('a') as l_log:
            collector = subprocess.Popen(collect, cwd=ROOT, env=environment, stdout=c_log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            learner = subprocess.Popen(train, cwd=ROOT, env=environment, stdout=l_log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            state.update(phase='running')
            atomic_json(state_path, state)
            while learner.poll() is None:
                if collector.poll() is not None:
                    raise RuntimeError(f'Continuous collector exited early ({collector.returncode}); see collector.log')
                time.sleep(.5)
            if learner.returncode:
                raise RuntimeError(f'Learner failed ({learner.returncode}); see learner.log')
            stop_child(collector, stop_collection)
            if collector.returncode:
                raise RuntimeError(f'Collector failed during shutdown ({collector.returncode})')
        payload = torch.load(target, map_location='cpu', weights_only=True)
        if not training_complete(payload, cfg):
            raise RuntimeError('Learner stopped before the requested actor update budget')
        publication = json.loads((output / 'policy.json').read_text())
        if publication['actor_step'] != cfg.max_actor_steps or file_sha256(publication['checkpoint']) != publication['sha256']:
            raise ValueError('Final publication is incomplete or corrupt')
        state.update(phase='complete', actor_step=payload['actor_step'], critic_step=payload['critic_step'],
                     reference_actor_step=payload['reference_actor_step'],
                     reference_refresh_count=payload['reference_refresh_count'],
                     policy=publication['checkpoint'], policy_sha256=publication['sha256'])
        atomic_json(state_path, state)
        return state
    except BaseException as error:
        state.update(phase='interrupted', error=type(error).__name__ + ': ' + str(error))
        atomic_json(state_path, state)
        raise
    finally:
        # Request both stops before waiting for either child to finish.
        stop_collection.touch()
        stop_training.touch()
        stop_child(collector, stop_collection)
        stop_child(learner, stop_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--norm-stats', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--reward-config')
    parser.add_argument('--actor-init')
    parser.add_argument('--train-processes', type=int, default=1)
    parser.add_argument('--collector-device', default='cuda:0')
    parser.add_argument('--checkpoint-interval', type=int, default=100)
    args = parser.parse_args()
    cfg = Config.load(local_path(args.config))
    if args.adapter == 'capo.robots.flexiv_client:create':
        from capo.identity import bind_robot
        from capo.robots.flexiv_config import load_config
        bind_robot(cfg, load_config(local_path(os.environ.get('CAPO_FLEXIV_CONFIG', 'configs/robots/flexiv.yaml'))))
    print(json.dumps(run(cfg, **{k:v for k,v in vars(args).items() if k != 'config'}), indent=2))


if __name__ == '__main__':
    main()
