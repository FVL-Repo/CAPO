# Robot integration

`configs/robots/flexiv.yaml` contains example values, not a calibrated robot installation. Fill in robot and camera serial numbers, the robot-facing network interface, camera crops, workspace, reset joints, tool conventions, and SpaceMouse mapping. Keep an identical file on the robot host and training host; the client checks its digest.

The Flexiv server uses Python 3.10 or newer with `requirements-flexiv.txt`. Install the requirements on the robot host and copy this project's `src/capo` and configuration files there. SDK distributions are obtained from their vendors/PyPI rather than bundled here.

```bash
python -m pip install -r requirements-flexiv.txt
PYTHONPATH=src python -m capo.robots.flexiv_server --config configs/robots/flexiv.yaml --inventory
PYTHONPATH=src python -m capo.robots.flexiv_server --config configs/robots/flexiv.yaml --probe
```

After calibrating the robot installation, the hardware server needs `--allow-motion --allow-reset` for the interactive collection/reset workflow. Without those flags it only reads telemetry. The adapter preserves the gripper width; coordinate 6 must remain masked for this configuration.

The server binds to loopback on port 8766. Use an SSH tunnel from the training host, substituting your own account and host:

```bash
ssh -N -L 8766:127.0.0.1:8766 USER@ROBOT_HOST
export CAPO_FLEXIV_URL=http://127.0.0.1:8766
```

Keep the server terminal focused for Space/Enter review controls. `h` switches to human control, `p` returns to policy control, and `x` aborts. Collection begins through the dataset/online CLI on the training host.

## Custom adapters

Set the YAML `adapter` field to a factory `package.module:create(cfg)`. Follow `src/capo/robots/interface.py` for the adapter protocol and `flexiv_client.py` for the remote implementation:

- `reset()` / `observe()` return physical state, RGB images, task prompt, decision ID, and coordinate metadata.
- `choose_action(observation, proposal)` returns selected commands, human-control status, and same-state alignment status.
- `execute(selected)` returns commands, applied actions, per-step rewards, true successor, human step count, and termination/truncation status.
- `close()` releases resources.

Adapters used for demonstrations expose `demonstration` mode. Interactive reward adapters additionally implement review configuration, stop/review/resume, and reset gating as shown in `flexiv_client.py`. Do not retry actuation after an ambiguous network failure; reconcile its journal entry first.
