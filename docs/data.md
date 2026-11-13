# Data formats

All camera arrays are RGB uint8. Raw images can have different resolutions; preprocessing produces 224 × 224 model inputs. Camera names must match the YAML configuration. State/action conventions and task prompts must remain identical across demonstration preparation and online execution.

## Offline demonstrations

Place NPZ episodes in the dataset YAML's `input` directory. SFT alone can use observation/action-only data. RL now requires transition-complete expert chunks so the demo buffer can participate in TD.

- `state`: `[N, D_state]`, physical units before normalization.
- `actions`: `[N, D_action]` for synchronized primitive sequences, or `[N, H, D_action]` for observed command chunks.
- `lengths`: `[N]` valid lengths when storing chunks.
- `image/<camera_key>`: `[N, height, width, 3]`, aligned with each state.

For TD-capable demonstrations, use chunked `actions` and additionally store:

- `next_state`: `[N, D_state]`, the observed successor after the executed chunk.
- `next_image/<camera_key>`: corresponding successor images.
- `rewards`: `[N, H]`, one real reward per executed primitive action (only `lengths[i]` values are used).
- `terminated`, `truncated`: `[N]` real boundary flags.
- Optional `applied_actions`: `[N, H, D_action]` for execution diagnostics.

The adapter collection command writes these fields, including short chunks and delayed boundary reward overrides. Preparation stores both supervised samples and normalized TD transitions; the manifest binds the discount and action horizons. Legacy observation/action-only datasets remain valid for SFT, but RL rejects them instead of fabricating rewards or successors. Preparation preserves chunk boundaries and masks padded/inactive action coordinates. Missing policy camera keys are masked; configured reward cameras must be present.

`prepare` writes immutable sample shards and a manifest identifying source files, preprocessing, normalization, and tokenizer. Rebuild into a new directory when any of these change. SFT uses the sample dataset directly. RL imports the observed expert transitions into demo; online BC selects from the shared replay batch.

## Reward labels

Place NPZ episodes into separate training and validation directories:

- `success`: `[N]`, integer values 0 or 1.
- `image/<camera_key>`: `[N, height, width, 3]` for every configured reward camera.

Each split must include both classes. Split by episode, not adjacent frames. Interactive review appends labeled single-frame files to the training directory; it does not change validation data.

## Online replay

The collector writes schema-checked transition shards containing the current observation, selected commands, rewards, duration, true successor, terminal/truncation flags, source, and action masks. Applied actions and command IDs remain available for diagnostics. An aligned human correction additionally stores the rejected policy proposal from the same observation.

The replay has exactly three logical buffers, with independent FIFO capacities (`replay_capacity` per buffer). Membership indices share immutable transition storage rather than copying image tensors:

| Transition | demo | exec | inter |
|---|---|---|---|
| Pre-collected expert execution | yes | no | no |
| Online policy execution | no | yes | no |
| Online human intervention without aligned proposal | yes | yes | no |
| Online human intervention with a same-state, same-horizon rejected proposal | yes | yes | yes |

CVL samples **demo / exec / inter = 40 / 40 / 20** by buffer, not by mutually exclusive source label. TD uses **all sampled demo and exec transitions**, plus **only the executed corrective side of sampled inter pairs**. This includes human corrections sampled from any of the three buffers. Only pairs selected through inter enter ranking; a paired correction selected through demo or exec contributes TD alone. The rejected unexecuted proposal never enters TD.

Each CVL global batch must realize the ratio exactly (a multiple of 5 for 40/40/20, also divisible by GPU world size). Defaults use 240 samples. Positive-weight empty buffers raise an error; there is no silent renormalization. Actor updates also draw one 40/40/20 batch. APO uses only its demo/exec states (50/50 within that subset). BC uses **only sampled demo-buffer segments**, including offline experts and online human interventions. It uses the executed actions and their masks, never rejected actions or autonomous exec actions. There is no separate BC draw, resampling, or extra expert/correction quota. A correction sampled through exec/inter is excluded from BC even when the same record also belongs to demo.

Ranking is averaged over the global inter subset, BC over the global sampled demo subset, and APO over the global demo/exec subset, including when individual ranks or microbatches contain no eligible samples. Logs record each buffer's TD, actor, BC, and APO sample counts, correction counts, and number of ranking pairs used. New replay snapshots preserve the three independent memberships; old source-mixture snapshots cannot be exactly resumed. The demo-only BC objective has checkpoint/workflow identity `three_buffer_demo_bc_v1`: checkpoints using independent BC or demo+exec+inter BC cannot be exactly resumed under this objective. Remove `bc_correction_fraction` from legacy configs; `bc_weight` controls the BC loss scale, and the ablation generator now compares ranking/BC on versus off. Use `--actor-init` and a new output directory to retain old actor weights while starting the new replay/critic training.

This demo/exec routing matches [HIL-SERL's official implementation](https://github.com/rail-berkeley/hil-serl/blob/main/examples/train_rlpd.py), which inserts interventions into both replay and demo and uses two-buffer 50/50 sampling. Inter, same-state ranking, and 40/40/20 are CAPO extensions.


Online collection is continuous: episode endings do not trigger or block learning. Each finalized shard retains the policy version used for its own action proposal, including when the collector reloads a publication during an episode. On stop, the last pending executed transition is flushed as truncated. The persistent learner ingests new immutable shards between optimizer updates; training scheduling and reference refresh are based on optimizer counters, not episode counts.
