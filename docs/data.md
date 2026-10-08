# Data formats

All camera arrays are RGB uint8. Raw images can have different resolutions; preprocessing produces 224 × 224 model inputs. Camera names must match the YAML configuration. State/action conventions and task prompts must remain identical across demonstration preparation and online execution.

## Offline demonstrations

Place NPZ episodes in the dataset YAML's `input` directory. No rewards or next observations are needed.

- `state`: `[N, D_state]`, physical units before normalization.
- `actions`: `[N, D_action]` for synchronized primitive sequences, or `[N, H, D_action]` for observed command chunks.
- `lengths`: `[N]` valid lengths when storing chunks.
- `image/<camera_key>`: `[N, height, width, 3]`, aligned with each state.

The adapter collection command writes this format. Preparation preserves chunk boundaries and masks padded/inactive action coordinates. Missing policy camera keys are masked; configured reward cameras must be present.

`prepare` writes immutable sample shards and a manifest identifying source files, preprocessing, normalization, and tokenizer. Rebuild into a new directory when any of these change. SFT and online BC use this dataset directly.

## Reward labels

Place NPZ episodes into separate training and validation directories:

- `success`: `[N]`, integer values 0 or 1.
- `image/<camera_key>`: `[N, height, width, 3]` for every configured reward camera.

Each split must include both classes. Split by episode, not adjacent frames. Interactive review appends labeled single-frame files to the training directory; it does not change validation data.

## Online replay

The collector writes schema-checked transition shards containing the current observation, selected commands, rewards, duration, true successor, terminal/truncation flags, source, and action masks. Applied actions and command IDs remain available for diagnostics. An aligned human correction additionally stores the rejected policy proposal from the same observation.

The in-memory replay maintains demonstration/correction/autonomous source views and a paired ranking view; these are indices over transitions. The offline SFT dataset is separate. Sampling weights are renormalized over available groups. BC samples offline experts and eligible corrections; TD samples transitions; ranking samples only aligned correction/proposal pairs.
