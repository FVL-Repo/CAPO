# Training design

The pi_05 VLM and base expert weights are frozen. Expert attention/MLP LoRA and configured action/time projections are trainable. Trainable storage and Adam moments use FP32; model matrix operations use the configured precision.

The critic directly predicts an ensemble of Q values. Each camera passes through a shared frozen ResNet-10 encoder and its own trainable spatial pooling layer; visual features, projected proprioception, and masked executed action chunks feed independent Q heads. There is no V/A decomposition.

CVL draws one batch from demo/exec/inter buffers in a strict 40/40/20 ratio. Every sampled executed transition contributes bootstrapped TD, including interventions drawn from all three buffers. Only the inter subset contributes per-head squared hinge ranking on same-state corrected/rejected action pairs. TD is averaged over the whole global batch; ranking over the global inter subset. Rejected proposals have no observed transition and never receive TD. Duration-aware discounting uses the number of executed primitive commands. True termination stops bootstrapping; truncation retains it when the successor is valid.

Actor updates draw one shared 40/40/20 batch. APO uses its demo/exec states (50/50 within that subset) and uses a frozen actor reference, rolls out its flow, computes terminal Q action gradients, and propagates discrete adjoints through the reference velocity field. The current expert matches corrective velocity targets on sampled flow intervals. Unexecuted/inactive action coordinates have zero terminal critic gradient. BC uses only the sampled demo-buffer segments (offline experts and online human interventions), with the existing executed-action masks and flow-matching loss. It never draws an independent BC batch or imitates autonomous/rejected actions. BC is averaged over the global sampled demo subset; APO over its global demo/exec subset, even when a rank or microbatch has no eligible samples. BC eligibility follows the buffer selected for each draw: a correction drawn through exec/inter is excluded from BC even if that record also belongs to demo.

The default YAML exposes training settings that should be tuned and reported for an experiment: demo/exec/inter buffer sampling fractions, BC loss weight, actor update budgets, reference refresh interval, and hardware reward rules.

All models use 224 × 224 image inputs. The upstream encoder's original 128-pixel preprocessing is a default rather than a constraint on pretrained convolution/GroupNorm weights. At 224 pixels, the encoder produces a 7 × 7 feature map; the critic's trainable spatial kernel is sized accordingly. The reward classifier averages spatially and keeps its original head feature dimension. Older critic/reward checkpoints are rejected through architecture identifiers.


## Continuous collection and optimizer schedule

The collector continuously writes immutable replay shards while a persistent learner consumes them. Episode boundaries reset the environment and produce statistics; they do not gate learning. Initial training waits until each positive-weight replay buffer has data. The learner refreshes its replay view between optimizer updates.

Run exactly `critic_warmup_steps=200` critic-only updates once. Then repeat one critic-only update followed by a joint slot containing one critic update and one actor update (critic parameters are frozen for the actor computation). Thus N actor updates correspond to 200 + 2N critic updates. `max_actor_steps=10000` is the overall stopping budget, not a per-episode quota.

The first actor update creates a frozen current-actor reference. After every `reference_refresh_interval=500` completed actor updates, copy the just-updated actor into the reference and reset the interval counter. Critic updates and episode boundaries never refresh it. The checkpoint preserves the exact optimizer substep, warmup progress, reference weights, actor count at the last refresh, optimizers and per-rank RNG; resuming between either critic update and the actor update does not repeat work.

Policy publication is independent of reference refresh: by default the learner publishes an immutable actor checkpoint every 100 actor updates. The collector reads its atomic pointer and reloads between action chunks. `policy_publish_interval` can be tuned independently. Online workflow budgets may be increased to continue a completed run; other identity changes require a new output directory. Old episode-round RL checkpoints require actor-only initialization into a new run, since their scheduling state is incompatible.
