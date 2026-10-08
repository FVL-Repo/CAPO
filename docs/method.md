# Training design

The pi_05 VLM and base expert weights are frozen. Expert attention/MLP LoRA and configured action/time projections are trainable. Trainable storage and Adam moments use FP32; model matrix operations use the configured precision.

The critic directly predicts an ensemble of Q values. Each camera passes through a shared frozen ResNet-10 encoder and its own trainable spatial pooling layer; visual features, projected proprioception, and masked executed action chunks feed independent Q heads. There is no V/A decomposition.

CVL combines a bootstrapped TD loss with a per-head squared hinge ranking loss on aligned human corrections and rejected policy proposals. Duration-aware discounting uses the number of executed primitive commands. True termination stops bootstrapping; truncation retains it when the successor is valid.

APO snapshots the actor at the phase boundary, rolls out its flow, computes terminal Q action gradients, and propagates discrete adjoints through the reference velocity field. The current expert matches corrective velocity targets on sampled flow intervals. Unexecuted/inactive action coordinates have zero terminal critic gradient. BC combines offline experts and eligible online corrections using flow matching.

The default YAML exposes training settings that should be tuned and reported for an experiment: source sampling fractions, expert/correction BC fraction, training rounds, update budgets, and hardware reward rules.

All models use 224 × 224 image inputs. The upstream encoder's original 128-pixel preprocessing is a default rather than a constraint on pretrained convolution/GroupNorm weights. At 224 pixels, the encoder produces a 7 × 7 feature map; the critic's trainable spatial kernel is sized accordingly. The reward classifier averages spatially and keeps its original head feature dimension. Older critic/reward checkpoints are rejected through architecture identifiers.
