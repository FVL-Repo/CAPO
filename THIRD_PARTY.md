# Third-party notices

This repository is an implementation of *CAPO: Real-World Reinforcement Learning for VLA via Corrective Adjoint Policy Optimization*. It includes only the upstream code required by the runtime/conversion path. Reference repositories, datasets, model weights, and local experiment results are not distributed.

| Component | Source | Terms / retained notice |
| --- | --- | --- |
| OpenPI model, normalization, converter, client and serving subset | https://github.com/Physical-Intelligence/openpi | `third_party/openpi/LICENSE`, `third_party/openpi/LICENSE_GEMMA.txt` |
| OpenPI replacements for Transformers Gemma/PaliGemma/SigLIP | OpenPI above; https://github.com/huggingface/transformers | Upstream headers and `third_party/licenses/TRANSFORMERS_LICENSE` |

The vendored OpenPI subset has local modifications for image-key configuration, cached prefix/reference use, mixed precision, and conversion entry points. Unused examples, training recipes, and dependency lockfiles have been removed. Transformers itself is installed as a pinned dependency; `scripts/setup_transformers.py` installs the retained OpenPI replacement modules. No upstream commit identifier was available for the original source snapshot.

Downloaded pi_05 checkpoints and PaliGemma assets are provided by their original publishers and retain the applicable OpenPI/Gemma/model terms. The ResNet encoder download is the upstream pretrained release and is verified by SHA-256 before deserialization. Robot SDKs are installed separately under vendor terms. The project's Apache-2.0 license does not replace any of these upstream terms.

Reference projects, adapted robot code, encoder sources, and their retained licenses are listed in the [README acknowledgments](README.md#acknowledgments).
