# OpenPI runtime subset

Source: https://github.com/Physical-Intelligence/openpi

This directory retains only the pi_05 model, conversion, normalization, and serving dependencies used by CAPO, plus OpenPI's replacement modules for Transformers 4.53.2. Local runtime adaptations support configurable camera keys and CAPO's reference/cached-prefix training path. See the project root `THIRD_PARTY.md` for provenance and `scripts/setup_transformers.py` for installation.

For complete upstream training configurations and examples, use the upstream repository. LICENSE and LICENSE_GEMMA.txt are retained unchanged.
