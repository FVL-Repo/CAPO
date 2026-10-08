# Downloaded assets

No weights or tokenizer are bundled. From the project root, run:

```bash
python scripts/download_assets.py
```

This downloads the official PaliGemma tokenizer and pretrained ResNet-10 source, verifies their SHA-256 hashes, and converts the encoder to safetensors. Pi_05 weights come from OpenPI; see the root README for download and conversion commands. Task normalization is generated from your own demonstrations.

Contents of this directory, except this README, are excluded from version control.
