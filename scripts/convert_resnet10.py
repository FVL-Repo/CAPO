import argparse
import hashlib
import json
import pickle

import numpy as np
import torch
from safetensors.torch import save_file

from capo.config import local_path
from capo.resnet import ResNet10, SOURCE_SHA256


def load_source(path):
    content = local_path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
        raise ValueError("Refusing to deserialize a pickle that differs from the pinned pretrained encoder release")
    return pickle.loads(content)


def convert(source):
    output = {}
    used = set()

    def conv(source_key, target_key):
        keys = source_key.split("/")
        subtree = source
        for key in keys:
            subtree = subtree[key]
        output[target_key + ".weight"] = (
            torch.from_numpy(np.array(subtree["kernel"])).permute(3, 2, 0, 1).contiguous()
        )
        used.add(source_key + "/kernel")

    def norm(source_key, target_key):
        subtree = source
        for key in source_key.split("/"):
            subtree = subtree[key]
        for a, b in [("scale", "weight"), ("bias", "bias")]:
            output[target_key + "." + b] = torch.from_numpy(np.array(subtree[a])).contiguous()
            used.add(source_key + "/" + a)

    conv("conv_init", "conv_init")
    norm("norm_init", "norm_init")
    for i in range(4):
        prefix = f"ResNetBlock_{i}"
        for j in range(2):
            conv(f"{prefix}/Conv_{j}", f"blocks.{i}.conv{j + 1}")
            norm(f"{prefix}/MyGroupNorm_{j}", f"blocks.{i}.norm{j + 1}")
        if i:
            conv(f"{prefix}/conv_proj", f"blocks.{i}.projection")
            norm(f"{prefix}/norm_proj", f"blocks.{i}.projection_norm")

    def leaves(tree, prefix=""):
        result = set()
        for key, value in tree.items():
            if key == "output_head" and not prefix:
                continue
            if isinstance(value, dict):
                result |= leaves(value, prefix + key + "/")
            else:
                result.add(prefix + key)
        return result

    if leaves(source) != used or len(output) != 36 or sum(t.numel() for t in output.values()) != 4_905_792:
        raise ValueError("Unexpected pretrained backbone tree")
    ResNet10().load_state_dict(output, strict=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="assets/resnet10_params.pkl")
    parser.add_argument("--output", default="assets/resnet10.safetensors")
    args = parser.parse_args()
    source = load_source(args.input)
    weights = convert(source)
    output = local_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    save_file(weights, str(output))
    print(
        json.dumps(
            {
                "file": str(output),
                "leaves": len(weights),
                "parameters": sum(t.numel() for t in weights.values()),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
