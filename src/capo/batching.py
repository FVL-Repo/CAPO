import torch


def tree_map(fn, tree):
    if isinstance(tree, dict):
        return {k: tree_map(fn, v) for k, v in tree.items()}
    return fn(tree)


def stack_trees(trees):
    if isinstance(trees[0], dict):
        return {k: stack_trees([t[k] for t in trees]) for k in trees[0]}
    return torch.stack(trees)


def slice_batch(batch, start, end, device):
    return tree_map(lambda t: t[start:end].to(device), batch)
