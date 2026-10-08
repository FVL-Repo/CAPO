from datetime import timedelta
import os

import torch
import torch.distributed as dist


def active():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if active() else 0


def world_size():
    return dist.get_world_size() if active() else 1


def initialize(cfg):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        if str(cfg.device).startswith("cuda"):
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            cfg.device = f"cuda:{local_rank}"
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend, timeout=timedelta(minutes=30))
    if cfg.batch_size % world_size():
        raise ValueError("Global batch_size must be divisible by world size")


@torch.no_grad()
def broadcast_model(model):
    if active():
        for tensor in (*model.parameters(), *model.buffers()):
            dist.broadcast(tensor, src=0)


def mean_metrics(metrics, device):
    if not active():
        return metrics
    values = torch.tensor(list(metrics.values()), dtype=torch.float64, device=device)
    dist.all_reduce(values)
    return dict(zip(metrics, (values / world_size()).tolist()))


@torch.no_grad()
def average_gradients(params, bucket_elements=4_000_000):
    params = [p for p in params if p.requires_grad]
    if not active() or not params:
        return
    used = torch.tensor([p.grad is not None for p in params], device=params[0].device, dtype=torch.int32)
    dist.all_reduce(used)
    bucket, size = [], 0

    def flush():
        if not bucket:
            return
        flat = torch.cat([g.float().reshape(-1) for g in bucket])
        dist.all_reduce(flat)
        flat.div_(world_size())
        offset = 0
        for grad in bucket:
            grad.copy_(flat[offset : offset + grad.numel()].view_as(grad))
            offset += grad.numel()
        bucket.clear()

    for p, count in zip(params, used.tolist()):
        if count == 0:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)

        for chunk in p.grad.view(-1).split(bucket_elements):
            if size + chunk.numel() > bucket_elements:
                flush()
                size = 0
            bucket.append(chunk)
            size += chunk.numel()
    flush()


def gather_objects(value):
    if not active():
        return [value]
    values = [None] * world_size()
    dist.all_gather_object(values, value)
    return values
