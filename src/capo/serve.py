import argparse

from capo.backbone import use_vendored_openpi
from capo.config import Config, local_path
from capo.policy import Policy
from capo.processing import Processor


def main():
    parser = argparse.ArgumentParser(description="Serve CAPO with the native pi_05 Euler sampler")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="CAPO actor checkpoint; omit for an exported base model")
    parser.add_argument(
        "--norm-stats", help="Omit only when the client already sends normalized, tokenized observations"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--mean-std", action="store_true", help="Match mean/std-normalized replay")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    cfg = Config.load(local_path(args.config))
    processor = Processor(cfg, args.norm_stats, quantile=not args.mean_std) if args.norm_stats else None
    policy = Policy(cfg, args.checkpoint, processor)
    use_vendored_openpi()
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata={
            "method": "CAPO",
            "action_horizon": cfg.action_horizon,
            "execution_horizon": cfg.execution_horizon,
            "normalized": processor is None,
        },
    ).serve_forever()


if __name__ == "__main__":
    main()
