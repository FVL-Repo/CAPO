import argparse
import json

from capo.collect import make_record
from capo.config import Config, local_path
from capo.processing import Processor
from capo.replay import save_shard
from capo.robots.flexiv_rpc import Client, pack, unpack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--journal", help="Local copied server journal directory")
    parser.add_argument("--url", help="Loopback forwarded server URL, query result only")
    args = parser.parse_args()
    cfg = Config.load(local_path(args.config))
    processor = Processor(
        cfg, args.norm_stats, quantile=(cfg.data_spec or {}).get("normalization_mode") != "mean_std"
    )
    directory = local_path(cfg.replay)
    counts = {"already_published": 0, "recovered": 0, "quarantined": 0, "boundary_or_retry": 0}
    for path in sorted((directory / "staging").glob("*.msgpack")):
        target = directory / (path.stem + ".pt")
        if target.exists():
            counts["already_published"] += 1
            continue
        staged = unpack(path.read_bytes())
        result = staged.get("result")
        if result is None:
            journal_path = local_path(args.journal) / path.name if args.journal else None
            status = (
                unpack(journal_path.read_bytes())
                if journal_path and journal_path.exists()
                else (
                    Client(args.url).call("result", decision_id=path.stem)
                    if args.url
                    else {"status": "unknown"}
                )
            )
            if status["status"] == "complete":
                result = status["result"]
            else:
                staged["journal_status"] = status
        if result and (result.get("boundary") or result.get("retry")):
            counts["boundary_or_retry"] += 1
            continue
        if result is None or result.get("quarantined") or result.get("next_obs") is None:
            quarantine = directory / "quarantine"
            quarantine.mkdir(exist_ok=True)
            (quarantine / path.name).write_bytes(pack(staged))
            counts["quarantined"] += 1
            continue
        result = dict(result)
        result["truncated"] = not result["terminated"]
        record = make_record(
            cfg,
            processor,
            staged["obs"],
            staged["proposal"],
            result,
            human=staged["human"],
            aligned=staged["aligned"],
            demonstration=staged["demonstration"],
        )
        record["metadata"].update(
            decision_id=path.stem,
            policy_version=staged["policy_version"],
            episode_id=staged["episode_id"],
            recovered_fragment=True,
        )
        save_shard(target, [record], cfg)
        counts["recovered"] += 1
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
