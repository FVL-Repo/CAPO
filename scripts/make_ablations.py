import argparse
from dataclasses import replace
from capo.config import Config, local_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = Config.load(local_path(args.config))
    output = local_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for ranking in (False, True):
        for correction_bc in (False, True):
            name = f"rank{int(ranking)}_correction_bc{int(correction_bc)}"
            variant = replace(
                cfg,
                rank_weight=cfg.rank_weight if ranking else 0.0,
                bc_correction_fraction=cfg.bc_correction_fraction if correction_bc else 0.0,
                output=str(local_path(cfg.output) / name),
            )
            variant.save(output / (name + ".json"))
            print(output / (name + ".json"))


if __name__ == "__main__":
    main()
