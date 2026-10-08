import math


def continuation_metrics(rows, normalizer):
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("NMAE normalizer Z must be finite and positive")
    errors, improvements = [], []
    for row in rows:
        if not row.get("frozen_policy_sha256") or not row.get("independent_continuations"):
            raise ValueError("Continuation evidence must identify frozen policy and independent rollouts")
        if row["kind"] == "value":
            if not row.get("evaluated_action_executed"):
                raise ValueError("MC return requires executing the evaluated action")
            predicted, returns = row["q"], row["returns"]
            if not returns or not all(math.isfinite(x) for x in [predicted, *returns]):
                raise ValueError("Invalid MC measurements")
            errors.append(abs(predicted - sum(returns) / len(returns)))
        elif row["kind"] == "gradient":
            gradient, random = row["gradient_returns"], row["random_returns"]
            norms = row["projected_gradient_norm"], row["projected_random_norm"]
            if not gradient or not random or not all(math.isfinite(x) for x in [*gradient, *random, *norms]):
                raise ValueError("Invalid gradient continuation measurements")
            if norms[0] <= 0 or not math.isclose(*norms, rel_tol=1e-5, abs_tol=1e-7):
                raise ValueError("Perturbation norms must match AFTER clipping/projection")
            improvements.append(sum(gradient) / len(gradient) - sum(random) / len(random))
        else:
            raise ValueError("Unknown continuation diagnostic kind")
    return {
        "mc_samples": len(errors),
        "nmae_normalizer": normalizer,
        "mc_nmae": sum(errors) / (len(errors) * normalizer) if errors else None,
        "gradient_contexts": len(improvements),
        "gradient_minus_random_return": sum(improvements) / len(improvements) if improvements else None,
    }


def event_key(record):
    metadata = record.get("metadata", {})

    key = metadata.get("intervention_event_id") or metadata.get("episode_id")
    if not key:
        raise ValueError("Held-out evaluation requires intervention_event_id or episode_id")
    return key


def check_heldout(training, heldout):
    train_events = {event_key(row) for row in training}
    heldout_events = {event_key(row) for row in heldout}
    if train_events & heldout_events:
        raise ValueError("Training/evaluation intervention events overlap")
    return {"training_events": len(train_events), "heldout_events": len(heldout_events), "event_overlap": 0}
