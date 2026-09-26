"""Training-only selections for a finite algorithm/implementation diagnostic.

These selections describe a fully measured catalog. They are not independent,
budget-matched search runs. Validation measurements are never used to select.
"""
from __future__ import annotations


def score(measurement):
    return (-measurement["n_solved"], measurement["penalized_sgm1_s"])


def check_group_split(instances, train, validation):
    """Reject unknown/overlapping filename-derived patient groups before use."""
    if set(train) & set(validation):
        raise ValueError("training and validation image indices overlap")
    groups = [m.get("patient_group_sha256") for m in instances]
    if any(not groups[i] for i in list(train) + list(validation)):
        raise ValueError("patient group is missing; split cannot be validated")
    left = {groups[i] for i in train}
    right = {groups[i] for i in validation}
    if left & right:
        raise ValueError("patient groups overlap between training and validation")
    return {"train_groups": len(left), "validation_groups": len(right),
            "patient_overlap": 0, "grouping": "dataset filename patient token"}


def select_catalog(rows, baseline_algorithm, baseline_policy):
    """Return deterministic choices using training measurements exclusively."""
    catalog = {(r["algorithm"], r["policy"]): r for r in rows}
    if len(catalog) != len(rows):
        raise ValueError("duplicate algorithm/policy measurement")
    algorithms = sorted({a for a, _ in catalog})
    policies = sorted({b for _, b in catalog})
    if len(catalog) != len(algorithms) * len(policies):
        raise ValueError("a complete factorial catalog is required")
    baseline = (baseline_algorithm, baseline_policy)
    if baseline not in catalog:
        raise ValueError("baseline must belong to the measured catalog")

    def best(keys):
        return min(keys, key=lambda k: (score(catalog[k]["train"]), k))

    algorithm_only = best((a, baseline_policy) for a in algorithms)
    policy_only = best((baseline_algorithm, b) for b in policies)
    return {
        "fixed": baseline,
        "algorithm_only": algorithm_only,
        "implementation_only": policy_only,
        "algorithm_then_implementation": best((algorithm_only[0], b) for b in policies),
        "implementation_then_algorithm": best((a, policy_only[1]) for a in algorithms),
        "joint": best(catalog),
    }
