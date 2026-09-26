from atlas.bench.factorial import select_catalog, check_group_split
import pytest


def row(a, b, seconds, solved=4):
    return {"algorithm": a, "policy": b,
            "train": {"n_solved": solved, "penalized_sgm1_s": seconds},
            "validation": {"n_solved": 4, "penalized_sgm1_s": 999 - seconds}}


def test_joint_can_find_interaction_missed_by_both_sequential_orders():
    rows = [row("a", "x", 2), row("a", "y", 3),
            row("b", "x", 4), row("b", "y", 1)]
    choices = select_catalog(rows, "a", "x")
    assert choices["joint"] == ("b", "y")
    assert choices["algorithm_then_implementation"] == ("a", "x")
    assert choices["implementation_then_algorithm"] == ("a", "x")
    # Validation rankings intentionally favor another recipe.
    for r in rows:
        r["validation"]["penalized_sgm1_s"] = -r["train"]["penalized_sgm1_s"]
    assert select_catalog(rows, "a", "x") == choices


def test_failure_is_not_rewarded_for_a_short_runtime():
    rows = [row("a", "x", 2), row("b", "x", .001, solved=3)]
    assert select_catalog(rows, "a", "x")["joint"] == ("a", "x")


def test_incomplete_catalog_is_not_silently_selected():
    with pytest.raises(ValueError, match="complete factorial"):
        select_catalog([row("a", "x", 2), row("b", "y", 1)], "a", "x")


def test_group_check_detects_patient_leakage_despite_distinct_images():
    from atlas.bench.kermany import patient_group_hash
    first = patient_group_hash("CNV-1234567-1.jpeg")
    second = patient_group_hash("NORMAL-1234567-2.jpeg")
    assert first == second
    with pytest.raises(ValueError, match="patient groups overlap"):
        check_group_split([{"patient_group_sha256": first},
                           {"patient_group_sha256": second}], [0], [1])
    assert patient_group_hash("unrecognized.jpg") is None
    with pytest.raises(ValueError, match="missing"):
        check_group_split([{}, {}], [0], [1])


def test_group_check_accepts_disjoint_patients():
    result = check_group_split([{"patient_group_sha256": "a"},
                                {"patient_group_sha256": "b"}], [0], [1])
    assert result["patient_overlap"] == 0
    assert result["train_groups"] == result["validation_groups"] == 1
