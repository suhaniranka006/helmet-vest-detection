"""pytest-style tests for verdict.py: tables (parametrize), a fixture, and an exhaustive check.

These run together with the older unittest-style tests (pytest collects both).
"""
from itertools import chain, combinations

import pytest

from verdict import UNSAFE_CLASSES, assign_ppe, center_inside, decide, overall

ALL_LABELS = ["Helmet", "No-Helmet", "No-Vest", "Vest"]


# ---------------------------------------------------------------- decide: one table, many cases
@pytest.mark.parametrize(
    "labels, expected",
    [
        pytest.param({"Helmet", "Vest"}, "SAFE", id="helmet+vest"),
        pytest.param({"No-Helmet"}, "NOT SAFE", id="only-no-helmet"),
        pytest.param({"No-Vest"}, "NOT SAFE", id="only-no-vest"),
        pytest.param({"No-Helmet", "No-Vest"}, "NOT SAFE", id="no-helmet+no-vest"),
        pytest.param({"Helmet", "No-Vest"}, "NOT SAFE", id="helmet+no-vest"),
        pytest.param({"Vest", "No-Helmet"}, "NOT SAFE", id="vest+no-helmet"),
        pytest.param({"Helmet", "Vest", "No-Vest"}, "NOT SAFE", id="conflict-red-wins"),
        pytest.param({"Helmet"}, "UNCERTAIN", id="only-helmet"),
        pytest.param({"Vest"}, "UNCERTAIN", id="only-vest"),
        pytest.param(set(), "UNCERTAIN", id="nothing-detected"),
        pytest.param(["Helmet", "Vest"], "SAFE", id="list-input"),
    ],
)
def test_decide_verdict(labels, expected):
    assert decide(labels)[0] == expected


def _all_subsets(items):
    return chain.from_iterable(combinations(items, r) for r in range(len(items) + 1))


@pytest.mark.parametrize("subset", [set(s) for s in _all_subsets(ALL_LABELS)],
                         ids=lambda s: "+".join(sorted(s)) or "empty")
def test_safe_only_with_helmet_and_vest_and_no_negative_labels(subset):
    """Exhaustive: there are only 16 label combinations, so check every one of them.

    SAFE must happen if and only if Helmet and Vest are both present and no No-* label is.
    """
    must_be_safe = {"Helmet", "Vest"} <= subset and not (subset & UNSAFE_CLASSES)
    assert (decide(subset)[0] == "SAFE") == must_be_safe


def test_decide_always_gives_a_reason():
    for subset in _all_subsets(ALL_LABELS):
        verdict, reason = decide(set(subset))
        assert verdict in {"SAFE", "NOT SAFE", "UNCERTAIN"}
        assert isinstance(reason, str) and reason


# ---------------------------------------------------------------- overall
@pytest.mark.parametrize(
    "verdicts, expected",
    [
        ([], "NO PERSON"),
        (["SAFE"], "SAFE"),
        (["SAFE", "SAFE"], "SAFE"),
        (["NOT SAFE"], "NOT SAFE"),
        (["SAFE", "NOT SAFE", "UNCERTAIN"], "NOT SAFE"),
        (["SAFE", "UNCERTAIN"], "UNCERTAIN"),
        (["UNCERTAIN"], "UNCERTAIN"),
    ],
)
def test_overall(verdicts, expected):
    assert overall(verdicts) == expected


# ---------------------------------------------------------------- center_inside
PERSON = (100, 100, 200, 300)


@pytest.mark.parametrize(
    "box, inside",
    [
        ((120, 120, 160, 160), True),   # clearly inside
        ((300, 120, 340, 160), False),  # to the right of the person
        ((0, 0, 10, 10), False),        # far away
        ((90, 90, 210, 310), True),     # bigger than the person, centre still inside
        ((90, 100, 110, 100), True),    # centre exactly on the left edge counts as inside
        ((190, 290, 230, 330), False),  # centre (210, 310) is outside
        ((150, 200, 152, 202), True),   # tiny box in the middle
    ],
)
def test_center_inside(box, inside):
    assert center_inside(box, PERSON) is inside


# ---------------------------------------------------------------- assign_ppe with a fixture
@pytest.fixture
def two_separate_people():
    """Two people standing apart: a fixture is set-up code that tests receive as an argument."""
    return [(0, 0, 100, 200), (150, 0, 250, 200)]


def test_each_item_goes_to_its_own_person(two_separate_people):
    items = [("Helmet", (30, 10, 70, 50)), ("No-Vest", (180, 80, 220, 140))]
    labels, ignored = assign_ppe(items, two_separate_people)
    assert labels == [{"Helmet"}, {"No-Vest"}]
    assert ignored == 0


def test_item_outside_everyone_is_ignored(two_separate_people):
    labels, ignored = assign_ppe([("Vest", (500, 500, 600, 600))], two_separate_people)
    assert labels == [set(), set()]
    assert ignored == 1


def test_overlapping_people_both_get_the_box_known_limitation():
    """Documents a known weakness: a box whose centre is inside two overlapping person boxes
    is given to BOTH people. A better rule (body regions) is listed as a next step."""
    overlapping = [(0, 0, 100, 200), (50, 0, 150, 200)]
    labels, ignored = assign_ppe([("Helmet", (60, 10, 90, 50))], overlapping)
    assert labels == [{"Helmet"}, {"Helmet"}]
    assert ignored == 0
