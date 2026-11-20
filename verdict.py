"""Pure-Python verdict logic (no model / no image libraries), so it can be unit tested quickly."""

UNSAFE_CLASSES = {"No-Helmet", "No-Vest"}


def center_inside(box, person_box):
    """True if the centre of `box` lies inside `person_box`. Boxes are (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    px1, py1, px2, py2 = person_box
    return px1 <= cx <= px2 and py1 <= cy <= py2


def decide(labels):
    """Verdict for ONE person from the set of PPE labels found on them.

    Returns (verdict, reason).
    """
    labels = set(labels)
    if labels & UNSAFE_CLASSES:
        return "NOT SAFE", ", ".join(sorted(labels & UNSAFE_CLASSES))
    if {"Helmet", "Vest"} <= labels:
        return "SAFE", "Helmet + Vest"
    if labels & {"Helmet", "Vest"}:
        return "UNCERTAIN", "only " + ", ".join(sorted(labels)) + " seen"
    return "UNCERTAIN", "no PPE detected on this person"


def assign_ppe(ppe_items, person_boxes):
    """Give each PPE box to every person whose box contains its centre.

    ppe_items: list of (label, box). person_boxes: list of boxes.
    Returns (labels_per_person, ignored_count) where labels_per_person is a list of
    label-sets (same order as person_boxes) and ignored_count is the number of PPE
    boxes that were inside no person box.
    """
    labels_per_person = [set() for _ in person_boxes]
    used = set()
    for idx, (label, box) in enumerate(ppe_items):
        for p, pbox in enumerate(person_boxes):
            if center_inside(box, pbox):
                labels_per_person[p].add(label)
                used.add(idx)
    return labels_per_person, len(ppe_items) - len(used)


def overall(verdicts):
    """One summary verdict for a whole image from the per-person verdicts."""
    if not verdicts:
        return "NO PERSON"
    if "NOT SAFE" in verdicts:
        return "NOT SAFE"
    if all(v == "SAFE" for v in verdicts):
        return "SAFE"
    return "UNCERTAIN"
