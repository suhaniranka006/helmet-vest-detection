"""Pure-Python metrics comparing predicted verdicts with hand-labelled ground truth.

Ground truth is a COUNT per image (how many people were fully equipped / not equipped /
unsure), not a box per person, so people are compared by counts, not matched one to one.
These numbers are therefore approximations and should be reported as such.
"""


def _ratio(a, b):
    return None if b == 0 else a / b


def compute(truth, pred):
    """truth: {image: {"people": int, "eq": int, "ne": int, "un": int}}
    pred:  {image: [verdict, ...]}  (one verdict per detected person)

    Returns a dict of metrics. A value is None when its denominator is 0.
    """
    images = sorted(set(truth) & set(pred))

    true_people = pred_people = found = extra = 0
    tp = fp = fn = tn = 0           # image-level "contains an unsafe person"
    pred_safe_total = false_safe = 0
    clean_images = clean_flagged = 0  # images that are truly all-safe

    for name in images:
        t, verdicts = truth[name], pred[name]
        n_pred = len(verdicts)
        true_people += t["people"]
        pred_people += n_pred
        found += min(n_pred, t["people"])
        extra += max(0, n_pred - t["people"])

        truth_unsafe = t["ne"] > 0
        pred_unsafe = "NOT SAFE" in verdicts
        if truth_unsafe and pred_unsafe:
            tp += 1
        elif truth_unsafe and not pred_unsafe:
            fn += 1
        elif not truth_unsafe and pred_unsafe:
            fp += 1
        else:
            tn += 1

        n_safe = verdicts.count("SAFE")
        pred_safe_total += n_safe
        false_safe += max(0, n_safe - t["eq"])

        if t["eq"] > 0 and t["ne"] == 0 and t["un"] == 0:
            clean_images += 1
            if pred_unsafe:
                clean_flagged += 1

    return {
        "images": len(images),
        "true_people": true_people,
        "pred_people": pred_people,
        "people_recall": _ratio(found, true_people),
        "extra_people": extra,
        "unsafe_image_tp": tp, "unsafe_image_fp": fp,
        "unsafe_image_fn": fn, "unsafe_image_tn": tn,
        "unsafe_image_precision": _ratio(tp, tp + fp),
        "unsafe_image_recall": _ratio(tp, tp + fn),
        "image_accuracy": _ratio(tp + tn, len(images)),
        "pred_safe_people": pred_safe_total,
        "false_safe_people": false_safe,
        "false_safe_rate": _ratio(false_safe, pred_safe_total),
        "clean_images": clean_images,
        "clean_images_wrongly_flagged": clean_flagged,
    }


def fmt(x):
    return "n/a" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))
