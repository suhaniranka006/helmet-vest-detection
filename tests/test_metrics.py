import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))

from metrics import compute  # noqa: E402


def row(eq=0, ne=0, un=0):
    return {"people": eq + ne + un, "eq": eq, "ne": ne, "un": un}


class TestMetrics(unittest.TestCase):
    def test_perfect_prediction(self):
        truth = {"a.jpg": row(eq=2), "b.jpg": row(eq=1, ne=1)}
        pred = {"a.jpg": ["SAFE", "SAFE"], "b.jpg": ["SAFE", "NOT SAFE"]}
        m = compute(truth, pred)
        self.assertEqual(m["people_recall"], 1.0)
        self.assertEqual(m["extra_people"], 0)
        self.assertEqual(m["unsafe_image_precision"], 1.0)
        self.assertEqual(m["unsafe_image_recall"], 1.0)
        self.assertEqual(m["image_accuracy"], 1.0)
        self.assertEqual(m["false_safe_rate"], 0.0)
        self.assertEqual(m["clean_images_wrongly_flagged"], 0)

    def test_missed_people_lower_recall(self):
        truth = {"a.jpg": row(eq=4)}
        pred = {"a.jpg": ["SAFE"]}
        self.assertEqual(compute(truth, pred)["people_recall"], 0.25)

    def test_extra_people_are_counted(self):
        truth = {"a.jpg": row(eq=1)}
        pred = {"a.jpg": ["SAFE", "UNCERTAIN", "UNCERTAIN"]}
        m = compute(truth, pred)
        self.assertEqual(m["extra_people"], 2)
        self.assertEqual(m["people_recall"], 1.0)

    def test_false_safe_is_counted(self):
        # model says 2 SAFE but only 1 person was really fully equipped
        truth = {"a.jpg": row(eq=1, ne=1)}
        pred = {"a.jpg": ["SAFE", "SAFE"]}
        m = compute(truth, pred)
        self.assertEqual(m["false_safe_people"], 1)
        self.assertEqual(m["false_safe_rate"], 0.5)
        self.assertEqual(m["unsafe_image_fn"], 1)

    def test_false_alarm_on_clean_image(self):
        truth = {"a.jpg": row(eq=2)}
        pred = {"a.jpg": ["NOT SAFE", "SAFE"]}
        m = compute(truth, pred)
        self.assertEqual(m["unsafe_image_fp"], 1)
        self.assertEqual(m["clean_images"], 1)
        self.assertEqual(m["clean_images_wrongly_flagged"], 1)

    def test_image_with_unsure_is_not_a_clean_image(self):
        truth = {"a.jpg": row(eq=1, un=1)}
        pred = {"a.jpg": ["NOT SAFE", "SAFE"]}
        self.assertEqual(compute(truth, pred)["clean_images"], 0)

    def test_empty_denominators_give_none(self):
        truth = {"a.jpg": row()}
        pred = {"a.jpg": []}
        m = compute(truth, pred)
        self.assertIsNone(m["people_recall"])
        self.assertIsNone(m["unsafe_image_precision"])
        self.assertIsNone(m["false_safe_rate"])

    def test_only_common_images_are_used(self):
        truth = {"a.jpg": row(eq=1), "b.jpg": row(eq=1)}
        pred = {"a.jpg": ["SAFE"]}
        self.assertEqual(compute(truth, pred)["images"], 1)


if __name__ == "__main__":
    unittest.main()
