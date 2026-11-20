import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verdict import assign_ppe, center_inside, decide, overall  # noqa: E402


class TestDecide(unittest.TestCase):
    def test_helmet_and_vest_is_safe(self):
        self.assertEqual(decide({"Helmet", "Vest"})[0], "SAFE")

    def test_no_helmet_is_not_safe(self):
        self.assertEqual(decide({"No-Helmet", "Vest"})[0], "NOT SAFE")

    def test_no_vest_is_not_safe(self):
        self.assertEqual(decide({"Helmet", "No-Vest"})[0], "NOT SAFE")

    def test_unsafe_label_wins_over_safe_labels(self):
        # a person with both Helmet and No-Helmet boxes is treated cautiously
        self.assertEqual(decide({"Helmet", "No-Helmet", "Vest"})[0], "NOT SAFE")

    def test_only_helmet_is_uncertain(self):
        verdict, reason = decide({"Helmet"})
        self.assertEqual(verdict, "UNCERTAIN")
        self.assertIn("Helmet", reason)

    def test_only_vest_is_uncertain(self):
        self.assertEqual(decide({"Vest"})[0], "UNCERTAIN")

    def test_nothing_detected_is_uncertain_not_safe(self):
        # must never claim SAFE without evidence
        self.assertEqual(decide(set())[0], "UNCERTAIN")

    def test_accepts_list_input(self):
        self.assertEqual(decide(["Helmet", "Vest"])[0], "SAFE")


class TestCenterInside(unittest.TestCase):
    PERSON = (100, 100, 200, 300)

    def test_inside(self):
        self.assertTrue(center_inside((120, 120, 160, 160), self.PERSON))

    def test_outside(self):
        self.assertFalse(center_inside((300, 120, 340, 160), self.PERSON))

    def test_box_larger_than_person_but_centre_inside(self):
        self.assertTrue(center_inside((90, 90, 210, 310), self.PERSON))

    def test_edge_counts_as_inside(self):
        self.assertTrue(center_inside((90, 100, 110, 100), self.PERSON))  # centre x=100


class TestAssign(unittest.TestCase):
    def test_each_ppe_goes_to_the_right_person(self):
        persons = [(0, 0, 100, 200), (150, 0, 250, 200)]
        items = [("Helmet", (30, 10, 70, 50)), ("No-Vest", (180, 80, 220, 140))]
        labels, ignored = assign_ppe(items, persons)
        self.assertEqual(labels[0], {"Helmet"})
        self.assertEqual(labels[1], {"No-Vest"})
        self.assertEqual(ignored, 0)

    def test_ppe_outside_every_person_is_ignored(self):
        # e.g. a bright logo mistaken for a vest, with no person around
        labels, ignored = assign_ppe([("Vest", (500, 500, 600, 600))], [(0, 0, 100, 200)])
        self.assertEqual(labels, [set()])
        self.assertEqual(ignored, 1)

    def test_no_people_means_everything_ignored(self):
        labels, ignored = assign_ppe([("Helmet", (0, 0, 10, 10))], [])
        self.assertEqual(labels, [])
        self.assertEqual(ignored, 1)


class TestOverall(unittest.TestCase):
    def test_no_people(self):
        self.assertEqual(overall([]), "NO PERSON")

    def test_one_unsafe_makes_image_unsafe(self):
        self.assertEqual(overall(["SAFE", "NOT SAFE", "UNCERTAIN"]), "NOT SAFE")

    def test_all_safe(self):
        self.assertEqual(overall(["SAFE", "SAFE"]), "SAFE")

    def test_mix_of_safe_and_uncertain_is_uncertain(self):
        self.assertEqual(overall(["SAFE", "UNCERTAIN"]), "UNCERTAIN")


if __name__ == "__main__":
    unittest.main()
