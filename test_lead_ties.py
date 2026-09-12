"""Musical contracts for Lead ties across half-bar and bar boundaries."""
from pathlib import Path
import unittest

import harmony_rhythm as hr
import score_builder as sb
from joint_frontend_generate import _merge_lead_continuations, _tie_probability
from melody_plan import resolve_melody_plan
from scale_config import load_scale


class ConstantBundle:
    def mixed_entry(self, steps, *weights):
        return (1.0, 0.5)


class LeadTieTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = Path(__file__).parent / "scales"
        cls.spec = load_scale(
            folder / "tiangan_72.json",
            rules=folder / "tiangan_72_5.rules.json",
            style=folder / "tiangan_72_5_norm.style.json",
            require_composition=True,
        )
        hr.configure_scale(cls.spec, ConstantBundle())
        sb.configure_scale(cls.spec)

    def config(self):
        return resolve_melody_plan(self.spec.style["melody_plan"])["joint_generation"]

    @staticmethod
    def harmony(chord_id):
        return [{"chord_id": chord_id, "offset": 0.0, "duration": 2.0}]

    @staticmethod
    def state(pitch=0, chord_id="P3_024", duration=0.5):
        return {"prev": pitch, "last_chord_id": chord_id,
                "last_note_duration": duration}

    def test_common_tone_can_cross_both_boundary_types(self):
        joint = self.config()
        section = {"start_bar": 0}
        changed = self.harmony("P3_026")  # common tones 甲、丙
        half = _tie_probability(self.state(), changed, (0.5, 1.5),
                                section, 1, "continuation", joint)
        bar = _tie_probability(self.state(), changed, (0.5, 1.5),
                               section, 2, "continuation", joint)
        self.assertGreater(half, 0.0)
        self.assertGreater(bar, 0.0)
        self.assertGreater(half, bar)

    def test_nonchord_formal_opening_and_overlong_holds_are_rejected(self):
        joint = self.config()
        changed = self.harmony("P3_026")
        self.assertEqual(_tie_probability(
            self.state(pitch=30), changed, (0.5, 1.5),
            {"start_bar": 0}, 1, "continuation", joint), 0.0)
        self.assertEqual(_tie_probability(
            self.state(), changed, (0.5, 1.5),
            {"start_bar": 1}, 2, "theme_statement", joint), 0.0)
        self.assertEqual(_tie_probability(
            self.state(duration=2.75), changed, (0.5, 1.5),
            {"start_bar": 0}, 1, "continuation", joint), 0.0)

    def test_merge_records_half_bar_and_bar_continuations(self):
        events = [
            {"start_beat": 0.0, "duration_beats": 2.0, "step": 0,
             "lead_continuation": False, "continuation_boundary": None},
            {"start_beat": 2.0, "duration_beats": 2.0, "step": 0,
             "lead_continuation": True, "continuation_boundary": "half_bar"},
            {"start_beat": 4.0, "duration_beats": 0.5, "step": 0,
             "lead_continuation": True, "continuation_boundary": "bar"},
        ]
        merged = _merge_lead_continuations(events)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["duration_beats"], 4.5)
        self.assertEqual(merged[0]["cross_half_bar_ties"], 1)
        self.assertEqual(merged[0]["cross_bar_ties"], 1)

    def test_tie_configuration_validation(self):
        with self.assertRaises(ValueError):
            resolve_melody_plan({
                "joint_generation": {"cross_bar_tie_probability": 1.1}
            })


if __name__ == "__main__":
    unittest.main()
