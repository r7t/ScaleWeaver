"""Mathematical contracts for local-ratio classification; no spectral cache."""
from collections import Counter
import unittest

from three_tone import classify_three_tone, tiangan_audit, tonic_relation


class ThreeToneTests(unittest.TestCase):
    def test_all_32_tiangan_chords(self):
        rows = tiangan_audit()
        self.assertEqual(len(rows), 32)
        self.assertEqual(Counter(r['three_tone_name'] for r in rows),
                         {'甲': 8, '庚': 7, '辛': 8, '乙': 4, '己': 3, '癸': 1, '第19步': 1})
        special = [r for r in rows if r['three_tone']['status'] == 'outside_scale']
        self.assertEqual([r['chord_name'] for r in special], ['辛甲己'])
        self.assertEqual(special[0]['three_tone']['inferred_pc'], 19)
        self.assertEqual(special[0]['three_tone']['pc'], 19)
        self.assertEqual(sum(r['three_tone']['status'] == 'implied' for r in rows), 5)

    def test_major_minor_and_modes(self):
        # C major in 31-EDO; each chord uses its own local ratio.
        scale = (0, 5, 10, 13, 18, 23, 28)
        for degree, ratio, anchor in [(0,(4,5,6),18), (1,(10,12,15),13),
                                      (2,(10,12,15),18), (3,(4,5,6),0),
                                      (4,(4,5,6),5), (5,(10,12,15),0)]:
            chord = tuple(scale[(degree+i) % 7] for i in (0,2,4))
            tone = classify_three_tone(chord, ratio, 31, scale)
            self.assertEqual((tone.status, tone.pc), ('present', anchor))
        tone = classify_three_tone((0,10,18), (4,5,6), 31, scale)
        self.assertEqual(tonic_relation(tone, 0, 31), 'tonic_reference')
        self.assertEqual(tonic_relation(tone, 18, 31), 'subdominant_reference')

    def test_transposition_inversion_and_common_scaling(self):
        for shift in (0, 7, 19, 71):
            for pcs, ratio in [((0,23,42),(4,5,6)), ((42,0,23),(3,4,5)),
                               ((23,42,72),(5,6,8)), ((0,23,42),(12,15,18)),
                               ((0,23,42,114),(4,5,6,12))]:
                tone = classify_three_tone(tuple(p+shift for p in pcs), ratio,
                                           72, tuple(p+shift for p in (0,23,42)))
                self.assertEqual((tone.status,tone.pc), ('present',(42+shift)%72))

    def test_completion_and_scale_boundary(self):
        self.assertEqual(classify_three_tone((0,23,58),(4,5,7),72,(0,23,42,58)).pc,42)
        tone = classify_three_tone((0,23,58),(4,5,7),72,(0,23,58))
        self.assertEqual((tone.status,tone.pc),('outside_scale',42))

    def test_three_membership_classes_share_the_same_val_reference(self):
        for pcs, ns, scale, status in (
            ((0,23,42,58),(4,5,6,7),(0,23,42,58),'present'),
            ((0,23,58),(4,5,7),(0,23,42,58),'implied'),
            ((0,23,58),(4,5,7),(0,23,58),'outside_scale')):
            tone = classify_three_tone(pcs,ns,72,scale)
            self.assertEqual((tone.pc,tone.status),(42,status))
            self.assertEqual(tone.to_dict()['in_chord'],status=='present')
            self.assertEqual(tone.to_dict()['in_scale'],status!='outside_scale')

    def test_external_reference_transposes_and_wraps(self):
        for shift in (0,7,60,71):
            tone = classify_three_tone(tuple(p+shift for p in (49,0,35)),(4,5,7),72,
                                       tuple(p+shift for p in (49,0,35)))
            self.assertEqual((tone.pc,tone.status),((19+shift)%72,'outside_scale'))

    def test_tempered_alias_counts_as_present(self):
        # In 12-EDO 375 = 3 * 5**3 aliases 3 modulo octaves.
        from ji_ratio import patent_integer_value
        ns=(375,8)
        pcs=tuple(patent_integer_value(n,12)%12 for n in ns)
        tone=classify_three_tone(pcs,ns,12,pcs)
        self.assertEqual(tone.status,'present')
        self.assertFalse(tone.explicit_harmonic)

    def test_bad_or_unavailable_interpretation(self):
        tone = classify_three_tone((0,23,42),(1,2,3),72,(0,23,42))
        self.assertEqual(tone.reason,'inconsistent_patent_mapping')
        with self.assertRaises(ValueError):
            classify_three_tone((0,23),(4,),72,(0,23))
        with self.assertRaises(ValueError):
            classify_three_tone((0,23),(0,5),72,(0,23))


if __name__ == '__main__':
    unittest.main()
