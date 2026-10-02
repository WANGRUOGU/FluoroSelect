import unittest
import numpy as np
from purchase_advice import rank_replacements


class PurchaseAdviceTests(unittest.TestCase):
    def test_high_overlap_improves_panel_and_preserves_fixed_probe(self):
        selected = np.array([[1., 1.], [0., 0.]])
        rows = rank_replacements(selected, ['A – red', 'B – orange'],
                                 np.array([[0.], [1.]]), ['Library – blue'], ['A – red'])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['Probe'], 'B')
        self.assertEqual(rows[0]['Predicted panel maximum similarity after replacement'], 0.)

    def test_low_score_does_not_trigger(self):
        self.assertEqual(rank_replacements(np.eye(2), ['A – red', 'B – blue'],
                         np.eye(2), ['Library – red', 'Library – blue']), [])

    def test_no_duplicate_or_dark_label_and_no_fixed_replacement(self):
        selected = np.array([[1., 1.], [0., 0.]])
        labels = ['A – red', 'B – orange']
        self.assertEqual(rank_replacements(selected, labels, np.zeros((2, 1)), ['Library – dark']), [])
        self.assertEqual(rank_replacements(selected, labels, np.eye(2), ['Library – red', 'Library – orange'], labels), [])
