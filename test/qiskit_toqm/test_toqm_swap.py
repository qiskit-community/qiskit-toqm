import unittest

from qiskit_toqm.toqm_swap import ToqmSwap

from qiskit.transpiler import TranspilerError


class TestCalcCycleMax(unittest.TestCase):
    def test_empty_durations(self):
        with self.assertRaises(TranspilerError):
            ToqmSwap._calc_cycle_max([], 10000)

    def test_gcf_scaling(self):
        """
        Test that cycle max of a non-linear sequence with a gcf is
        the max duration, reduced by that gcf.
        """
        gcf = 3
        self.assertEqual(
            ToqmSwap._calc_cycle_max(
                durations=(x*x*gcf for x in range(100)),
                max_cycle_limit=10000
            ),
            99*99
        )

    def test_1_2_6(self):
        """Test that cycle max of an already-reduced sequence is the max duration."""
        self.assertEqual(
            ToqmSwap._calc_cycle_max([1, 2, 6], 10000),
            6
        )

    def test_2_6(self):
        """
        Test that cycle max of a sequence with a gcf is the max duration,
        reduced by that gcf.
        """
        self.assertEqual(
            ToqmSwap._calc_cycle_max([2, 6], 10000),
            3
        )

    def test_0(self):
        """
        Test that cycle max of a sequence of 0 is 0.
        """
        self.assertEqual(
            ToqmSwap._calc_cycle_max([0], 10000),
            0
        )

    def test_capped_max(self):
        durations = [i for i in range(100)]
        self.assertEqual(
            ToqmSwap._calc_cycle_max(durations, 50),
            50
        )
