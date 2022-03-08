import unittest

from qiskit_toqm.toqm_swap import ToqmSwap

from qiskit.transpiler import TranspilerError
from math import ceil
from numpy import interp


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
        durations = [i for i in range(101)]
        self.assertEqual(
            ToqmSwap._calc_cycle_max(durations, 50),
            50
        )

    def test_capped_max_3(self):
        """
        Test that duration from 0 can be used as the smallest difference.
        In this test, the smallest diff (between 1 and 3) is too small
        :return:
        """
        durations = [1, 3, 61]

        res = ToqmSwap._calc_cycle_max(durations, 60)
        mapped_to_limit = [ceil(x) for x in interp(durations, [0, durations[-1]], [0, 60])]
        mapped_to_act = [ceil(x) for x in interp(durations, [0, durations[-1]], [0, res])]

        self.assertEqual(
            ToqmSwap._calc_cycle_max(durations, 60),
            31
        )

    def test_capped_max_2(self):
        diffs = [
            5.555e-10,
            5.
        ]
        durations = [
            3.5555555555555554e-08,
            2.2755555555555555e-07,
            2.7e-07,
            2.702222222222222e-07,
            3.0577777777777775e-07,
            4.622222222222222e-07,
            4.977777777777778e-07,
        ]

        cap = 1000
        res = ToqmSwap._calc_cycle_max(durations, cap)

        from numpy import interp
        temp = [ceil(x) for x in interp(durations, [0, durations[-1]], [0, res])]
        temp2 = [ceil(x) for x in interp(durations, [0, durations[-1]], [0, cap])]
