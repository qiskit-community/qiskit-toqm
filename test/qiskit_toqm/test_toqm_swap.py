import unittest

from qiskit_toqm.toqm_swap import ToqmSwap
from qiskit.transpiler import CouplingMap, InstructionDurations, TranspilerError


class TestBuildLatencyDescriptions(unittest.TestCase):
    def setUp(self) -> None:
        super()
        self.num_qubits = 3
        self.coupling_map = CouplingMap.from_line(self.num_qubits)
        self.basis_gates = ["rz", "x", "cx", "measure"]

        def durations_for_1q(gate, duration, unit="dt"):
            for i in range(self.num_qubits):
                yield gate, (i,), duration, unit

        def durations_for_2q(gate, duration, unit="dt"):
            for i, j in self.coupling_map.get_edges():
                yield gate, (i, j), duration, unit

        self.durations_for_1q = durations_for_1q
        self.durations_for_2q = durations_for_2q

    def test_already_normalized(self):
        """Test that durations are used verbatim when they're already normalized."""
        durations = InstructionDurations([
            *self.durations_for_1q("rz", 0),
            *self.durations_for_1q("x", 1),
            *self.durations_for_2q("cx", 2),
            *self.durations_for_2q("swap", 6)
        ], dt=1)

        swapper = ToqmSwap(self.coupling_map, durations)
        latencies = list(swapper._build_latency_descriptions())

        self.assertTrue(
            all(x.latency == 0 for x in latencies if x.type == "rz")
        )

        self.assertTrue(
            all(x.latency == 1 for x in latencies if x.type == "x")
        )

        self.assertTrue(
            all(x.latency == 2 for x in latencies if x.type == "cx")
        )

        self.assertTrue(
            all(x.latency == 6 for x in latencies if x.type == "swap")
        )

    def test_normalize_s(self):
        durations = InstructionDurations([
            *self.durations_for_1q("rz", 0, unit="s"),
            *self.durations_for_1q("x", 3.5555555555555554e-08, unit="s"),
            *self.durations_for_2q("cx", 2.2755555555555555e-07, unit="s"),
            *self.durations_for_2q("swap", 4.977777777777778e-07, unit="s")
        ])

        swapper = ToqmSwap(self.coupling_map, durations)
        latencies = list(swapper._build_latency_descriptions())

        self.assertTrue(
            all(x.latency == 0 for x in latencies if x.type == "rz")
        )

        self.assertTrue(
            all(x.latency == 1 for x in latencies if x.type == "x")
        )

        self.assertTrue(
            all(x.latency == 7 for x in latencies if x.type == "cx")
        )

        self.assertTrue(
            all(x.latency == 15 for x in latencies if x.type == "swap")
        )

    def test_missing_swap_durations(self):
        # Create durations with no swap info
        durations = InstructionDurations([
            *self.durations_for_1q("rz", 0),
            *self.durations_for_1q("x", 1),
            *self.durations_for_2q("cx", 2)
        ], dt=1)

        # Attempt to construct ToqmSwap without backend info
        with self.assertRaisesRegex(TranspilerError, "Both 'basis_gates' and 'backend_properties' must be specified.*"):
            ToqmSwap(self.coupling_map, durations)

    def test_all_0_durations(self):
        """
        Constructing ToqmSwap should fail if all gate durations are 0.
        """
        # Create durations such that all instructions finish instantaneously.
        durations = InstructionDurations([
            *self.durations_for_1q("rz", 0),
            *self.durations_for_1q("x", 0),
            *self.durations_for_2q("cx", 0),
            *self.durations_for_2q("swap", 0)
        ], dt=1)

        # Attempt to construct ToqmSwap.
        with self.assertRaisesRegex(TranspilerError, "Durations must be specified for the target."):
            ToqmSwap(self.coupling_map, durations)