import unittest

from qiskit_toqm.toqm_latency import latencies_from_target
from qiskit.transpiler import CouplingMap, InstructionDurations, TranspilerError
from qiskit.providers.fake_provider import FakeMontrealV2


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

    def test_from_target(self):
        backend = FakeMontrealV2()
        target = backend.target

        # Due to a units bug in Terra, we need to set dt ourselves
        target.dt = .2222222222222 * 1e-9
        latencies = list(latencies_from_target(target=target, normalize_scale=1))

        self.assertTrue(len(latencies) > 0)
        self.assertTrue(any(l.latency == 1 for l in latencies))

    def test_already_normalized(self):
        """
        Already normalized durations are used as cycle count without conversion.
        """
        durations = InstructionDurations([
            *self.durations_for_1q("rz", 0),
            *self.durations_for_1q("x", 1),
            *self.durations_for_2q("cx", 2),
            *self.durations_for_2q("swap", 6)
        ], dt=1)

        latencies = list(latencies_from_target(self.coupling_map, durations))

        self.assertTrue(
            all(x.latency == 0 for x in latencies if x.type == "rz")
        )

        self.assertTrue(
            all(x.latency == 2 for x in latencies if x.type == "x")
        )

        self.assertTrue(
            all(x.latency == 4 for x in latencies if x.type == "cx")
        )

        self.assertTrue(
            all(x.latency == 12 for x in latencies if x.type == "swap")
        )

    def test_normalize_s(self):
        """
        Durations provided in unit 's' should produce expected cycles.
        """
        durations = InstructionDurations([
            *self.durations_for_1q("rz", 0, unit="s"),
            *self.durations_for_1q("x", 3.5555555555555554e-08, unit="s"),
            *self.durations_for_2q("cx", 2.2755555555555555e-07, unit="s"),
            *self.durations_for_2q("swap", 4.977777777777778e-07, unit="s")
        ])

        latencies = list(latencies_from_target(self.coupling_map, durations))

        self.assertTrue(
            all(x.latency == 0 for x in latencies if x.type == "rz")
        )

        self.assertTrue(
            all(x.latency == 2 for x in latencies if x.type == "x")
        )

        self.assertTrue(
            all(x.latency == 13 for x in latencies if x.type == "cx")
        )

        self.assertTrue(
            all(x.latency == 28 for x in latencies if x.type == "swap")
        )

    def test_missing_swap_durations(self):
        """
        Calling latencies_from_target without providing swap durations or a backend should fail.
        """
        # Create durations with no swap info
        durations = InstructionDurations([
            *self.durations_for_1q("rz", 0),
            *self.durations_for_1q("x", 1),
            *self.durations_for_2q("cx", 2)
        ], dt=1)

        # Attempt to call latencies_from_target without backend info
        with self.assertRaisesRegex(TranspilerError,
                                    ".*'basis_gates' and 'backend_properties' must be specified.*"):
            list(latencies_from_target(self.coupling_map, durations))

    def test_all_0_durations(self):
        """
        Calling latencies_from_target should fail if all gate durations are 0.
        """
        # Create durations such that all instructions finish instantaneously.
        durations = InstructionDurations([
            *self.durations_for_1q("rz", 0),
            *self.durations_for_1q("x", 0),
            *self.durations_for_2q("cx", 0),
            *self.durations_for_2q("swap", 0)
        ], dt=1)

        with self.assertRaisesRegex(TranspilerError, "Durations must be specified for the target."):
            list(latencies_from_target(self.coupling_map, durations))

    def test_normalize_dt(self):
        """
        Not yet normalized durations provided in unit 's' should produce expected cycles.
        """
        durations = InstructionDurations([
            *self.durations_for_1q("rz", 0),
            *self.durations_for_1q("x", 10),
            *self.durations_for_2q("cx", 100),
            *self.durations_for_2q("swap", 152)
        ], dt=1)

        latencies = list(latencies_from_target(self.coupling_map, durations))

        self.assertTrue(
            all(x.latency == 0 for x in latencies if x.type == "rz")
        )

        self.assertTrue(
            all(x.latency == 2 for x in latencies if x.type == "x")
        )

        self.assertTrue(
            all(x.latency == 20 for x in latencies if x.type == "cx")
        )

        # round(152*2/10) = 30
        self.assertTrue(
            all(x.latency == 30 for x in latencies if x.type == "swap")
        )

    def test_normalize_close(self):
        """
        Durations less than 1x the duration of the shortest duration should round
        to 1 or 2 depending on magnitude.
        """
        durations = InstructionDurations([
            *self.durations_for_1q("rz", 0),
            *self.durations_for_1q("x", 3),
            *self.durations_for_2q("cx", 4),
            *self.durations_for_2q("swap", 5)
        ], dt=1)

        latencies = list(latencies_from_target(self.coupling_map, durations))

        self.assertTrue(
            all(x.latency == 0 for x in latencies if x.type == "rz")
        )

        self.assertTrue(
            all(x.latency == 2 for x in latencies if x.type == "x")
        )

        # round(4*2/3) = 3
        self.assertTrue(
            all(x.latency == 3 for x in latencies if x.type == "cx")
        )

        # round(5*2/3) = 3
        self.assertTrue(
            all(x.latency == 3 for x in latencies if x.type == "swap")
        )
