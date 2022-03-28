import unittest

import qiskit_toqm.native as toqm


class TestTOQM(unittest.TestCase):

    def test_version(self):
        self.assertEqual(toqm.__version__, "0.1.0")

    def test_basic(self):
        num_q = 4
        gates = [
            toqm.GateOp(0, "cx", 0, 1),
            toqm.GateOp(1, "cx", 0, 2),
            toqm.GateOp(2, "cx", 0, 3),
            toqm.GateOp(3, "cx", 1, 2),
            toqm.GateOp(4, "cx", 1, 3),
            toqm.GateOp(5, "cx", 2, 3)
        ]

        coupling = toqm.CouplingMap(5, {(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)})

        q = toqm.DefaultQueue()
        exp = toqm.DefaultExpander()
        cf = toqm.CXFrontier()
        lat = toqm.Latency_1_2_6()
        fs = [toqm.HashFilter(), toqm.HashFilter2()]
        nms = []

        mapper = toqm.ToqmMapper(q, exp, cf, lat, nms, fs, -1)
        mapper.setRetainPopped(0)

        result = mapper.run(gates, num_q, coupling)

        # Print result
        for g in result.scheduledGates:
            print(f"{g.gateOp.type} ", end='')
            if g.physicalControl >= 0:
                print(f"q[{g.physicalControl}],", end='')
            print(f"q[{g.physicalTarget}]; ", end='')

            print(f"//cycle: {g.cycle}", end='')

            if (g.gateOp.type.lower() != "swap"):
                print(f" //{g.gateOp.type} ", end='')
                if g.gateOp.control >= 0:
                    print(f"q[{g.gateOp.control}],", end='')
                print(f"q[{g.gateOp.target}]; ", end='')
            print()