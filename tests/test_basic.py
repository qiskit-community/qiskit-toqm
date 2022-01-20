import qiskit_toqm as toqm

def test_version():
    assert toqm.__version__ == "0.1.0"

def test_basic():
    num_q = 4
    gates = [
        toqm.GateOp("cx", 1, 0),
        toqm.GateOp("cx", 2, 0),
        toqm.GateOp("cx", 3, 0),
        toqm.GateOp("cx", 2, 1),
        toqm.GateOp("cx", 3, 1),
        toqm.GateOp("cx", 3, 2)
    ]

    coupling = toqm.CouplingMap(5, {(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)})

    q = toqm.DefaultQueue()
    exp = toqm.DefaultExpander()
    cf = toqm.CXFrontier()
    lat = toqm.Latency_1_2_6()
    fs = [toqm.HashFilter(), toqm.HashFilter2()]
    nms = []

    mapper = toqm.ToqmMapper(q, exp, cf, lat, nms, fs)
    mapper.setInitialSearchCycles(-1)

    result = mapper.run(gates, num_q, coupling)

    # Print result
    for g in result.scheduledGates:
        print(f"{g.gateOp.type} ", end='')
        if g.physicalControl >= 0:
            print(f"q[{g.physicalControl}],", end='')
        print(f"q[{g.physicalTarget}]; ", end='')

        print(f"//cycle: {g.cycle}", end='')

        if (g.gateOp.type.lower() != "swp"):
            print(f" //{g.gateOp.type} ", end='')
            if g.gateOp.control >= 0:
                print(f"q[{g.gateOp.control}],", end='')
            print(f"q[{g.gateOp.target}]; ", end='')

        print()

