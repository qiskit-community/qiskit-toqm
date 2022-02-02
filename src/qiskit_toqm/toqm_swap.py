# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
import qiskit.dagcircuit.dagcircuit
import qiskit_toqm as toqm
import logging
from collections import defaultdict
from copy import copy, deepcopy
import numpy as np

from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.circuit.quantumregister import Qubit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGOpNode

logger = logging.getLogger(__name__)


class ToqmSwap(TransformationPass):
    r"""Map input circuit onto a backend topology via insertion of SWAPs.
    Implementation of the SWAP-based approach from Time-Optimal Qubit
    Mapping paper [1] (Algorithm 1).
    **References:**
    [1] Chi Zhang, Ari B. Hayes, Longfei Qiu, Yuwei Jin, Yanhao Chen, and Eddy Z. Zhang. 2021. Time-Optimal Qubit
    Mapping. In Proceedings of the 26th ACM International Conference on Architectural Support for Programming
    Languages and Operating Systems (ASPLOS ’21), April 19–23, 2021, Virtual, USA.
    ACM, New York, NY, USA, 14 pages.
    `<https://doi.org/10.1145/3445814.3446706>`_
    """

    def __init__(self, coupling_map, initial_search_limit = None):
        r"""ToqmSwap initializer.
        Args:
            coupling_map (CouplingMap): CouplingMap of the target backend.
        """
        super().__init__()

        queue = toqm.DefaultQueue()
        expander = toqm.DefaultExpander()
        cost_func = toqm.CXFrontier()
        latency = toqm.Latency_1_2_6()
        filters = [toqm.HashFilter(), toqm.HashFilter2()]
        node_mods = []

        self.mapper = toqm.ToqmMapper(queue, expander, cost_func, latency, node_mods, filters)
        self.coupling_map = coupling_map
        self.initial_search_limit = initial_search_limit

    def run(self, dag: qiskit.dagcircuit.dagcircuit.DAGCircuit):
        """Run the ToqmSwap pass on `dag`.
        Args:
            dag (DAGCircuit): the directed acyclic graph to be mapped.
        Returns:
            DAGCircuit: A dag mapped to be compatible with the coupling_map.
        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG
        """
        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError("More virtual qubits exist than physical.")

        layout = self.property_set["layout"]
        if layout is None:
            self.mapper.clearInitialMapping()
        else:
            p2v = layout.get_physical_bits()
            qal = [p2v[p][1] for p in range(len(p2v))]
            self.mapper.setInitialMappingQal(qal)

        # Always use search cycles from user if provided, even if a layout
        # was also provided. Else, if no layout, do a full search (-1), else skip
        # search entirely (0).
        cycles_default = -1 if layout is None else 0
        search_cycles = cycles_default if self.initial_search_limit is None else self.initial_search_limit
        self.mapper.setInitialSearchCycles(search_cycles)

        # Create TOQM topological gate list
        qubit_to_vidx = {bit: idx for idx, bit in enumerate(dag.qubits)}

        def gates():
            for node in dag.topological_op_nodes():
                if len(node.qargs) == 2:
                    yield toqm.GateOp(node.op.name, qubit_to_vidx[node.qargs[1]], qubit_to_vidx[node.qargs[0]])
                elif len(node.qargs) == 1:
                    yield toqm.GateOp(node.op.name, qubit_to_vidx[node.qargs[0]], -1)
                else:
                    raise "unexpected num gates!"

        gate_ops = list(gates())

        # Create TOQM coupling map
        edges = {e for e in self.coupling_map.get_edges()}
        coupling = toqm.CouplingMap(self.coupling_map.size(), edges)

        result = self.mapper.run(gate_ops, dag.num_qubits(), coupling)

        # TODO: return modified dag
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

        return dag
