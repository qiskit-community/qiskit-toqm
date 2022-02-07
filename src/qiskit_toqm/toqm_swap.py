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
import logging
import qiskit.dagcircuit.dagcircuit
import qiskit_toqm as toqm

from collections import namedtuple
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

logger = logging.getLogger(__name__)

ToqmLayoutSettings = namedtuple("ToqmLayoutSettings", "search_cycle_limit")


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

    def __init__(self, coupling_map, layout_settings: ToqmLayoutSettings = None):
        r"""ToqmSwap initializer.
        Args:
            coupling_map (CouplingMap): CouplingMap of the target backend.
            layout_settings (ToqmLayoutSettings): If specified, enables TOQM
                layout using the specified settings, which modifies
                property_set['layout'].
        """
        super().__init__()

        queue = toqm.DefaultQueue()
        expander = toqm.DefaultExpander()
        cost_func = toqm.CXFrontier()
        latency = toqm.Latency_1_2_6()
        filters = [toqm.HashFilter(), toqm.HashFilter2()]
        node_mods = []

        self.mapper = toqm.ToqmMapper(queue, expander, cost_func, latency, node_mods, filters)
        self.mapper.setRetainPopped(0)

        self.coupling_map = coupling_map

        # If user provided layout settings, use specified search cycle limit.
        # If limit is specified as None, use -1 for no limit.
        self.search_cycles = 0 if layout_settings is None else layout_settings.search_cycle_limit
        if self.search_cycles is None:
            self.search_cycles = -1

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
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("TOQM swap runs on physical circuits only.")

        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError("More virtual qubits exist than physical.")

        reg = dag.qregs["q"]

        # Generate UIDs for each gate node from the original circuit so we can
        # look them up later when rebuilding the circuit.
        # Note: this is still sorted by topological order from above.
        uid_to_op_node = {uid: op for uid, op in enumerate(dag.topological_op_nodes())}

        # Create TOQM topological gate list
        def gates():
            for uid, node in uid_to_op_node.items():
                if len(node.qargs) == 2:
                    yield toqm.GateOp(uid, node.op.name, reg.index(node.qargs[0]), reg.index(node.qargs[1]))
                elif len(node.qargs) == 1:
                    yield toqm.GateOp(uid, node.op.name, reg.index(node.qargs[0]))
                else:
                    # TODO: add handling for barrier and measure
                    raise TranspilerError("Unexpected num gates!")

        gate_ops = list(gates())

        # Create TOQM coupling map
        edges = {e for e in self.coupling_map.get_edges()}
        coupling = toqm.CouplingMap(self.coupling_map.size(), edges)

        result: toqm.ToqmResult = self.mapper.run(gate_ops, dag.num_qubits(), coupling, self.search_cycles)

        # Preserve input DAG's name, regs, wire_map, etc. but replace the graph.
        mapped_dag = dag._copy_circuit_metadata()

        for g in result.scheduledGates:
            if g.gateOp.type.lower() == "swp":
                mapped_dag.apply_operation_back(SwapGate(), qargs=[reg[g.physicalControl], reg[g.physicalTarget]])
                continue

            original_op = uid_to_op_node[g.gateOp.uid]
            if g.physicalControl >= 0:
                mapped_dag.apply_operation_back(original_op.op, cargs=original_op.cargs, qargs=[
                    reg[g.physicalControl],
                    reg[g.physicalTarget]
                ])
            else:
                mapped_dag.apply_operation_back(original_op.op, cargs=original_op.cargs, qargs=[
                    reg[g.physicalTarget]
                ])

        # Print result
        # TODO: remove. This is just for debugging.
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

        if self.search_cycles != 0:
            self.update_layout(result)

        return mapped_dag

    def update_layout(self, result: toqm.ToqmResult):
        layout = self.property_set['layout']

        # Need to copy this mapping since layout updates
        # might overwrite original vbits we need to read!
        p2v = layout.get_physical_bits().copy()

        # Update the layout if TOQM made changes.
        for vidx in range(result.numLogicalQubits):
            pidx = result.inferredLaq[vidx]

            if pidx != vidx:
                # Bit was remapped!
                # First, we need to get the original virtual bit from the layout.
                vbit = p2v[vidx]

                # Then, map updated pidx from TOQM to original virtual bit.
                layout[pidx] = vbit

        # Bits after the last logical qubit are ancilla.
        ancilla_vbits = [p2v[vidx] for vidx in range(result.numLogicalQubits, result.numPhysicalQubits)]

        # Map any unmapped physical bits to ancilla.
        for pidx, vidx in enumerate(result.inferredQal):
            if vidx < 0:
                # Current physical bit isn't mapped. Map it to an ancialla.
                layout[pidx] = ancilla_vbits.pop(0)
