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

import qiskit_toqm.native as toqm

from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.dagcircuit import DAGCircuit

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

logger = logging.getLogger(__name__)


class ToqmSwap(TransformationPass):
    r"""Map input circuit onto a backend topology via insertion of SWAPs.
    Implementation of the SWAP-based approach from Time-Optimal Qubit
    Mapping paper [1].
    **References:**
    [1] Chi Zhang, Ari B. Hayes, Longfei Qiu, Yuwei Jin, Yanhao Chen, and Eddy Z. Zhang. 2021. Time-Optimal Qubit
    Mapping. In Proceedings of the 26th ACM International Conference on Architectural Support for Programming
    Languages and Operating Systems (ASPLOS ’21), April 19–23, 2021, Virtual, USA.
    ACM, New York, NY, USA, 14 pages.
    `<https://doi.org/10.1145/3445814.3446706>`_
    """

    def __init__(
            self,
            coupling_map,
            strategy):
        """
        ToqmSwap initializer.

        Args:
            coupling_map (CouplingMap): CouplingMap of the target backend.
            strategy (typing.Callable[[List[toqm.GateOp], int, toqm.CouplingMap], toqm.ToqmResult]):
                A callable responsible for running the native ``ToqmMapper`` and
                returning a native ``ToqmResult``.
        """
        super().__init__()

        if coupling_map is None:
            # We cannot construct a proper TOQM mapper without a coupling map,
            # but we gracefully handle construction without one, and then
            # assert that `run` is never called on this instance.
            return

        if coupling_map.size() > 127:
            raise TranspilerError("ToqmSwap currently supports a max of 127 qubits.")

        self.coupling_map = coupling_map
        self.toqm_strategy = strategy
        self.toqm_result = None

    def run(self, dag: DAGCircuit):
        """Run the ToqmSwap pass on `dag`.
        Args:
            dag (DAGCircuit): the directed acyclic graph to be mapped.
        Returns:
            DAGCircuit: A dag mapped to be compatible with the coupling_map.
        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG
        """
        if self.coupling_map is None:
            raise TranspilerError("TOQM swap not properly initialized.")

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
                    raise TranspilerError(f"ToqmSwap only works with 1q and 2q gates! "
                                          f"Bad gate: {node.op.name} {node.qargs}")

        gate_ops = list(gates())
        edges = {e for e in self.coupling_map.get_edges()}
        couplings = toqm.CouplingMap(self.coupling_map.size(), edges)

        self.toqm_result = self.toqm_strategy(gate_ops, dag.num_qubits(), couplings)

        # Preserve input DAG's name, regs, wire_map, etc. but replace the graph.
        mapped_dag = dag.copy_empty_like()

        for g in self.toqm_result.scheduledGates:
            if g.gateOp.type.lower() == "swap":
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

        self._update_layout()

        return mapped_dag

    def _update_layout(self):
        layout = self.property_set['layout']

        # Need to copy this mapping since layout updates
        # might overwrite original vbits we need to read!
        p2v = layout.get_physical_bits().copy()

        # Update the layout if TOQM made changes.
        ancilla_vbits = []
        for vidx in range(self.toqm_result.numPhysicalQubits):
            pidx = self.toqm_result.inferredLaq[vidx]

            if pidx == -1:
                # bit is not mapped to physical qubit
                ancilla_vbits.append(p2v[vidx])
                continue

            if pidx != vidx:
                # Bit was remapped!
                # First, we need to get the original virtual bit from the layout.
                vbit = p2v[vidx]

                # Then, map updated pidx from TOQM to original virtual bit.
                layout[pidx] = vbit

        # Map any unmapped physical bits to ancilla.
        for pidx, vidx in enumerate(self.toqm_result.inferredQal):
            if vidx < 0:
                # Current physical bit isn't mapped. Map it to an ancilla.
                layout[pidx] = ancilla_vbits.pop(0)
