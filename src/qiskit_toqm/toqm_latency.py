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

import qiskit
from qiskit.transpiler import TranspilerError
import qiskit_toqm.native as toqm

from itertools import chain


def _calc_swap_durations(coupling_map, instruction_durations, basis_gates, backend_properties, target):
    """Calculates the durations of swap gates between each coupling on the target."""
    # Filter for couplings that don't already have a native swap.
    couplings = [
        c for c in coupling_map.get_edges()
        if ("swap", c) not in instruction_durations.duration_by_name_qubits
    ]

    if not couplings:
        return

    backend_aware = target is not None or (basis_gates is not None and backend_properties is not None)
    if not backend_aware:
        raise TranspilerError(
            "`target` must be specified, or both 'basis_gates' and 'backend_properties' must be specified unless"
            "'instruction_durations' has durations for all swap gates."
        )

    def gen_swap_circuit(s, t):
        # Generates a circuit with a single swap gate between src and tgt
        c = qiskit.QuantumCircuit(coupling_map.size())
        c.swap(s, t)
        return c

    # Batch transpile generated swap circuits
    swap_circuits = qiskit.transpile(
        [gen_swap_circuit(*pair) for pair in couplings],
        basis_gates=basis_gates,
        coupling_map=coupling_map,
        backend_properties=backend_properties if target is None else None,
        instruction_durations=instruction_durations,
        target=target,
        optimization_level=0,
        layout_method="trivial",
        scheduling_method="asap"
    )

    for (src, tgt), qc in zip(couplings, swap_circuits):
        if instruction_durations.dt is None and qc.unit == "dt":
            # TODO: should be able to convert by looking up an op in both
            raise TranspilerError("Incompatible units.")

        duration = qc.qubit_duration(src, tgt)
        yield src, tgt, duration


def latencies_from_target(
    coupling_map=None,
    instruction_durations=None,
    basis_gates=None,
    backend_properties=None,
    target=None,
    normalize_scale=2
):
    """
    Generate a list of native ``LatencyDescription`` objects for
    the specified target device.

    Args:
        coupling_map (Optional[CouplingMap]): CouplingMap of the target backend.
            Required unless ``target`` is specified.
        instruction_durations (Optional[InstructionDurations]): Durations for gates
            in the target's basis. Must include durations for all gates
            that appear in input DAGs other than ``swap`` (for which
            durations are calculated through decomposition if not supplied).
            Required unless ``target`` is specified.
        basis_gates (Optional[List[str]]): The list of basis gates for the
            target. Required unless ``instruction_durations``
            contains durations for all swap gates or ``target`` is specified.
        backend_properties (Optional[BackendProperties]): The backend
            properties of the target. Required unless
            ``instruction_durations`` contains durations for all swap gates
            or ``target`` is specified.
        target (Optional[Target]): The backend transpiler target. If specified,
            overrides ``coupling_map``, ``instruction_durations``, ``basis_gates``,
            and ``backend_properties``. All gates that appear in input DAG other
            than ``swap`` must be supported by the target and have duration
            information available therein.
        normalize_scale (int): Multiple by this factor when converting
            relative durations to cycle count. The conversion is:
            cycles = ceil(duration * NORMALIZE_SCALE / min_duration)
            where min_duration is the length of the fastest non-zero duration
            instruction on the target.
    """
    if target is not None:
        coupling_map = target.build_coupling_map()
        instruction_durations = target.durations()
        basis_gates = target.operation_names

    unit = "dt" if instruction_durations.dt else "s"

    swap_durations = list(_calc_swap_durations(coupling_map, instruction_durations, basis_gates, backend_properties, target))
    default_op_durations = [
        (op_name, instruction_durations.get(op_name, [], unit))
        for op_name in instruction_durations.duration_by_name
    ]
    op_durations = [
        (op_name, bits, instruction_durations.get(op_name, bits, unit))
        for (op_name, bits) in instruction_durations.duration_by_name_qubits
    ]

    non_zero_durations = [d for d in chain(
        (d for (_, d) in default_op_durations),
        (d for (_, _, d) in op_durations),
        (d for (_, _, d) in swap_durations)
    ) if d > 0]

    if not non_zero_durations:
        raise TranspilerError("Durations must be specified for the target.")

    min_duration = min(non_zero_durations)

    def normalize(d):
        return round(d * normalize_scale / min_duration)

    # Yield latency descriptions with durations interpolated to cycles.
    for op_name, duration in default_op_durations:
        # We don't know if the instruction is for 1 or 2 qubits, so emit
        # defaults for both.
        yield toqm.LatencyDescription(1, op_name, normalize(duration))
        yield toqm.LatencyDescription(2, op_name, normalize(duration))

    for op_name, qubits, duration in op_durations:
        yield toqm.LatencyDescription(op_name, *qubits, normalize(duration))

    for src, tgt, duration in swap_durations:
        yield toqm.LatencyDescription("swap", src, tgt, normalize(duration))


def latencies_from_simple(one_qubit_cycles, two_qubit_cycles, swap_cycles):
    """
    Generate a list of native ``LatencyDescription`` objects for
    the specified hard-coded cycle counts.

    The resulting latency descriptions describe a circuit in which
    all 1Q, 2Q, and SWAP gates execute in the corresponding number
    of cycles, irrespective of which qubits they execute on.

    Args:
        one_qubit_cycles (int): The number of cycles for all 1Q gates.
        two_qubit_cycles (int): The number of cycles for all 2Q gates.
        swap_cycles (int): The number of cycles for all swap gates.
    """
    return [
        toqm.LatencyDescription(1, one_qubit_cycles),
        toqm.LatencyDescription(2, two_qubit_cycles),
        toqm.LatencyDescription(2, "swap", swap_cycles)
    ]
