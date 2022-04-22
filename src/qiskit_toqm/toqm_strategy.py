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

import qiskit_toqm.native as toqm


class ToqmStrategy:
    def __init__(self):
        self.coupling_map = None
        self.latency_descriptions = None

    def on_pass_init(self, coupling_map, latency_descriptions):
        """
        Called by ``ToqmSwap`` during pass initialization with information about
        the transpilation.

        Args:
            coupling_map (toqm.CouplingMap): The coupling map of the target.
            latency_descriptions (List[toqm.LatencyDescription]): The latency descriptions for all target gates,
            including swaps.
        """
        self.coupling_map = coupling_map
        self.latency_descriptions = latency_descriptions

    def run(self, gates, num_qubits):
        """
        Run native ToqmMapper and return the native result.

        Args:
            gates (List[toqm.GateOp]): The topologically ordered list of gate operations.
            num_qubits (int): The number of virtual qubits used in the circuit.

        Returns:
            toqm.ToqmResult: The native result.
        """
        pass

    # The following defaults are based on:
    # https://github.com/time-optimal-qmapper/TOQM/blob/main/code/README.txt
    def _default_optimal_mapper(self):
        return toqm.ToqmMapper(
            toqm.DefaultQueue(),
            toqm.DefaultExpander(),
            toqm.CXFrontier(),
            toqm.Table(self.latency_descriptions),
            [],
            [toqm.HashFilter(), toqm.HashFilter2()],
            -1
        )

    def _default_optimal_mapper_no_swaps(self):
        return toqm.ToqmMapper(
            toqm.DefaultQueue(),
            toqm.NoSwaps(),
            toqm.CXFrontier(),
            toqm.Table(self.latency_descriptions),
            [],
            [],
            -1
        )

    # NOTE: currently, the heuristic mapper uses the hard-coded latencies of 1, 2 and 6
    # for 1Q, 2Q and SWAP gates, respectively. This is because when gate-specific latencies
    # are used with heuristic components, the run sometimes never terminates.
    # This is tracked here: https://github.com/qiskit-toqm/libtoqm/issues/15
    def _default_heuristic_mapper(self, max_nodes, min_nodes, k):
        mapper = toqm.ToqmMapper(
            toqm.TrimSlowNodes(max_nodes, min_nodes),
            toqm.GreedyTopK(k),
            toqm.CXFrontier(),
            toqm.Latency_1_2_6(),
            [toqm.GreedyMapper()],
            [],
            0
        )

        mapper.setRetainPopped(1)
        return mapper


class ToqmStrategyO0(ToqmStrategy):
    def __init__(self):
        super().__init__()

    def run(self, gates, num_qubits):
        mapper = self._default_heuristic_mapper(5000, 3000, 1)

        return mapper.run(gates, num_qubits, self.coupling_map)


class ToqmStrategyO1(ToqmStrategy):
    def __init__(self, optimality_threshold=6):
        """
        Initializer.

        Args:
            optimality_threshold (int): The number of qubits at which returned
                native mappers should begin using a non-optimal heuristic configuration.
        """
        super().__init__()
        self.threshold = optimality_threshold

    def run(self, gates, num_qubits):
        if self.coupling_map.numPhysicalQubits < self.threshold:
            mapper = self._default_optimal_mapper()
        else:
            mapper = self._default_heuristic_mapper(800, 400, 5)

        return mapper.run(gates, num_qubits, self.coupling_map)


class ToqmStrategyO2(ToqmStrategy):
    def __init__(self, optimality_threshold=6):
        """
        Initializer.

        Args:
            optimality_threshold (int): The number of qubits at which returned
                native mappers should begin using a non-optimal heuristic configuration.
        """
        super().__init__()
        self.threshold = optimality_threshold

    def run(self, gates, num_qubits):
        if self.coupling_map.numPhysicalQubits < self.threshold:
            mapper = self._default_optimal_mapper()
        else:
            mapper = self._default_heuristic_mapper(1000, 400, 11)

        return mapper.run(gates, num_qubits, self.coupling_map)


class ToqmStrategyO3(ToqmStrategy):
    def __init__(self, optimality_threshold=6):
        """
        Initializer.

        Args:
            optimality_threshold (int): The number of qubits at which returned
                native mappers should begin using a non-optimal heuristic configuration.
        """
        super().__init__()
        self.threshold = optimality_threshold

    def run(self, gates, num_qubits):
        if self.coupling_map.numPhysicalQubits < self.threshold:
            # try no swaps first
            try:
                return self._default_optimal_mapper_no_swaps().run(gates, num_qubits, self.coupling_map)
            except RuntimeError:
                mapper = self._default_optimal_mapper()
        else:
            mapper = self._default_heuristic_mapper(4800, 3600, 3)

        return mapper.run(gates, num_qubits, self.coupling_map)
