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


class ToqmHeuristicStrategy:
    def __init__(self, latency_descriptions, top_k, queue_target, queue_max, retain_popped=1):
        """
        Constructs a TOQM strategy that aims to minimize overall circuit duration.
        The priority queue used in the A* is configured to drop the worst nodes
        whenever it reaches the queue size limit. Node expansion is also limited
        to exploring the best ``K`` children.

        Args:
            latency_descriptions (List[toqm.LatencyDescription]): The latency descriptions
            for all gates that will appear in the circuit, including swaps.
            top_k (int): The maximum number of best child nodes that can be pushed to the
            queue during expansion of any given node.
            queue_target (int): When the priority queue reaches capacity, nodes are dropped
            until the size reaches this value.
            queue_max (int): The priority queue capacity.
            retain_popped (int): Final nodes to retain.

        Raises:
            RuntimeError: No routing was found.
        """
        # The following defaults are based on:
        # https://github.com/time-optimal-qmapper/TOQM/blob/main/code/README.txt
        self.mapper = toqm.ToqmMapper(
            toqm.TrimSlowNodes(queue_max, queue_target),
            toqm.GreedyTopK(top_k),
            toqm.CXFrontier(),
            toqm.Table(list(latency_descriptions)),
            [toqm.GreedyMapper()],
            [],
            0
        )

        self.mapper.setRetainPopped(retain_popped)

    def __call__(self, gates, num_qubits, coupling_map):
        """
        Run native ToqmMapper and return the native result.

        Args:
            gates (List[toqm.GateOp]): The topologically ordered list of gate operations.
            num_qubits (int): The number of virtual qubits used in the circuit.
            coupling_map (toqm.CouplingMap): The coupling map of the target.

        Returns:
            toqm.ToqmResult: The native result.
        """
        return self.mapper.run(gates, num_qubits, coupling_map)


class ToqmOptimalStrategy:
    def __init__(self, latency_descriptions, perform_layout=True, no_swaps=False):
        """
        Constructs a TOQM strategy that finds an optimal (minimal) routing
        in terms of overall circuit duration.

        Args:
            latency_descriptions (List[toqm.LatencyDescription]): The latency descriptions
            for all gates that will appear in the circuit, including swaps.
            perform_layout (Boolean): If true, permutes the initial layout rather than
            inserting swap gates at the start of the circuit.
            no_swaps (Boolean): If true, attempts to find a routing without inserting swaps.

        Raises:
            RuntimeError: No routing was found.
        """
        # The following defaults are based on:
        # https://github.com/time-optimal-qmapper/TOQM/blob/main/code/README.txt
        self.mapper = toqm.ToqmMapper(
            toqm.DefaultQueue(),
            toqm.NoSwaps() if no_swaps else toqm.DefaultExpander(),
            toqm.CXFrontier(),
            toqm.Table(latency_descriptions),
            [],
            [toqm.HashFilter(), toqm.HashFilter2()],
            -1 if perform_layout else 0
        )

    def __call__(self, gates, num_qubits, coupling_map):
        """
        Run native ToqmMapper and return the native result.

        Args:
            gates (List[toqm.GateOp]): The topologically ordered list of gate operations.
            num_qubits (int): The number of virtual qubits used in the circuit.
            coupling_map (toqm.CouplingMap): The coupling map of the target.

        Returns:
            toqm.ToqmResult: The native result.
        """
        return self.mapper.run(gates, num_qubits, coupling_map)
