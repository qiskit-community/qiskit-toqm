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


class ToqmProfile:
    def __init__(self):
        self.perform_layout = None
        self.coupling_map = None
        self.latency_descriptions = None

    def on_pass_init(self, perform_layout, coupling_map, latency_descriptions):
        """
        Called by ``ToqmSwap`` during pass initialization with information about
        the transpilation.
        :param perform_layout: If false, produced mappers MUST NOT change the layout.
        :param coupling_map: The coupling map of the target.
        :param latency_descriptions: The latency descriptions for all target gates,
            including swaps.
        """
        self.perform_layout = perform_layout
        self.coupling_map = coupling_map
        self.latency_descriptions = latency_descriptions

    def get_mapper(self, dag):
        """
        Return a native ``ToqmMapper`` instance suitable for the provided DAG.
        :param dag: The provided DAG.
        :return: A native ``ToqmMapper``.
        """
        pass

    def _default_optimal_mapper(self):
        return toqm.ToqmMapper(
            toqm.DefaultQueue(),
            toqm.DefaultExpander(),
            toqm.CXFrontier(),
            toqm.Table(self.latency_descriptions),
            [],
            [toqm.HashFilter(), toqm.HashFilter2()],
            -1 if self.perform_layout else 0
        )


class ToqmProfileO1(ToqmProfile):
    def __init__(self, optimality_threshold=6):
        """
        Initializer.
        :param optimality_threshold: The number of qubits at which returned
            native mappers should begin using a non-optimal heuristic configuration.
        """
        super().__init__()
        self.threshold = optimality_threshold

    def get_mapper(self, dag):
        if self.coupling_map.size < self.threshold:
            return self._default_optimal_mapper()

        # heuristic config
        mapper = toqm.ToqmMapper(
            toqm.TrimSlowNodes(2000, 1000),
            toqm.GreedyTopK(10),
            toqm.CXFrontier(),
            toqm.Table(self.latency_descriptions),
            [toqm.GreedyMapper()],
            [],
            -1 if self.perform_layout else 0
        )

        mapper.setRetainPopped(1)
        return mapper


class ToqmProfileO2(ToqmProfile):
    def __init__(self, optimality_threshold=6):
        """
        Initializer.
        :param optimality_threshold: The number of qubits at which returned
            native mappers should begin using a non-optimal heuristic configuration.
        """
        super().__init__()
        self.threshold = optimality_threshold

    def get_mapper(self, dag):
        if self.coupling_map.size < self.threshold:
            return self._default_optimal_mapper()

        # heuristic config
        mapper = toqm.ToqmMapper(
            toqm.TrimSlowNodes(3000, 2000),
            toqm.GreedyTopK(10),
            toqm.CXFrontier(),
            toqm.Table(self.latency_descriptions),
            [toqm.GreedyMapper()],
            [],
            -1 if self.perform_layout else 0
        )

        mapper.setRetainPopped(1)
        return mapper


class ToqmProfileO3(ToqmProfile):
    def __init__(self, optimality_threshold=7):
        """
        Initializer.
        :param optimality_threshold: The number of qubits at which returned
            native mappers should begin using a non-optimal heuristic configuration.
        """
        super().__init__()
        self.threshold = optimality_threshold

    def get_mapper(self, dag):
        if self.coupling_map.size < self.threshold:
            return self._default_optimal_mapper()

        # heuristic config
        mapper = toqm.ToqmMapper(
            toqm.TrimSlowNodes(4000, 3000),
            toqm.GreedyTopK(15),
            toqm.CXFrontier(),
            toqm.Table(self.latency_descriptions),
            [toqm.GreedyMapper()],
            [],
            -1 if self.perform_layout else 0
        )

        mapper.setRetainPopped(1)
        return mapper
