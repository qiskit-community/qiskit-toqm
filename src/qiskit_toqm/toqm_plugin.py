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

from qiskit.transpiler import TranspilerError
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin
from qiskit.transpiler.preset_passmanagers import common
from qiskit_toqm import *


class ToqmSwapPlugin(PassManagerStagePlugin):

    def pass_manager(self, pass_manager_config, optimization_level):
        if pass_manager_config.initial_layout:
            raise TranspilerError("Initial layouts are not supported with TOQM-based routing.")

        opt_level_to_strategy = {
            0: ToqmStrategyO0,
            1: ToqmStrategyO1,
            2: ToqmStrategyO2,
            3: ToqmStrategyO3,
        }

        toqm_strategy_preset = opt_level_to_strategy.get(optimization_level, ToqmStrategyO3)

        toqm_latencies = latencies_from_target(
            pass_manager_config.coupling_map,
            pass_manager_config.instruction_durations,
            pass_manager_config.basis_gates,
            pass_manager_config.backend_properties,
            pass_manager_config.target
        )

        routing_pass = ToqmSwap(
            pass_manager_config.coupling_map,
            strategy=toqm_strategy_preset(toqm_latencies),
        )

        vf2_call_limit = common.get_vf2_call_limit(
            optimization_level, pass_manager_config.layout_method, pass_manager_config.initial_layout
        )

        routing_pm = common.generate_routing_passmanager(
            routing_pass,
            pass_manager_config.target,
            coupling_map=pass_manager_config.coupling_map,
            vf2_call_limit=vf2_call_limit,
            seed_transpiler=pass_manager_config.seed_transpiler,
            use_barrier_before_measurement=False,
        )

        translation_pm = common.generate_translation_passmanager(
            pass_manager_config.target,
            pass_manager_config.basis_gates,
            pass_manager_config.translation_method or "translator",
            pass_manager_config.approximation_degree,
            pass_manager_config.coupling_map,
            pass_manager_config.backend_properties,
            pass_manager_config.unitary_synthesis_method or "default",
            pass_manager_config.unitary_synthesis_plugin_config,
            pass_manager_config.hls_config,
        )

        out = common.generate_error_on_control_flow(
            "TOQM routing does not yet support circuits with control flow."
        )
        out += translation_pm
        out += routing_pm

        return out
