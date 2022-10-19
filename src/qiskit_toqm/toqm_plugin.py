from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin
from qiskit.transpiler.preset_passmanagers import common
from qiskit_toqm import *


class ToqmSwapPlugin(PassManagerStagePlugin):

    def pass_manager(self, pass_manager_config, optimization_level):
        opt_level_to_strategy = {
            0: ToqmStrategyO0,
            1: ToqmStrategyO1,
            2: ToqmStrategyO2,
            3: ToqmStrategyO3,
        }

        toqm_strategy_preset = opt_level_to_strategy.get(optimization_level, default=ToqmStrategyO3)

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

        routing_pm = common.generate_routing_passmanager(
            routing_pass,
            pass_manager_config.target,
            coupling_map=pass_manager_config.coupling_map,
            seed_transpiler=pass_manager_config.seed_transpiler,
            use_barrier_before_measurement=False,
        )

        translation_pm = common.generate_translation_passmanager(
            pass_manager_config.target,
            pass_manager_config.basis_gates,
            pass_manager_config.translation_method,
            pass_manager_config.approximation_degree,
            pass_manager_config.coupling_map,
            pass_manager_config.backend_properties,
            pass_manager_config.unitary_synthesis_method,
            pass_manager_config.unitary_synthesis_plugin_config,
            pass_manager_config.hls_config,
        )

        out = common.generate_error_on_control_flow(
            "TOQM routing does not yet support circuits with control flow."
        )
        out += translation_pm
        out += routing_pm

        return out
