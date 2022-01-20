#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <libtoqm/ToqmMapper.hpp>
#include <libtoqm/CostFunc/CXFrontier.hpp>
#include <libtoqm/CostFunc/CXFull.hpp>
#include <libtoqm/CostFunc/SimpleCost.hpp>
#include <libtoqm/Expander/DefaultExpander.hpp>
#include <libtoqm/Expander/GreedyTopK.hpp>
#include <libtoqm/Expander/NoSwaps.hpp>
#include <libtoqm/Filter/HashFilter.hpp>
#include <libtoqm/Filter/HashFilter2.hpp>
#include <libtoqm/Latency/Latency_1.hpp>
#include <libtoqm/Latency/Latency_1_2_6.hpp>
#include <libtoqm/Latency/Latency_1_3.hpp>
#include <libtoqm/Latency/Table.hpp>
#include <libtoqm/NodeMod/GreedyMapper.hpp>
#include <libtoqm/Queue/DefaultQueue.hpp>
#include <libtoqm/Queue/TrimSlowNodes.hpp>
#include <utility>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

using namespace toqm;
namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
	py::class_<CouplingMap>(m, "CouplingMap")
			.def(py::init<unsigned int, std::set<std::pair<int, int>>>())
			.def_readwrite("numPhysicalQubits", &CouplingMap::numPhysicalQubits)
			.def_readwrite("edges", &CouplingMap::edges);
	
	py::class_<GateOp>(m, "GateOp")
	        .def(py::init<std::string, int, int>())
			.def_readwrite("type", &GateOp::type)
			.def_readwrite("target", &GateOp::target)
			.def_readwrite("control", &GateOp::control);
	
	py::class_<ScheduledGateOp>(m, "ScheduledGateOp")
			.def_readwrite("gateOp", &ScheduledGateOp::gateOp)
			.def_readwrite("physicalTarget", &ScheduledGateOp::physicalTarget)
			.def_readwrite("physicalControl", &ScheduledGateOp::physicalControl)
			.def_readwrite("cycle", &ScheduledGateOp::cycle)
			.def_readwrite("latency", &ScheduledGateOp::latency);
	
	py::class_<ToqmResult>(m, "ToqmResult")
			.def_readwrite("scheduledGates", &ToqmResult::scheduledGates)
			.def_readwrite("remainingInQueue", &ToqmResult::remainingInQueue)
			.def_readwrite("numPhysicalQubits", &ToqmResult::numPhysicalQubits)
			.def_readwrite("numLogicalQubits", &ToqmResult::numLogicalQubits)
			.def_readwrite("laq", &ToqmResult::laq)
			.def_readwrite("inferredQal", &ToqmResult::inferredQal)
			.def_readwrite("inferredLaq", &ToqmResult::inferredLaq)
			.def_readwrite("idealCycles", &ToqmResult::idealCycles)
			.def_readwrite("numPopped", &ToqmResult::numPopped)
			.def_readwrite("filterStats", &ToqmResult::filterStats);
	
	py::class_<Queue>(m, "Queue");
	py::class_<DefaultQueue, Queue>(m, "DefaultQueue").def(py::init<>());
	py::class_<TrimSlowNodes, Queue>(m, "TrimSlowNodes")
	        .def(py::init<>())
			.def(py::init<int, int>());
	
	py::class_<Expander>(m, "Expander");
	py::class_<DefaultExpander, Expander>(m, "DefaultExpander").def(py::init<>());
	py::class_<GreedyTopK, Expander>(m, "GreedyTopK").def(py::init<unsigned int>());
	py::class_<NoSwaps, Expander>(m, "NoSwaps").def(py::init<>());
	
	py::class_<CostFunc>(m, "CostFunc");
	py::class_<CXFrontier, CostFunc>(m, "CXFrontier").def(py::init<>());
	py::class_<CXFull, CostFunc>(m, "CXFull").def(py::init<>());
	py::class_<SimpleCost, CostFunc>(m, "SimpleCost").def(py::init<>());
	
	py::class_<Latency>(m, "Latency");
	py::class_<Latency_1, Latency>(m, "Latency_1").def(py::init<>());
	py::class_<Latency_1_2_6, Latency>(m, "Latency_1_2_6").def(py::init<>());
	py::class_<Latency_1_3, Latency>(m, "Latency_1_3").def(py::init<>());
	py::class_<Table, Latency>(m, "Table").def(py::init<std::istream&>());
	
	py::class_<NodeMod>(m, "NodeMod");
	py::class_<GreedyMapper, NodeMod>(m, "GreedyMapper").def(py::init<>());
	
	py::class_<Filter>(m, "Filter");
	py::class_<HashFilter, Filter>(m, "HashFilter").def(py::init<>());
	py::class_<HashFilter2, Filter>(m, "HashFilter2").def(py::init<>());
	
	py::class_<ToqmMapper>(m, "ToqmMapper")
			.def(py::init([](const Queue& node_queue,
							 const Expander& expander,
							 const CostFunc& cost_func,
							 const Latency& latency,
							 const py::list& node_mods,
							 const py::list& filters) {
				
				auto queue_factory = [&]() {
					return node_queue.clone();
				};
				
				std::vector<std::unique_ptr<NodeMod>> nms{};
				nms.reserve(node_mods.size());
				
				for (const auto & i : node_mods) {
					nms.emplace_back(i.cast<NodeMod*>()->clone());
				}
				
				std::vector<std::unique_ptr<Filter>> fs{};
				fs.reserve(filters.size());
				
				for (const auto & i : filters) {
					fs.emplace_back(i.cast<Filter*>()->clone());
				}
				
				return std::unique_ptr<ToqmMapper>(new ToqmMapper(
						std::move(queue_factory),
						expander.clone(),
						cost_func.clone(),
						latency.clone(),
						std::move(nms),
						std::move(fs)));
			}))
			.def("setInitialSearchCycles", &ToqmMapper::setInitialSearchCycles)
			.def("setRetainPopped", &ToqmMapper::setRetainPopped)
			.def("setInitialMappingQal", &ToqmMapper::setInitialMappingQal)
			.def("setInitialMappingLaq", &ToqmMapper::setInitialMappingLaq)
			.def("clearInitialMapping", &ToqmMapper::clearInitialMapping)
			.def("setVerbose", &ToqmMapper::setVerbose)
			.def("run", &ToqmMapper::run);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
