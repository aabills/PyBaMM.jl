import pybamm
from copy import deepcopy
import networkx as nx
import numpy as np
import pandas as pd
from collections import OrderedDict
import pybamm2julia
import pack


class CoarsePack(pack.Pack):
    def __init__(
        self,
        full_model,
        reduced_model,
        full_cells,
        netlist,
        parameter_values=None,
        functional=False,
        voltage_functional=False,
        thermal=False,
        build_jac=False,
        implicit=False,
        top_bc = "ambient",
        bottom_bc = "ambient",
        left_bc = "ambient",
        right_bc = "ambient",
        distribution_params = None,
        operating_mode = "CC",
        input_parameter_order = None,
        initial_soc = 1.0
    ):
        # this is going to be a work in progress for a while:
        # for now, will just do it at the julia level

        # Build the cell expression tree with necessary parameters.
        # think about moving this to a separate function.
        self.top_bc = top_bc
        self.bottom_bc = bottom_bc
        self.left_bc = left_bc
        self.right_bc = right_bc
        self._operating_mode = operating_mode

        self._input_parameter_order = input_parameter_order
        self._full_models = full_cells

        self._implicit = implicit

        self.functional = functional
        self.voltage_functional = voltage_functional
        if self.voltage_functional:
            self.full_voltage_func = None
            self.reduced_voltage_func = None
        self.build_jac = build_jac
        self._thermal = thermal

        if parameter_values is None:
            parameter_values = full_model.default_parameter_values
        
        self._distribution_params = {}
        if distribution_params is not None:
            if not self.functional:
                raise NotImplementedError(
                    "Distribution Parameters only works with functional packs!"
                )
            for param in distribution_params:
                dp = pybamm2julia.DistributionParameter(
                    distribution_params[param]["name"],
                    distribution_params[param]["mean"],
                    distribution_params[param]["stddev"]
                )
                parameter_values[param] = dp
                self._distribution_params[param] = dp
        else:
            self._distribution_params = None


        cell_current = pybamm2julia.PsuedoInputParameter("cell_current")
        self.cell_current = cell_current
        parameter_values.update({"Current function [A]": cell_current})

        if self._thermal:
            self.pack_ambient = pybamm.Scalar(
                parameter_values["Ambient temperature [K]"]
            )
            ambient_temperature = pybamm2julia.PsuedoInputParameter("ambient_temperature")
            self.ambient_temperature = ambient_temperature
            parameter_values.update({"Ambient temperature [K]": ambient_temperature})

        self.cell_parameter_values = parameter_values
        
        self.unbuilt_full_model = full_model
        self.unbuilt_reduced_model = reduced_model

        full_sim = pybamm.Simulation(full_model, parameter_values=parameter_values)
        reduced_sim = pybamm.Simulation(reduced_model, parameter_values=parameter_values)

        if initial_soc is None:
            full_sim.build()
            reduced_sim.build()
        else:
            full_sim.build(initial_soc = initial_soc)
            reduced_sim.build(initial_soc = initial_soc)
        
        self.len_full_cell_rhs = full_sim.built_model.len_rhs
        self.len_reduced_cell_rhs = reduced_sim.built_model.len_rhs

        
        if self._implicit:
            self.full_cell_model = pybamm.numpy_concatenation(
                (pybamm.StateVectorDot(slice(0,self.len_full_cell_rhs)) - full_sim.built_model.concatenated_rhs), full_sim.built_model.concatenated_algebraic
            )
            self.reduced_cell_model = pybamm.numpy_concatenation(
                (pybamm.StateVectorDot(slice(0,self.len_reduced_cell_rhs)) - reduced_sim.built_model.concatenated_rhs), reduced_sim.built_model.concatenated_algebraic
            )
        else:
            self.full_cell_model = pybamm.numpy_concatenation(
                full_sim.built_model.concatenated_rhs, full_sim.built_model.concatenated_algebraic
            )
            self.reduced_cell_model = pybamm.numpy_concatenation(
                reduced_sim.built_model.concatenated_rhs, reduced_sim.built_model.concatenated_algebraic
            )

        #really hope these are the same lol
        self.full_timescale = full_sim.built_model.timescale
        self.reduced_timescale = reduced_sim.built_model.timescale

        self.len_full_cell_algebraic = full_sim.built_model.len_alg
        self.len_reduced_cell_algebraic = reduced_sim.built_model.len_alg

        self.full_cell_size = self.full_cell_model.shape[0]
        self.reduced_cell_size = self.reduced_cell_model.shape[0]

        self.full_built_model = full_sim.built_model
        self.reduced_built_model = reduced_sim.built_model


        if self.functional:
            full_sv = pybamm.StateVector(slice(0, self.full_cell_size))
            reduced_sv = pybamm.StateVector(slice(0, self.reduced_cell_size))
            full_dsv = pybamm.StateVectorDot(slice(0,self.len_full_cell_rhs))
            reduced_dsv = pybamm.StateVectorDot(slice(0,self.len_reduced_cell_rhs))


            if self._distribution_params is None:
                lsv = []
            else:
                lsv = list(self._distribution_params.values())
            

            if self._input_parameter_order is not None:
                for p in self._input_parameter_order:
                    psuedo_parameter = pybamm2julia.PsuedoInputParameter(p)
                    psuedo_parameter.children = [pybamm.InputParameter(p)]
                    psuedo_parameter.set_id()
                    lsv.append(psuedo_parameter)

            if self._implicit:
                if self._thermal:
                    self.full_cell_model = pybamm2julia.PybammJuliaFunction(
                        [full_sv, cell_current, ambient_temperature, full_dsv] + lsv,
                        self.full_cell_model,
                        "full_cell!",
                        True,
                    )
                    self.reduced_cell_model = pybamm2julia.PybammJuliaFunction(
                        [reduced_sv, cell_current, ambient_temperature, reduced_dsv] + lsv,
                        self.reduced_cell_model,
                        "reduced_cell!",
                        True,
                    )
                else:
                    self.full_cell_model = pybamm2julia.PybammJuliaFunction(
                        [full_sv, cell_current, full_dsv] + lsv, self.full_cell_model, "full_cell!", True
                    )
                    self.reduced_cell_model = pybamm2julia.PybammJuliaFunction(
                        [reduced_sv, cell_current, reduced_dsv] + lsv, self.reduced_cell_model, "reduced_cell!", True
                    )
            else:
                if self._thermal:
                    self.full_cell_model = pybamm2julia.PybammJuliaFunction(
                        [full_sv, cell_current, ambient_temperature] + lsv,
                        self.full_cell_model,
                        "full_cell!",
                        True,
                    )
                    self.reduced_cell_model = pybamm2julia.PybammJuliaFunction(
                        [reduced_sv, cell_current, ambient_temperature] + lsv,
                        self.reduced_cell_model,
                        "reduced_cell!",
                        True,
                    )
                else:
                    self.full_cell_model = pybamm2julia.PybammJuliaFunction(
                        [full_sv, cell_current] + lsv, self.full_cell_model, "full_cell!", True
                    )
                    self.reduced_cell_model = pybamm2julia.PybammJuliaFunction(
                        [reduced_sv, cell_current] + lsv, self.reduced_cell_model, "reduced_cell!", True
                    )
        self._sv_done = []


        self.netlist = netlist
        self.process_netlist()

        # get x and y coords for nodes from graph.
        node_xs = [n for n in range(max(self.circuit_graph.nodes) + 1)]
        node_ys = [n for n in range(max(self.circuit_graph.nodes) + 1)]
        for row in netlist.itertuples():
            node_xs[row.node1] = row.node1_x
            node_ys[row.node1] = row.node1_y
        self.node_xs = node_xs
        self.node_ys = node_ys
        self.batt_string = None


    def build_pack(self):
        # this function builds expression trees to compute the current.

        # cycle basis is the list of loops over which we will do kirchoff mesh analysis
        mcb = nx.cycle_basis(self.circuit_graph)
        self.cyc_basis = deepcopy(mcb)

        # generate loop currents and current source voltages-- this is what we don't know.
        num_loops = len(mcb)

        curr_sources = [
            edge
            for edge in self.circuit_graph.edges
            if self.circuit_graph.edges[edge]["desc"][0] == "I"
        ]

        loop_currents = [
            pybamm.StateVector(slice(n, n + 1), name="current_{}".format(n))
            for n in range(num_loops)
        ]

        curr_sources = []
        n = num_loops
        for edge in self.circuit_graph.edges:
            if self.circuit_graph.edges[edge]["desc"][0] == "I":
                self.circuit_graph.edges[edge]["voltage"] = pybamm.StateVector(
                    slice(n, n + 1), name="current_source_{}".format(n)
                )
                n += 1
                curr_sources.append(edge)

        # now we know the offset, we should "build" the batteries here. will still need to replace the currents later.
        self.offset = num_loops + len(curr_sources)
        self.batteries = OrderedDict()
        for index, row in self.netlist.iterrows():
            desc = row["desc"]
            # I'd like a better way to do this.
            if desc[0] == "V":
                if desc in self._full_models:
                    new_cell = self.add_new_full_cell()
                    terminal_voltage = self.get_new_full_terminal_voltage()
                    ics = self.get_new_full_ics()
                else:
                    new_cell = self.add_new_reduced_cell()
                    terminal_voltage = self.get_new_reduced_terminal_voltage()
                    ics = self.get_new_reduced_ics()
                self.batteries[desc] = {
                    "cell": new_cell,
                    "voltage": terminal_voltage,
                    "current_replaced": False,
                    "ics": ics
                }
                if self._thermal:
                    node1_x = row["node1_x"]
                    node2_x = row["node2_x"]
                    node1_y = row["node1_y"]
                    node2_y = row["node2_y"]
                    if node1_x != node2_x:
                        raise AssertionError("x's must be the same")
                    if abs(node1_y - node2_y) != 1:
                        raise AssertionError("batteries can only take up one y")
                    batt_y = min(node1_y, node2_y) + 0.5
                    if desc in self._full_models:
                        temperature = self.get_new_full_cell_temperature()
                    else:
                        temperature = self.get_new_reduced_cell_temperature()
                    self.batteries[desc].update(
                        {"x": node1_x, "y": batt_y, "temperature": temperature}
                    )
                params = {}
                if self._distribution_params is not None:
                    for param in self._distribution_params:
                        expr = self._distribution_params[param].sample()
                        self._distribution_params[param].set_psuedo(self.batteries[desc]["cell"], expr)
                        self._distribution_params[param].set_psuedo(self.batteries[desc]["ics"], expr)
                        if self.voltage_functional:
                            self._distribution_params[param].set_psuedo(self.batteries[desc]["cell"], expr)
                        params.update({self._distribution_params[param] : expr})
                        
                self.batteries[desc].update({"offset": self.offset})
                self.batteries[desc].update({"distribution parameters" : params})
                if desc in self._full_models:
                    self.offset += self.full_cell_size
                else:
                    self.offset += self.reduced_cell_size

        if self._thermal:
            self.build_thermal_equations()

        self.num_cells = len(self.batteries)

        if len(curr_sources) != 1:
            raise NotImplementedError("can't do this yet")
        # copy the basis which we can use to place the loop currents
        basis_to_place = deepcopy(mcb)
        self.place_currents(loop_currents, basis_to_place)
        pack_eqs_vec = self.build_pack_equations(loop_currents, curr_sources)
        self.len_pack_eqs = len(pack_eqs_vec)
        pack_eqs = pybamm.numpy_concatenation(*pack_eqs_vec)

        cells = [d["cell"] for d in self.batteries.values()]
        cell_eqs = pybamm.numpy_concatenation(*cells)




        self.pack = pybamm.numpy_concatenation(pack_eqs, cell_eqs)
        if self._implicit:
            len_sv = len(cells)*self.cell_size + len(pack_eqs_vec)
            sv = pybamm.StateVector(slice(0,len_sv))
            dsv = pybamm.StateVectorDot(slice(0,len_sv))
            p = pybamm2julia.PsuedoInputParameter("lolol")
            t = pybamm.Time()
            self.pack = pybamm2julia.PybammJuliaFunction(
                [dsv, sv, p, t], self.pack, "pack", True
            )
        self.ics = self.initialize_pack(len(loop_currents),1)
    
    def add_new_full_cell(self):
        new_model = deepcopy(self.full_cell_model)
        # at some point need to figure out parameters
        my_offsetter = pack.offsetter(self.offset)
        my_offsetter.add_offset_to_state_vectors(new_model)
        new_model.set_id()
        return new_model
    
    def add_new_reduced_cell(self):
        new_model = deepcopy(self.reduced_cell_model)
        # at some point need to figure out parameters
        my_offsetter = pack.offsetter(self.offset)
        my_offsetter.add_offset_to_state_vectors(new_model)
        new_model.set_id()
        return new_model

    def get_new_full_terminal_voltage(self):
        symbol = deepcopy(self.full_built_model.variables["Terminal voltage [V]"])
        my_offsetter = pack.offsetter(self.offset)
        if self.voltage_functional:
            sv = pybamm.StateVector(slice(0,self.len_full_cell_rhs+self.len_full_cell_algebraic))
            if self.full_voltage_func is None:
                if self._distribution_params is not None:
                    ldp = list(self._distribution_params)
                else:
                    ldp = []
                voltage_func = pybamm2julia.PybammJuliaFunction([sv, self.cell_current] + ldp, symbol, "full_voltage_func", True)
                self.full_voltage_func = voltage_func
            symbol = deepcopy(self.full_voltage_func)
        my_offsetter.add_offset_to_state_vectors(symbol)
        return symbol
    
    def get_new_reduced_terminal_voltage(self):
        symbol = deepcopy(self.reduced_built_model.variables["Terminal voltage [V]"])
        my_offsetter = pack.offsetter(self.offset)
        if self.voltage_functional:
            sv = pybamm.StateVector(slice(0,self.len_reduced_cell_rhs+self.len_reduced_cell_algebraic))
            if self.reduced_voltage_func is None:
                if self._distribution_params is not None:
                    ldp = list(self._distribution_params)
                else:
                    ldp = []
                voltage_func = pybamm2julia.PybammJuliaFunction([sv, self.cell_current] + ldp, symbol, "reduced_voltage_func", True)
                self.reduced_voltage_func = voltage_func
            symbol = deepcopy(self.reduced_voltage_func)
        my_offsetter.add_offset_to_state_vectors(symbol)
        return symbol

    def get_new_full_cell_temperature(self):
        symbol = pybamm.Index(
            deepcopy(self.full_built_model.variables["Cell temperature [K]"]), slice(0, 1)
        )
        my_offsetter = pack.offsetter(self.offset)
        my_offsetter.add_offset_to_state_vectors(symbol)
        return symbol
    
    def get_new_reduced_cell_temperature(self):
        symbol = pybamm.Index(
            deepcopy(self.reduced_built_model.variables["Cell temperature [K]"]), slice(0, 1)
        )
        my_offsetter = pack.offsetter(self.offset)
        my_offsetter.add_offset_to_state_vectors(symbol)
        return symbol
    
    def get_new_full_ics(self):
        ics = self.full_built_model.concatenated_initial_conditions
        return ics
    
    def get_new_reduced_ics(self):
        ics = self.reduced_built_model.concatenated_initial_conditions
        return ics

