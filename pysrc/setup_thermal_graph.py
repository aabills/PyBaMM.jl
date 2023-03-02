import networkx as nx
from collections import OrderedDict
import pybamm
import copy


class LegacyThermalGraph(object):
    def __init__(
        self,
        circuit_graph,
        top_bc = "ambient",
        bottom_bc = "ambient",
        left_bc = "ambient",
        right_bc = "ambient"    
    ):
        self.thermal_graph = nx.Graph()
        self.top_bc = top_bc
        self.bottom_bc = bottom_bc
        self.left_bc = left_bc
        self.right_bc = right_bc
        self.batteries = OrderedDict()
        for edge in circuit_graph.edges:
            row = circuit_graph.edges[edge]
            desc = row["desc"]
            # I'd like a better way to do this.
            if desc[0] == "V":
                node1_x = row["node1_x"]
                node2_x = row["node2_x"]
                node1_y = row["node1_y"]
                node2_y = row["node2_y"]
                if node1_x != node2_x:
                    raise AssertionError("x's must be the same")
                if abs(node1_y - node2_y) != 1:
                    raise AssertionError("batteries can only take up one y")
                batt_y = min(node1_y, node2_y) + 0.5
                self.batteries[desc] = {
                    "x": node1_x,
                    "y": batt_y
                }
        self.add_thermal_nodes()
        self.add_thermal_edges()

    def add_thermal_nodes(self):
        self.thermal_graph.add_nodes_from(self.batteries)
        if self.left_bc == "ambient":
            self.thermal_graph.add_node("T_AMB_L")
        if self.right_bc == "ambient":
            self.thermal_graph.add_node("T_AMB_R")
        if self.top_bc == "ambient":
            self.thermal_graph.add_node("T_AMB_T")
        if self.bottom_bc == "ambient":
            self.thermal_graph.add_node("T_AMB_B")
        
    def add_thermal_edges(self):
        for desc in self.batteries:
            batt = self.batteries[desc]
            batt_x = batt["x"]
            batt_y = batt["y"]
            x_diffs = []
            y_diffs = []
            for other_desc in self.batteries:
                if other_desc == desc:
                    # its the same battery
                    continue
                else:
                    other_x = self.batteries[other_desc]["x"]
                    other_y = self.batteries[other_desc]["y"]
                    y_diff = other_y - batt_y
                    x_diff = other_x - batt_x
                    x_diffs.append(x_diff)
                    y_diffs.append(y_diff)
                    is_vert = (abs(y_diff) == 3) and other_x == batt_x
                    is_horz = (abs(x_diff) == 1) and other_y == batt_y
                    #Add an edge if the two batteries are next to each other
                    if is_vert or is_horz:
                        self.thermal_graph.add_edge(desc, other_desc)
            #Left Cell. 
            if all([x_diff <= 0.1 for x_diff in x_diffs]):
                if self.left_bc == "ambient":
                    self.thermal_graph.add_edge(desc, "T_AMB_L")
                elif self.left_bc == "symmetry":
                    self.thermal_graph.add_edge(desc, desc)
                else:
                    raise NotImplementedError("BC's must be ambient or symmetry")
            #Right Cell
            if all([x_diff >= 0 for x_diff in x_diffs]):
                if self.right_bc == "ambient":
                    self.thermal_graph.add_edge(desc, "T_AMB_R")
                elif self.top_bc == "symmetry":
                    self.thermal_graph.add_edge(desc, desc)
                else:
                    raise NotImplementedError("BC's must be ambient or symmetry")
            #Top Cell
            if all([y_diff <= 0 for y_diff in y_diffs]):
                if self.top_bc == "ambient":
                    self.thermal_graph.add_edge(desc, "T_AMB_T")
                elif self.top_bc == "symmetry":
                    self.thermal_graph.add_edge(desc, desc)
                else:
                    raise NotImplementedError("BC's must be ambient or symmetry")
            #Bottom Cell
            if all([y_diff >= 0 for y_diff in y_diffs]):
                if self.bottom_bc == "ambient":
                    self.thermal_graph.add_edge(desc, "T_AMB_B")
                elif self.bottom_bc == "symmetry":
                    self.thermal_graph.add_edge(desc, desc)
                else:
                    raise NotImplementedError("BC's must be ambient or symmetry")
    
    def build_thermal_equations_with_graph(self, pack):
        for node in pack.thermals.thermal_graph.nodes:
            if node[0] == "V":
                #node is a battery
                expr = 0
                for neighbor in pack.thermals.thermal_graph.neighbors(node):
                    if neighbor[0:5] == "T_AMB":
                        expr += pack.pack_ambient
                    elif neighbor[0] == "V":
                        expr += pack.batteries[neighbor]["temperature"]
                    else:
                        raise NotImplementedError("only batteries and ambient temperature can be calculated right now.")
                num_neighbors = len(pack.thermals.thermal_graph[node])
                expr = expr/num_neighbors
                pack.ambient_temperature.set_psuedo(pack.batteries[node]["cell"], expr)
            elif node[0:5] == "T_AMB":
                continue
            else:
                raise NotImplementedError("only batteries and ambient temperature can be calculated right now.")


class RibbonCoolingGraph(object):
    def __init__(
        self,
        circuit_graph,
        mdot=None,
        cp=None,
        T_i = 293,
    ):
        xs = []
        ys = []
        self.thermal_graph = nx.Graph()
        self.batteries = OrderedDict()
        for edge in circuit_graph.edges:
            row = circuit_graph.edges[edge]
            desc = row["desc"]
            # I'd like a better way to do this.
            if desc[0] == "V":
                node1_x = row["node1_x"]
                node2_x = row["node2_x"]
                node1_y = row["node1_y"]
                node2_y = row["node2_y"]
                #All batteries are vertical so only need to check node1 for x.
                #However we want all possible ys (the first one will be the inlet (constant T))
                batt_y = min(node1_y, node2_y) + 0.5
                if node1_x not in xs:
                    xs.append(node1_x)
                if batt_y not in ys:
                    ys.append(batt_y)
                if node1_x != node2_x:
                    raise AssertionError("x's must be the same")
                if abs(node1_y - node2_y) != 1:
                    raise AssertionError("batteries can only take up one y")
                self.batteries[desc] = {
                    "x": node1_x,
                    "y": batt_y
                }
        #Number of pipes is the number of potential x's minus 1
        self.num_pipes = len(xs) - 1
        self.nodes_per_pipe = len(ys)
        xs.sort()
        ys.sort()
        #x = -1 is the 
        xs = xs
        self.xs = xs
        self.ys = ys  

        self.T_i = T_i
        self.mdot = mdot
        self.cp = cp

        self.add_thermal_nodes()
        self.add_thermal_edges()

    def add_thermal_nodes(self):
        #start by adding batteries
        for batt in self.batteries:
            batt_loc = (self.batteries[batt]["x"], self.batteries[batt]["y"])
            self.thermal_graph.add_node(batt, loc=batt_loc, type="battery")
        self.thermal_graph.add_nodes_from(self.batteries)
        # Now add pipes 
        # (p is pipe number, x is x location of pipe)
        min_y = min(self.ys)
        for p,x in enumerate(self.xs[0:-1]):
            # (n is node number, y is y location of pipe)
            for n, y in enumerate(self.ys):
                node_name = "P_" + str(p) + "_" + str(n)
                node_loc = (x + 0.5, y)
                self.thermal_graph.add_node(node_name, loc=node_loc, p=p, n=n, type="pipe")
            #Add inlet
            node_name = "INLET_" + str(p)
            loc = (x + 0.5, min_y - 3)
            self.thermal_graph.add_node(node_name, p=p, loc=loc, type="inlet")
        
        
    def add_thermal_edges(self):
        for node_name in self.thermal_graph.nodes:
            node = self.thermal_graph.nodes[node_name]
            node_x,node_y = node["loc"]
            for other_name in self.thermal_graph.nodes:
                other_node = self.thermal_graph.nodes[other_name]
                #no self edges for pipes
                if other_name != node_name:
                    other_x,other_y = other_node["loc"]
                    y_diff = other_y - node_y
                    x_diff = other_x - node_x
                    #if node is a battery and other node is a pipe
                    if node["type"] == "battery" and other_node["type"] == "pipe":
                        is_horz = (abs(x_diff) == 0.5) and other_y == node_y
                        if is_horz:
                            self.thermal_graph.add_edge(node_name, other_name)
                    #if both are batteries
                    #elif node["type"] == "battery" and other_node["type"] == "battery":
                    #    #connect if they are vertically connected
                    #    is_vert = abs(y_diff) == 3 and other_x == node_x
                    #    if is_vert:
                    #        self.thermal_graph.add_edge(node_name, other_name)
                    ##if both are pipes
                    elif node["type"] == "pipe" and other_node["type"] == "pipe":
                        node_p = node["p"]
                        other_p = other_node["p"]
                        node_n = node["n"]
                        other_node_n = other_node["n"]
                        same_pipe = node_p == other_p
                        prox = abs(node_n - other_node_n) == 1
                        #connect if they are the same pipe and next to each other
                        if same_pipe and prox:
                            self.thermal_graph.add_edge(node_name, other_name)
                    elif node["type"] == "inlet" and other_node["type"] == "pipe":
                        node_p = node["p"]
                        other_p = other_node["p"]
                        is_node_0 = other_node["n"] == 0
                        same_pipe = node_p == other_p
                        if same_pipe and is_node_0:
                            self.thermal_graph.add_edge(node_name, other_name)
    
    def build_thermal_equations_with_graph(self, pack):
        #begin by making sure the proper inputs are in place.
        if self.mdot is None:
            self.mdot = pybamm.InputParameter("mdot")
            if "mdot" not in pack._input_parameter_order:
                raise AssertionError("please supply the pack with a mass flow rate")
        if (self.cp is None):
            self.cp = pybamm.InputParameter("cp")
            if  "cp" not in pack._input_parameter_order:
                raise AssertionError("please supply the pack with a cp for the cooling fluid")
        if (self.T_i is None):
            self.T_i = pybamm.InputParameter("T_i")
            if  "T_i" not in pack._input_parameter_order:
                raise AssertionError("please supply the pack with a T_i for the cooling fluid")

        #Now build the ambient temperatures of the pipe nodes
        for p in range(self.num_pipes):
            inlet_temp = self.T_i
            for n in range(self.nodes_per_pipe):
                name = "P_" + str(p) + "_" + str(n)
                if n == 0:
                    T_in = inlet_temp
                else:
                    previous_node_name = "P_" + str(p) + "_" + str(n-1)
                    T_in = pack.thermals.thermal_graph.nodes[previous_node_name]["outlet temperature"]
                T_s = 0
                num_batts = 0
                for neighbor in pack.thermals.thermal_graph.neighbors(name):
                    neighbor_node = pack.thermals.thermal_graph.nodes[neighbor]
                    if neighbor_node["type"] == "battery":
                        T_s += pack.batteries[neighbor]["temperature"]
                        num_batts += 1
                    if num_batts == 0:
                        raise AssertionError("nodes must have a battery connected")
                T_s = T_s/num_batts
                h = pack._parameter_values["Total heat transfer coefficient [W.m-2.K-1]"]
                A = pack._parameter_values["Cell cooling surface area [m2]"]
                #Using the mean temperature for heat transfer:
                T_out = ((2*A*h*T_s) - (A*h*T_in) + (2*self.cp*self.mdot*T_in))/((A*h) + (2*self.cp*self.mdot))
                pack.thermals.thermal_graph.nodes[name]["outlet temperature"] = T_out
                pack.thermals.thermal_graph.nodes[name]["inlet temperature"] = T_in
        
        #Now go through the batteries and set psuedo ambient temperatures
        for batt in pack.batteries:
            #Find neighbors (there be 2)
            T_amb = 0
            num_neighbors = 0
            for neighbor in pack.thermals.thermal_graph.neighbors(batt):
                T_amb += ((pack.thermals.thermal_graph.nodes[neighbor]["inlet temperature"] + pack.thermals.thermal_graph.nodes[neighbor]["outlet temperature"])/2)
                num_neighbors += 1

            ambient_temperature = T_amb/num_neighbors
            pack.ambient_temperature.set_psuedo(pack.batteries[batt]["cell"], ambient_temperature)


class BandolierCoolingGraph(object):
    def __init__(
        self,
        circuit_graph,
        mdot=None,
        cp=None,
        T_i = 293,
        transient = False,
        rho = None,
        A = None,
        deltax = None
    ):
        self.transient = transient
        self.rho = rho
        self.A = A
        self.deltax = deltax
        xs = []
        ys = []
        self.thermal_graph = nx.Graph()
        self.batteries = OrderedDict()
        for edge in circuit_graph.edges:
            row = circuit_graph.edges[edge]
            desc = row["desc"]
            # I'd like a better way to do this.
            if desc[0] == "V":
                node1_x = row["node1_x"]
                node2_x = row["node2_x"]
                node1_y = row["node1_y"]
                node2_y = row["node2_y"]
                #All batteries are vertical so only need to check node1 for x.
                #However we want all possible ys (the first one will be the inlet (constant T))
                batt_y = min(node1_y, node2_y) + 0.5
                if node1_x not in xs:
                    xs.append(node1_x)
                if batt_y not in ys:
                    ys.append(batt_y)
                if node1_x != node2_x:
                    raise AssertionError("x's must be the same")
                if abs(node1_y - node2_y) != 1:
                    raise AssertionError("batteries can only take up one y")
                self.batteries[desc] = {
                    "x": node1_x,
                    "y": batt_y
                }
        #Number of pipes is the number of potential x's minus 1
        self.num_pipes = 1
        xs.sort()
        ys.sort()
        #x = -1 is the 
        xs = xs
        self.xs = xs
        self.ys = ys  

        self.T_i = T_i
        self.mdot = mdot
        self.cp = cp

        self.add_thermal_nodes()
        self.add_thermal_edges()

    def add_thermal_nodes(self):
        #start by adding batteries
        for batt in self.batteries:
            batt_loc = (self.batteries[batt]["x"], self.batteries[batt]["y"])
            self.thermal_graph.add_node(batt, loc=batt_loc, type="battery")
        self.thermal_graph.add_nodes_from(self.batteries)
        
        # Now add pipes.
        # (p is pipe number, x is x location of pipe)
        # Direction will be used to swap the direction of the bandolero.
        # True = top to bottom, False = Bottom to Top
        direction = False
        min_y = min(self.ys)
        min_x = min(self.xs)
        n = 0
        for x in self.xs[0::2]:
            # (n is node number, y is y location of pipe)
            ys = copy.deepcopy(self.ys)
            ys.sort(reverse=direction)
            for y in ys:
                node_name = "P_" + str(n)
                node_loc = (x + 0.5, y)
                self.thermal_graph.add_node(node_name, loc=node_loc, n=n, type="pipe")
                n += 1
            #change directions at the end of each column
            direction = not direction
        #Add inlet
        node_name = "INLET"
        loc = (min_x + 0.5, min_y - 3)
        self.thermal_graph.add_node(node_name, loc=loc, type="inlet")
        self.nodes_per_pipe = n   
        
    def add_thermal_edges(self):
        #could be faster for sure, but not worrying about it for now (it's not even close to the limiting factor.)
        for node_name in self.thermal_graph.nodes:
            node = self.thermal_graph.nodes[node_name]
            node_x,node_y = node["loc"]
            for other_name in self.thermal_graph.nodes:
                other_node = self.thermal_graph.nodes[other_name]
                #no self edges for pipes
                if other_name != node_name:
                    other_x,other_y = other_node["loc"]
                    y_diff = other_y - node_y
                    x_diff = other_x - node_x
                    #if node is a battery and other node is a pipe
                    if node["type"] == "battery" and other_node["type"] == "pipe":
                        is_horz = (abs(x_diff) == 0.5) and (y_diff == 0)
                        if is_horz:
                            self.thermal_graph.add_edge(node_name, other_name)
                    elif node["type"] == "pipe" and other_node["type"] == "pipe":
                        node_n = node["n"]
                        other_node_n = other_node["n"]
                        prox = (node_n - other_node_n) == 1
                        #connect if they are next to each other
                        if prox:
                            self.thermal_graph.add_edge(node_name, other_name)
                    elif node["type"] == "inlet" and other_node["type"] == "pipe":
                        is_node_0 = other_node["n"] == 0
                        if is_node_0:
                            self.thermal_graph.add_edge(node_name, other_name)
    
    def build_thermal_equations_with_graph(self, pack):
        #begin by making sure the proper inputs are in place.
        if self.mdot is None:
            self.mdot = pybamm.InputParameter("mdot")
            if "mdot" not in pack._input_parameter_order:
                raise AssertionError("please supply the pack with a mass flow rate")
        if (self.cp is None):
            self.cp = pybamm.InputParameter("cp")
            if  "cp" not in pack._input_parameter_order:
                raise AssertionError("please supply the pack with a cp for the cooling fluid")
        if (self.T_i is None):
            self.T_i = pybamm.InputParameter("T_i")
            if  "T_i" not in pack._input_parameter_order:
                raise AssertionError("please supply the pack with a T_i for the cooling fluid")
        
        if self.transient:
            eqs = self.build_thermal_equations_with_graph_transient(pack)
        else:
            eqs = self.build_thermal_equations_with_graph_ss(pack)
        return eqs
    
    def build_thermal_equations_with_graph_transient(self, pack):
        if (self.rho is None):
            self.rho = pybamm.InputParameter("rho_cooling")
            if  "rho_cooling" not in pack._input_parameter_order:
                raise AssertionError("please supply the pack with a rho_cooling for the cooling fluid")

        if (self.A is None):
            self.A = pybamm.InputParameter("A_cooling")
            if  "A_cooling" not in pack._input_parameter_order:
                raise AssertionError("please supply the pack with an A for the cooling fluid")        

        if (self.deltax is None):
            self.deltax = pybamm.InputParameter("deltax")
            if  "deltax" not in pack._input_parameter_order:
                raise AssertionError("please supply the pack with a deltax for the cooling fluid")      
        
        eqs = []
        for p in range(self.num_pipes):
            inlet_temp = self.T_i
            for n in range(self.nodes_per_pipe):
                name = "P_" + str(n)
                temperature = pybamm.StateVector(slice(pack.offset, pack.offset + 1), name=name)
                pack.thermals.thermal_graph.nodes[name]["temperature"] = temperature
                if n == 0:
                    T_in = inlet_temp
                else:
                    previous_node_name = "P_" + str(n-1)
                    T_in = pack.thermals.thermal_graph.nodes[previous_node_name]["temperature"]
                Q_in = T_in * self.mdot * self.cp
                Q_out = temperature * self.mdot * self.cp
                for neighbor in pack.thermals.thermal_graph.neighbors(name):
                    neighbor_node = pack.thermals.thermal_graph.nodes[neighbor]
                    if neighbor_node["type"] == "battery":
                        h = pack._parameter_values["Total heat transfer coefficient [W.m-2.K-1]"]
                        A = pack._parameter_values["Cell cooling surface area [m2]"]
                        Q_in += (h*A*(pack.batteries[neighbor]["temperature"] - temperature))
                rhs = (Q_in - Q_out) / (self.cp * self.rho * self.deltax * self.A)
                eqs.append(rhs)
                pack.offset += 1
        
                
        for batt in pack.batteries:
            #Find neighbors (there should be 1)
            num_neighbors = 0
            T_amb = 0
            for neighbor in pack.thermals.thermal_graph.neighbors(batt):
                T_amb += pack.thermals.thermal_graph.nodes[neighbor]["temperature"]
                num_neighbors += 1
            if num_neighbors > 1:
                raise AssertionError("uh oh")
            ambient_temperature = T_amb/num_neighbors
            pack.ambient_temperature.set_psuedo(pack.batteries[batt]["cell"], ambient_temperature)
            
        return eqs                          
                            
    def build_thermal_equations_with_graph_ss(self, pack):
        #Now build the ambient temperatures of the pipe nodes
        for p in range(self.num_pipes):
            inlet_temp = self.T_i
            for n in range(self.nodes_per_pipe):
                name = "P_" + str(n)
                if n == 0:
                    T_in = inlet_temp
                else:
                    previous_node_name = "P_" + str(n-1)
                    T_in = pack.thermals.thermal_graph.nodes[previous_node_name]["outlet temperature"]
                T_s = 0
                num_batts = 0
                for neighbor in pack.thermals.thermal_graph.neighbors(name):
                    neighbor_node = pack.thermals.thermal_graph.nodes[neighbor]
                    if neighbor_node["type"] == "battery":
                        T_s += pack.batteries[neighbor]["temperature"]
                        num_batts += 1
                    if num_batts == 0:
                        raise AssertionError("nodes must have a battery connected")
                T_s = T_s/num_batts
                h = pack._parameter_values["Total heat transfer coefficient [W.m-2.K-1]"]
                A = pack._parameter_values["Cell cooling surface area [m2]"]
                #Using the mean temperature for heat transfer:
                T_out = ((2*A*h*T_s) - (A*h*T_in) + (2*self.cp*self.mdot*T_in))/((A*h) + (2*self.cp*self.mdot))
                pack.thermals.thermal_graph.nodes[name]["outlet temperature"] = T_out
                pack.thermals.thermal_graph.nodes[name]["inlet temperature"] = T_in
        
        #Now go through the batteries and set psuedo ambient temperatures
        for batt in pack.batteries:
            print(batt)
            #Find neighbors (there be 2)
            T_amb = 0
            num_neighbors = 0
            for neighbor in pack.thermals.thermal_graph.neighbors(batt):
                T_amb += ((pack.thermals.thermal_graph.nodes[neighbor]["inlet temperature"] + pack.thermals.thermal_graph.nodes[neighbor]["outlet temperature"])/2)
                num_neighbors += 1

            ambient_temperature = T_amb/num_neighbors
            pack.ambient_temperature.set_psuedo(pack.batteries[batt]["cell"], ambient_temperature)

