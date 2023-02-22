import networkx as nx
from collections import OrderedDict


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


class RibbonCoolingGraph(object):
    def __init__(
        self,
        circuit_graph
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
                    elif node["type"] == "battery" and other_node["type"] == "battery":
                        #connect if they are vertically connected
                        is_vert = abs(y_diff) == 3 and other_x == node_x
                        if is_vert:
                            self.thermal_graph.add_edge(node_name, other_name)
                    #if both are pipes
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
                        

