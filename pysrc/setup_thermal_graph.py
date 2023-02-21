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