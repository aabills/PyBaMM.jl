plt = pyimport("matplotlib.pyplot")
nx = pyimport("networkx")
using Statistics

pos = pydict()
for node in pybamm_pack.circuit_graph.nodes
    pos.update(pydict(Dict(node => pybamm_pack.circuit_graph.nodes[node]["loc"])))
end
battery_edges = pylist()
for edge in pybamm_pack.circuit_graph.edges
    if pyconvert(Any,pybamm_pack.circuit_graph.edges[edge]["desc"])[1] == 'V'
        battery_edges.append(edge)
    end
end
resistor_edges = pylist()
for edge in pybamm_pack.circuit_graph.edges
    if pyconvert(Any,pybamm_pack.circuit_graph.edges[edge]["desc"])[1] == 'R'
        resistor_edges.append(edge)
    end
end

thermal_pos = pydict()


max_x = -Inf
min_x = Inf
max_y = -Inf
min_y = Inf
for node in pybamm_pack.thermals.thermal_graph.nodes()
    thermal_pos.update(pydict(Dict(node => pybamm_pack.thermals.thermal_graph.nodes[node]["loc"])))
end
x_range = max_x - min_x
y_range = max_y - min_y

left = max_x + 0.1*x_range
right = min_x - 0.1*x_range
top = max_y + 0.1*y_range
bottom = min_y - 0.1*y_range
mid_y = min_y + y_range/2
mid_x = min_x + x_range/2




G = pybamm_pack.circuit_graph
H = pybamm_pack.thermals.thermal_graph
nx.draw_networkx_edges(G, pos=pos, edgelist=battery_edges, width=50,edge_color = "tab:green", label="Cell")
nx.draw_networkx_edges(G, pos=pos, edgelist=resistor_edges, width=1,edge_color = [0 0 0],label="Resistor")
nx.draw_networkx_nodes(G, pos=pos,node_size=15,node_color = [0 0 0], label="Electrical Node")
nx.draw_networkx_nodes(H, pos=thermal_pos, node_color="tab:red",label="Thermal Node", node_size=25)
nx.draw_networkx_labels(H, pos=thermal_pos)
nx.draw_networkx_edges(H, pos=thermal_pos,width=5, edge_color="tab:red", alpha=0.4,label="Thermal Connection")
plt.legend(loc="center right", bbox_to_anchor=(0, 0.5))
plt.show()



