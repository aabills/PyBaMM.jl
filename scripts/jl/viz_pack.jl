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
for edge in pybamm_pack.circuit_graph.edges
    if pyconvert(Any,pybamm_pack.circuit_graph.edges[edge]["desc"])[1] == 'V'
        node1_x = pyconvert(Float64, pos[edge[0]][0])
        node1_y = pyconvert(Float64, pos[edge[0]][1])
        node2_x = pyconvert(Float64, pos[edge[1]][0])
        node2_y = pyconvert(Float64, pos[edge[1]][1])
        if max(node1_x, node2_x) > max_x
            global max_x = max(node1_x, node2_x)
        end
        if max(node1_y, node2_y) > max_y
            global max_y = max(node1_y, node2_y)
        end
        if min(node1_x,node2_x) < min_x
            global min_x = max(node1_x, node2_x)
        end
        if min(node1_y, node2_y) < min_y
            global min_y = min(node1_y, node2_y)
        end
        battery = pybamm_pack.circuit_graph.edges[edge]["desc"]
        thermal_pos.update(pydict(Dict(battery => pylist([mean([node1_x,node2_x]),mean([node1_y,node2_y])]))))
    end
end
x_range = max_x - min_x
y_range = max_y - min_y

left = max_x + 0.1*x_range
right = min_x - 0.1*x_range
top = max_y + 0.1*y_range
bottom = min_y - 0.1*y_range
mid_y = min_y + y_range/2
mid_x = min_x + x_range/2

thermal_pos["T_AMB_L"] = pylist([left, mid_y])
thermal_pos["T_AMB_R"] = pylist([right, mid_y])
thermal_pos["T_AMB_T"] = pylist([mid_x, top])
thermal_pos["T_AMB_B"] = pylist([mid_x, bottom])

battery_nodelist = pylist([node for node in pybamm_pack.thermal_graph.nodes])
battery_nodelist.remove(pystr("T_AMB_L"))
battery_nodelist.remove(pystr("T_AMB_R"))
battery_nodelist.remove(pystr("T_AMB_B"))
battery_nodelist.remove(pystr("T_AMB_T"))

ambient_nodelist = pylist(["T_AMB_L","T_AMB_R","T_AMB_B","T_AMB_T"])




G = pybamm_pack.circuit_graph
H = pybamm_pack.thermal_graph
#nx.draw_networkx_edges(G, pos=pos, edgelist=battery_edges, width=10,edge_color = "t")
nx.draw_networkx_edges(G, pos=pos, edgelist=resistor_edges, width=1,edge_color = [0 0 0],label="Resistor")
nx.draw_networkx_nodes(G, pos=pos,node_size=15,node_color = [0 0 0])
nx.draw_networkx_nodes(H, pos=thermal_pos, node_color="tab:green",nodelist=battery_nodelist,label="Cell",node_shape="s")

nx.draw_networkx_nodes(H, pos=thermal_pos, node_color="tab:blue",nodelist=ambient_nodelist,label="Ambient")

nx.draw_networkx_edges(H, pos=thermal_pos,width=5, edge_color="tab:red", alpha=0.4,label="Thermal Connection")
plt.legend(loc="center right", bbox_to_anchor=(0, 0.5))


pipe_pos = pydict()
for node in thermal_pipe_graph.nodes()
    loc = thermal_pipe_graph.nodes[node]["loc"]
    pipe_pos.update(pydict(Dict(node => loc)))
end


nx.draw_networkx(thermal_pipe_graph, pos=pipe_pos,width=5, edge_color="tab:red", alpha=0.4)



