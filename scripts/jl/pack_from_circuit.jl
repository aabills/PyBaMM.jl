using PyBaMM

pybamm = PyBaMM.pybamm
pack = PyBaMM.pack
pybamm2julia = PyBaMM.pybamm2julia
setup_circuit = PyBaMM.setup_circuit
setup_thermal_graph = PyBaMM.setup_thermal_graph

Np = 20
Ns = 20
curr = 12
t = 0.0
functional = true
voltage_functional = true

options = pydict(Dict("thermal" => "lumped"))
model = pybamm.lithium_ion.DFN(name="DFN", options=options)

netlist = setup_circuit.setup_circuit(Np, Ns, I=curr)  
circuit_graph = setup_circuit.process_netlist_from_liionpack(netlist) 

thermal_pipe = setup_thermal_graph.BandolierCoolingGraph(circuit_graph, mdot=nothing, cp=nothing, T_i=nothing, transient=true)
thermal_pipe_graph = thermal_pipe.thermal_graph

#LIQUID GLYCOL
ρ = 1115.
cₚ = 0.895
μ = 1.61e-2


#Cooling System Parameters
ṁ = 1.0
height = 0.1
width = 0.01
Tᵢ = 298.0
COP = .5

#numerical Parameters (complete guess)
Δx = 0.04 

#Geometry
P = (2*height + 2*width)
A = height*width

input_parameter_order = ["T_i","mdot","cp", "rho_cooling", "A_cooling", "deltax"]
p = [Tᵢ, ṁ , cₚ, ρ, A, Δx]



Re = ṁ/(μ * (P))

if Re >= 2000
    error("turbulent flow not supported")
else
    fd = 84/Re
end

Dₕ = 4*height*width/(A)
Vₘ = ṁ/(ρ*A)
dpdx = (fd*Vₘ*Vₘ*ρ)/(2*Dₕ)
L = Δx * pyconvert(Any, thermal_pipe.nodes_per_pipe)
P_pump = L*dpdx*ṁ


pybamm_pack = pack.Pack(model, circuit_graph, functional=functional, thermals=thermal_pipe, voltage_functional=voltage_functional, input_parameter_order=input_parameter_order)
pybamm_pack.build_pack()


if voltage_functional
    voltageconverter = pybamm2julia.JuliaConverter(cache_type = "symbolic", inplace=true)
    voltageconverter.convert_tree_to_intermediate(pybamm_pack.voltage_func)
    voltage_str = voltageconverter.build_julia_code()
    voltage_str = pyconvert(String, voltage_str)
    voltage_func = eval(Meta.parse(voltage_str))
else
    voltage_str = ""
end


timescale = 1
cellconverter = pybamm2julia.JuliaConverter(cache_type = "symbolic", inplace=true)
cellconverter.convert_tree_to_intermediate(pybamm_pack.cell_model)
cell_str = cellconverter.build_julia_code()
cell_str = pyconvert(String, cell_str)
cell! = eval(Meta.parse(cell_str))



myconverter = pybamm2julia.JuliaConverter(cache_type = "symbolic", override_psuedo=true, input_parameter_order=input_parameter_order)
myconverter.convert_tree_to_intermediate(pybamm_pack.pack)
pack_str = myconverter.build_julia_code()

icconverter = pybamm2julia.JuliaConverter(override_psuedo = true, input_parameter_order=input_parameter_order)
icconverter.convert_tree_to_intermediate(pybamm_pack.ics)
ic_str = icconverter.build_julia_code()

u0 = eval(Meta.parse(pyconvert(String,ic_str)))
jl_vec = u0(p)

pack_str = pyconvert(String, pack_str)
jl_func = eval(Meta.parse(pack_str))

dy = similar(jl_vec)

jac_sparsity = float(Symbolics.jacobian_sparsity((du,u)->jl_func(du,u,p,t),dy,jl_vec))

if voltage_functional
    voltageconverter = pybamm2julia.JuliaConverter(cache_type = "dual", inplace=true)
    voltageconverter.convert_tree_to_intermediate(pybamm_pack.voltage_func)
    voltage_str = voltageconverter.build_julia_code()
    voltage_str = pyconvert(String, voltage_str)
    voltage_func = eval(Meta.parse(voltage_str))
else
    voltage_str = ""
end

cellconverter = pybamm2julia.JuliaConverter(cache_type = "dual", inplace=true)
cellconverter.convert_tree_to_intermediate(pybamm_pack.cell_model)
cell_str = cellconverter.build_julia_code()
cell_str = pyconvert(String, cell_str)
cell! = eval(Meta.parse(cell_str))

myconverter = pybamm2julia.JuliaConverter(cache_type = "dual", override_psuedo=true, input_parameter_order=input_parameter_order)
myconverter.convert_tree_to_intermediate(pybamm_pack.pack)
pack_str = myconverter.build_julia_code()

pack_voltage_index = Np + 1
pack_voltage = 1.0
jl_vec[1:Np] .=  curr
jl_vec[pack_voltage_index] =11

pack_str = pyconvert(String, pack_str)
jl_func = eval(Meta.parse(pack_str))

#build mass matrix.
pack_eqs = falses(pyconvert(Int,pybamm_pack.len_pack_eqs))

cell_rhs = trues(pyconvert(Int,pybamm_pack.len_cell_rhs))
cell_algebraic = falses(pyconvert(Int,pybamm_pack.len_cell_algebraic))
cells = repeat(vcat(cell_rhs,cell_algebraic),pyconvert(Int, pybamm_pack.num_cells))
thermals = trues(pyconvert(Int,pybamm_pack.len_thermal_eqs))
differential_vars = vcat(pack_eqs, cells, thermals)
mass_matrix = sparse(diagm(differential_vars))


func = ODEFunction(jl_func, mass_matrix=mass_matrix, jac_prototype=jac_sparsity)
prob = ODEProblem(func, jl_vec, (0.0, 100/timescale), p)

sol = solve(prob, QNDF(), save_everystep = true)

arr_sol = Array(sol)
arr_t = Array(sol.t)

I = Array(sol)[1, :]
V = Array(sol)[3, :]

P_pack = I.*V
P_fridge = ṁ*cₚ.*(Array(sol)[end, :] .- Tᵢ)./COP
η_pack = (P_pack .- P_pump .- P_fridge)./P_pack


@save "test.jld2" arr_sol arr_t I V η_pack
