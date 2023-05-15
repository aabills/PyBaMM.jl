using PyBaMM
using ProgressMeter
using Statistics

pybamm = PyBaMM.pybamm
pack = PyBaMM.pack
pybamm2julia = PyBaMM.pybamm2julia
setup_circuit = PyBaMM.setup_circuit
setup_thermal_graph = PyBaMM.setup_thermal_graph

Np = 2
Ns = 2
curr = 1.2
t = 0.0
functional = true
voltage_functional = true

options = pydict(Dict("thermal" => "lumped"))
model = pybamm.lithium_ion.SPMe(name="DFN", options=options)
parameter_values = model.default_parameter_values

netlist = setup_circuit.setup_circuit(Np, Ns, I=curr)  
circuit_graph = setup_circuit.process_netlist_from_liionpack(netlist) 

#Cooling System Parameters
h_cell = 0.065 #18650
a = 1.25
D_cell = 0.018
W_pack = Np*a*D_cell
H_pack = h_cell
L_pack = D_cell*a*Ns
A_inlet = W_pack*H_pack
COP = 1.0
ṁ = 1.0
Δx = a*D_cell

coolant = "Novec 7500" 

#Coolant Properties
ρ = PyBaMM.coolant_properties[coolant]["Density [kg.m3]"]
cₚ = PyBaMM.coolant_properties[coolant]["Specific heat capacity [J.kg.K]"]
μ = PyBaMM.coolant_properties[coolant]["Dynamic viscosity [Pa-s]"]
κₜ = PyBaMM.coolant_properties[coolant]["Thermal conductivity [W.m-K]"]


#Fluids numbers 
u_mean = ṁ/(A_inlet*ρ)
u_max = ((a - 1)/a)*u_mean
Dₕ = D_cell
Re = ρ*u_max*Dₕ/μ
Pr_f = cₚ*μ/κₜ

Nu = PyBaMM.nusselt_mixed(false, 1, Re, Pr_f, Pr_f)
h = Nu*κₜ/Dₕ

#Peclet Number
α = κₜ/(ρ*cₚ)
Pe = Δx*(ṁ/(ρ*A_inlet))/α


thermal_pipe = setup_thermal_graph.ForcedConvectionGraph(circuit_graph, mdot=ṁ, cp=cₚ, T_i=298., h=h, A_cooling=A_inlet, rho=ρ, deltax = Δx, A=pyconvert(Float64, model.default_parameter_values["Cell cooling surface area [m2]"]))
thermal_pipe_graph = thermal_pipe.thermal_graph

if Re >= 2000
    error("turbulent flow not supported")
else
    fd = 84/Re
end


pybamm_pack = pack.Pack(model, circuit_graph, functional=functional, thermals=thermal_pipe, voltage_functional=voltage_functional)

println("compiling pack...")
pybamm_pack.build_pack()
println("finished compiling pack...")

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



myconverter = pybamm2julia.JuliaConverter(cache_type = "symbolic", override_psuedo=true)
myconverter.convert_tree_to_intermediate(pybamm_pack.pack)
pack_str = myconverter.build_julia_code()

icconverter = pybamm2julia.JuliaConverter(override_psuedo = true)
icconverter.convert_tree_to_intermediate(pybamm_pack.ics)
ic_str = icconverter.build_julia_code()

u0 = eval(Meta.parse(pyconvert(String,ic_str)))
jl_vec = u0()

pack_str = pyconvert(String, pack_str)
jl_func = eval(Meta.parse(pack_str))

dy = similar(jl_vec)

println("building jacobian sparsity...")
jac_sparsity = float(Symbolics.jacobian_sparsity((du,u)->jl_func(du,u,nothing,t),dy,jl_vec))
println("done building jacobian sparsity...")


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

myconverter = pybamm2julia.JuliaConverter(cache_type = "dual", override_psuedo=true)
myconverter.convert_tree_to_intermediate(pybamm_pack.pack)
pack_str = myconverter.build_julia_code()

pack_voltage_index = Np + 1
pack_voltage = 1.0
jl_vec[1:Np] .=  curr./Np
jl_vec[pack_voltage_index] = 4 * Ns

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

println("building function")
func = ODEFunction(jl_func, mass_matrix=mass_matrix, jac_prototype=jac_sparsity)
prob = ODEProblem(func, jl_vec, (0.0, 3600/timescale), nothing)
println("problem created...")


sol = solve(prob, QNDF(linsolve=KLUFactorization(), concrete_jac = true), save_everystep = true)


Eu = PyBaMM.euler_inline(Re, a)
Δp = Eu*0.5*ρ*u_max*u_max*Ns
P_pump = Δp * ṁ
T_in = 298.0
T_out = sol[end][end]
P_fridge = ṁ*cₚ*(T_out - T_in)/COP