using PyBaMM
using ProgressMeter
using Statistics

pybamm = PyBaMM.pybamm
pack = PyBaMM.pack
pybamm2julia = PyBaMM.pybamm2julia
setup_circuit = PyBaMM.setup_circuit
setup_thermal_graph = PyBaMM.setup_thermal_graph

Np = 5
Ns = 5
curr = 6.80616
t = 0.0
functional = true
voltage_functional = true

options = pydict(Dict("thermal" => "lumped"))
model = pybamm.lithium_ion.SPMe(name="DFN", options=options)
parameter_values = model.default_parameter_values

netlist = setup_circuit.setup_circuit(Np, Ns, I=curr)  
circuit_graph = setup_circuit.process_netlist_from_liionpack(netlist) 

#Cooling System Parameters
ṁ = 1.0
height = 0.1
width = 0.01
P = (2*height + 2*width)
A = height*width
Dₕ = 4*height*width/(A)
Tᵢ = 298.0
COP = 2
Δx = 0.04 

#LIQUID GLYCOL
ρ = 1115.
cₚ = 0.895
μ = 1.61e-2
κₜ = 0.254
Nu = 5.6
h = Nu*κₜ/Dₕ

#Peclet Number
α = κₜ/(ρ*cₚ)
Pe = Δx*(ṁ/(ρ*A))/α



input_parameter_order = ["T_i","mdot","cp", "rho_cooling", "A_cooling", "deltax"]
p = [Tᵢ, ṁ , cₚ, ρ, A, Δx]


thermal_pipe = setup_thermal_graph.BandolierCoolingGraph(circuit_graph, mdot=nothing, cp=nothing, T_i=nothing, transient=true, h=h, A_cooling=0.0012250986127865292)
thermal_pipe_graph = thermal_pipe.thermal_graph

Re = ṁ/(μ * (P))

if Re >= 2000
    error("turbulent flow not supported")
else
    fd = 84/Re
end


pybamm_pack = pack.Pack(model, circuit_graph, functional=functional, thermals=thermal_pipe, voltage_functional=voltage_functional, input_parameter_order=input_parameter_order)

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

println("building jacobian sparsity...")
jac_sparsity = float(Symbolics.jacobian_sparsity((du,u)->jl_func(du,u,p,t),dy,jl_vec))
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

myconverter = pybamm2julia.JuliaConverter(cache_type = "dual", override_psuedo=true, input_parameter_order=input_parameter_order)
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
prob = ODEProblem(func, jl_vec, (0.0, 3600/timescale), p)
println("problem created...")

function calc_efficiency(p)
    ṁ = p[2]
    local_prob = remake(prob, p=p)
    sol = solve(local_prob, QNDF(linsolve=KLUFactorization(), concrete_jac = true), save_everystep = true)
    I = Array(sol)[1, :]
    V = Array(sol)[11, :]
    t = sol.t

    Vₘ = ṁ/(ρ*A)
    dpdx = (fd*Vₘ*Vₘ*ρ)/(2*Dₕ)
    L = Δx * pyconvert(Any, thermal_pipe.nodes_per_pipe)
    P_pump = L*dpdx*ṁ
    P_pack = I.*V
    P_fridge = abs.(ṁ*cₚ.*(Array(sol)[end, :] .- Tᵢ)./COP)
    E_pack = cumtrapz(t, P_pack)[end]
    E_pump = P_pump*(t[end] - t[1])
    E_fridge = cumtrapz(t, P_fridge)[end]
    E = calculate_theoretical_energy(parameter_values, 1.0, 0.0)
    η = (E_pack - E_pump - E_fridge)/(E*10*3600)
    return η
end

function cumtrapz(X::T, Y::T) where {T <: AbstractVector}
    # Check matching vector length
    @assert length(X) == length(Y)
    # Initialize Output
    out = similar(X)
    out[1] = 0
    # Iterate over arrays
    for i in 2:length(X)
      out[i] = out[i-1] + 0.5*(X[i] - X[i-1])*(Y[i] + Y[i-1])
    end
    # Return output
    out
  end

function calculate_theoretical_energy(parameter_values, initial_soc, final_soc, points = 100)
    n_i, p_i = pybamm.lithium_ion.electrode_soh.get_initial_stoichiometries(initial_soc, parameter_values)
    n_f, p_f = pybamm.lithium_ion.electrode_soh.get_initial_stoichiometries(final_soc, parameter_values)
    n_range = collect(range(pyconvert(Float64,n_i), stop=pyconvert(Float64,n_f), length=points))
    p_range = collect(range(pyconvert(Float64,p_i), stop=pyconvert(Float64,p_f), length=points))
    V = zeros(points)
    for i in 1:points
        U⁺ = pyconvert(Float64,parameter_values["Positive electrode OCP [V]"](p_range[i]).value)
        U⁻ = pyconvert(Float64,parameter_values["Negative electrode OCP [V]"](n_range[i]).value) 
        V[i] = U⁺ - U⁻
    end

    #ONLY NEED 1 Q
    Q_max_pos = parameter_values["Maximum concentration in positive electrode [mol.m-3]"]
    #Q_max_neg = parameter_values["Maximum concentration in negative electrode [mol.m-3]"]
    W = parameter_values["Electrode width [m]"]
    H = parameter_values["Electrode height [m]"]
    T_pos = parameter_values["Positive electrode thickness [m]"]
    #T_neg = parameter_values["Negative electrode thickness [m]"]
    #εₛ⁻ = parameter_values["Negative electrode active material volume fraction"]
    εₛ⁺ = parameter_values["Positive electrode active material volume fraction"]
    vol_pos = W*H*T_pos*εₛ⁺
    #vol_neg = W*H*T_neg*εₛ⁻
    #Q_n = vol_neg*Q_max_neg*(n_f - n_i)
    Q_p = vol_pos*Q_max_pos*(p_f - p_i)
    dQ = pyconvert(Float64,Q_p/points)
    E = sum(V.*dQ) * 26.8
    return E
end

E = calculate_theoretical_energy(parameter_values, 1.0, 0.0)




mdotarr = collect(0.01:0.01:1)
max_T = similar(mdotarr)
efficiency = similar(mdotarr)
@showprogress for (i,m) in enumerate(mdotarr)
    p[2] = m
    η = calc_efficiency(p) 
    efficiency[i] = η
end



