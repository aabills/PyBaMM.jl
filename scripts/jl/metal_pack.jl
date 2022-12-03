using PyBaMM
using JLD2
using ProgressMeter

pybamm = PyBaMM.pybamm
pack = PyBaMM.pack
pybamm2julia = PyBaMM.pybamm2julia
setup_circuit = PyBaMM.setup_circuit
sys = PyBaMM.sys

sys.path.append(joinpath(@__DIR__,"/Users/abills/ResearchProjects/li_metal/code"))

parameters = pyimport("parameters")
data_utils = pyimport("data_utils")

cell = "B1"
cap = parameters.capacities_mAh[cell] / 1e3
parameter_values = parameters.get_parameter_values(cap, "volume fraction")

distribution_params = Dict(
    "Cation transference number"=>Dict("mean"=>0.38,"stddev"=>0.01,"name"=>"inter_util")
)

options = pydict(Dict("working electrode"=>"positive"))
model = pybamm.lithium_ion.DFN(options=options)

#Quick and dirty eVTOL approximation:
# Assume at 3V, power is ~4C => 12 W for takeoff/landing
# 54W to/land => 3.5W cruise

#Power really doesn't like to converge so just approximating with current for now.
experiment = pybamm.Experiment([
        "Charge at $cap A until $(Ns*4.3) V",
        "Hold at $(Ns*4.3) V until 10 mA",
        "Rest for 30 sec",
        "Discharge at $(cap*Np*3)A for 75 sec",
        "Discharge at $(cap*Np)A for 800 sec",
        "Discharge at $(cap*Np*3)A for 105 sec",
    ]
)

Np = 3
Ns = 3
curr = 12
p = nothing 
t = 0.0
functional = true

netlist = setup_circuit.setup_circuit(Np, Ns, I=curr)   
pybamm_pack = pack.Pack(
    model,
    netlist, 
    functional=functional, 
    thermal=false, 
    parameter_values=parameter_values, 
    initial_soc = nothing, 
    operating_mode=experiment,
    distribution_params = distribution_params
)

pybamm_pack.build_pack()

timescale = pyconvert(Float64,pybamm_pack.timescale.evaluate())
cellconverter = pybamm2julia.JuliaConverter(cache_type = "symbolic", inplace=true)
cellconverter.convert_tree_to_intermediate(pybamm_pack.cell_model)
cell_str = cellconverter.build_julia_code()
cell_str = pyconvert(String, cell_str)
cell! = eval(Meta.parse(cell_str))


myconverter = pybamm2julia.JuliaConverter(cache_type = "symbolic")
myconverter.convert_tree_to_intermediate(pybamm_pack.pack)
pack_str = myconverter.build_julia_code()

forcing_function = pybamm_pack.forcing_functions[0]
forcing_converter = pybamm2julia.JuliaConverter(cache_type = "symbolic")
forcing_converter.convert_tree_to_intermediate(forcing_function)
forcing_str = forcing_converter.build_julia_code()
forcing_str = pyconvert(String,forcing_str)

forcing_function = eval(Meta.parse(forcing_str))
open("cvpack.jl","w") do io
    println(io, pack_str)
end


icconverter = pybamm2julia.JuliaConverter(override_psuedo = true)
icconverter.convert_tree_to_intermediate(pybamm_pack.ics)
ic_str = icconverter.build_julia_code()

u0 = eval(Meta.parse(pyconvert(String,ic_str)))
jl_vec = u0()

pack_str = pyconvert(String, pack_str)
jl_func = eval(Meta.parse(pack_str))

dy = similar(jl_vec)

jac_sparsity = float(Symbolics.jacobian_sparsity((du,u)->jl_func(du,u,p,t),dy,jl_vec))

cellconverter = pybamm2julia.JuliaConverter(cache_type = "dual", inplace=true)
cellconverter.convert_tree_to_intermediate(pybamm_pack.cell_model)
cell_str = cellconverter.build_julia_code()
cell_str = pyconvert(String, cell_str)
cell! = eval(Meta.parse(cell_str))

myconverter = pybamm2julia.JuliaConverter(cache_type = "dual")
myconverter.convert_tree_to_intermediate(pybamm_pack.pack)
pack_str = myconverter.build_julia_code()

forcing_function = pybamm_pack.forcing_functions[0]
forcing_converter = pybamm2julia.JuliaConverter(cache_type = "dual")
forcing_converter.convert_tree_to_intermediate(forcing_function)
forcing_str = forcing_converter.build_julia_code()
forcing_str = pyconvert(String,forcing_str)

forcing_function = eval(Meta.parse(forcing_str))

pack_voltage_index = Np + 1
pack_voltage = 3.5.*Ns
jl_vec[1:Np] .=  curr
jl_vec[pack_voltage_index] = pack_voltage

pack_str = pyconvert(String, pack_str)
jl_func = eval(Meta.parse(pack_str))

#build mass matrix.
pack_eqs = falses(pyconvert(Int,pybamm_pack.len_pack_eqs))

cell_rhs = trues(pyconvert(Int,pybamm_pack.len_cell_rhs))
cell_algebraic = falses(pyconvert(Int,pybamm_pack.len_cell_algebraic))
cells = repeat(vcat(cell_rhs,cell_algebraic),pyconvert(Int, pybamm_pack.num_cells))
differential_vars = vcat(pack_eqs,cells)
mass_matrix = sparse(diagm(differential_vars))

integrator = 0

func = ODEFunction(jl_func, mass_matrix=mass_matrix)
prob = ODEProblem(func, jl_vec, (0.0, 3600/timescale), nothing)
integrator = init(prob, QNDF(concrete_jac=true))



@showprogress "cycling..." for i in 1:length(experiment.operating_conditions)
    forcing_function = pybamm_pack.forcing_functions[i-1]
    termination_function = pybamm_pack.termination_functions[i-1]

    forcing_converter = pybamm2julia.JuliaConverter(cache_type = "dual")
    forcing_converter.convert_tree_to_intermediate(forcing_function)
    forcing_str = forcing_converter.build_julia_code()
    forcing_str = pyconvert(String,forcing_str)

    forcing_function = eval(Meta.parse(forcing_str))
    
    termination_converter = pybamm2julia.JuliaConverter(cache_type = "dual")
    termination_converter.convert_tree_to_intermediate(termination_function)
    termination_str = termination_converter.build_julia_code()
    termination_str = pyconvert(String, termination_str)

    termination_function = eval(Meta.parse(termination_str))

    done = false
    start_t = integrator.t
    while !done
        step!(integrator)
        done = any((Base.@invokelatest termination_function(integrator.u, (integrator.t - start_t)*pyconvert(Float64, pybamm_pack.built_model.timescale.evaluate()))).<0)
    end
end

saved_vars = get_pack_variables(pybamm_pack, integrator.sol, ["Current [A]","Terminal voltage [V]"])

arr_sol = Array(integrator.sol)
arr_t = Array(integrator.sol.t)

@save "metal_3x3_vtol_variation.jld2" saved_vars arr_sol arr_t
