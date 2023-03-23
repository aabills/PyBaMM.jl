using PyBaMM
using Statistics
using DataFrames
using CSV
using JLD2
using Plots
plotly()

pybamm = PyBaMM.pybamm
pack = PyBaMM.pack
pybamm2julia = PyBaMM.pybamm2julia
setup_circuit = PyBaMM.setup_circuit

df = CSV.read("VAH01_20.csv",DataFrame)
df.times = df.times .- df.times[1]
ndf = filter(row->row.Ns==7,df)
rest_time = ndf.times[end] - ndf.times[1]

Np = 1
Ns = 1
curr = 1.0
t = 0.0
functional = true
options = Dict("thermal" => "lumped")
model = pybamm.lithium_ion.DFN(name="DFN", options=options)
netlist = setup_circuit.setup_circuit(Np, Ns, I=curr, Rb=0.00, Rc=0.0, Ri=0.0)
#=
experiment = pybamm.Experiment([
    "Discharge at 54 W for 75 sec",
    "Discharge at 16 W for 800 sec",
    "Discharge at 54 W for 105 sec",
    "Rest for $rest_time sec",
    "Charge at 3 A until 4.2 V",
    "Hold at 4.2 V until 100 mA",
    "Rest for 100 sec"
])
=#
experiment = pybamm.Experiment(["Discharge at 1 W for 5 min"])

#experiment = pybamm.Experiment([
#    "Rest for 75 sec",
#])

parameter_values = pybamm.ParameterValues("Chen2020")

#Geometry
#=
thickness = 2.3e-5
volume = 1.65e-5
parameter_values["Cell volume [m3]"] = volume
NP = 1.15
parameter_values["Negative particle radius [m]"] = 6e-6
parameter_values["Positive electrode thickness [m]"] = thickness
parameter_values["Negative electrode thickness [m]"] = thickness*NP
parameter_values["Electrode height [m]"] = .06
parameter_values["Electrode width [m]"] = volume/(.06*(parameter_values["Positive electrode thickness [m]"]+
parameter_values["Negative electrode thickness [m]"]+
parameter_values["Positive current collector thickness [m]"]+
parameter_values["Separator thickness [m]"]+
parameter_values["Negative current collector thickness [m]"]))


#Positive Electrode
parameter_values["Positive electrode diffusivity [m2.s-1]"] = pybamm.InputParameter("positive_diffusivity")
parameter_values["Positive electrode active material volume fraction"] = 0.665
parameter_values["Positive electrode conductivity [S.m-1]"] = pybamm.InputParameter("pos_elec_cond")
parameter_values["Positive particle radius [m]"] = 6e-6
parameter_values["Positive electrode exchange-current density [A.m-2]"] = pybamm.InputParameter("pos_exch")
#Positive electrode (calculated)
parameter_values["Positive electrode porosity"] = 1 - parameter_values["Positive electrode active material volume fraction"]

#Negative Electrode
parameter_values["Negative electrode diffusivity [m2.s-1]"] = pybamm.InputParameter("negative_diffusivity")
parameter_values["Negative electrode active material volume fraction"] = 0.75
parameter_values["Negative electrode conductivity [S.m-1]"] = pybamm.InputParameter("neg_elec_cond")
parameter_values["Negative particle radius [m]"] = 6e-6
parameter_values["Negative electrode exchange-current density [A.m-2]"] = pybamm.InputParameter("neg_exch")
#Negative electrode (calculated)
parameter_values["Negative electrode porosity"] = pybamm.InputParameter("negative_porosity")


#Electrolyte
parameter_values["EC initial concentration in electrolyte [mol.m-3]"] = 4541.0
parameter_values["Initial concentration in electrolyte [mol.m-3]"] = 1000.0
parameter_values["Electrolyte conductivity [S.m-1]"] = pybamm.InputParameter("conductivity")

parameter_values["Electrolyte diffusivity [m2.s-1]"] = 1e-9
parameter_values["Cation transference number"] = pybamm.InputParameter("transference")

#SEI
parameter_values["Initial inner SEI thickness [m]"] = pybamm.InputParameter("inner_sei")
parameter_values["Initial outer SEI thickness [m]"] = 2.5e-9

input_parameter_order = ["conductivity","transference","positive_diffusivity","negative_diffusivity","inner_sei","negative_porosity","pos_elec_cond","pos_exch","neg_elec_cond","neg_exch"]


p = [0.05,0.25,4e-14,1.4e-14,2.5e-10,0.25,0.18,5,218,3]
ub = [0.1, 1.0,1e-13,1e-13,1e-8,0.25,1.0,1,1000,10]
lb = [0.01, 0.0, 1e-15,1e-15,1e-9,0.0,0.1,0,100,1]
=#
function elec_cond()



parameter_values["Electrolyte conductivity [S.m-1]"] =  1000000.0

#Running stuff
pybamm_pack = pack.Pack(model, netlist, functional=functional, thermal=true, operating_mode = experiment, parameter_values = parameter_values)
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


icconverter = pybamm2julia.JuliaConverter(override_psuedo = true)
icconverter.convert_tree_to_intermediate(pybamm_pack.ics)
ic_str = icconverter.build_julia_code()
p = nothing

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

dt = 1/pyconvert(Float64, pybamm_pack.built_model.timescale.evaluate())

func = ODEFunction(jl_func, mass_matrix=mass_matrix, jac_prototype=jac_sparsity)
prob = ODEProblem(func, jl_vec, (0.0, 3600/timescale), p)
integrator = init(prob, QNDF(concrete_jac=true))


forcing_function_arr = Array{Any,1}(undef,length(experiment.operating_conditions))
termination_function_arr = Array{Any,1}(undef, length(experiment.operating_conditions))

for i in 1:length(experiment.operating_conditions)
    forcing_function = pybamm_pack.forcing_functions[i-1]
    termination_function = pybamm_pack.termination_functions[i-1]

    forcing_converter = pybamm2julia.JuliaConverter(cache_type = "dual")
    forcing_converter.convert_tree_to_intermediate(forcing_function)
    forcing_str = forcing_converter.build_julia_code()
    forcing_str = pyconvert(String,forcing_str)

    forcing_function = eval(Meta.parse(forcing_str))
    forcing_function_arr[i] = forcing_function
    
    termination_converter = pybamm2julia.JuliaConverter(cache_type = "dual")
    termination_converter.convert_tree_to_intermediate(termination_function)
    termination_str = termination_converter.build_julia_code()
    termination_str = pyconvert(String, termination_str)

    termination_function = eval(Meta.parse(termination_str))
    termination_function_arr[i] = termination_function
end

df.newt = (df.times .- df.times[1])./pyconvert(Float64, pybamm_pack.built_model.timescale.evaluate())

integrator = init(prob, QNDF(concrete_jac=true))
for i in 1:length(experiment.operating_conditions)
    global forcing_function = forcing_function_arr[i]
    global termination_function = termination_function_arr[i]

    done = false
    start_t = integrator.t
    while !done
        step!(integrator,dt, true)
        done = |(any((termination_function(integrator.u, (integrator.t - start_t)*pyconvert(Float64, pybamm_pack.built_model.timescale.evaluate()))).<0),integrator.t>df.newt[end])
        if integrator.sol.retcode != :Default
            break
        end
    end
    if integrator.sol.retcode != :Default
        break
    end
end
if integrator.sol.retcode != :Default
    println("uh oh")
end
sol = hcat(integrator.sol.(df.newt)...)
V = sol[2,:]
rmse_V = sqrt.(mean((V .- df.EcellV).^2))
println(rmse_V)


plot(integrator.sol.t,integrator.sol[2,:])
plot!(df.newt,df.EcellV)
