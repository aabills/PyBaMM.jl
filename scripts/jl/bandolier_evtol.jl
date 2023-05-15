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
λ = 100
curr = 40.0
t = 0.0
functional = true
voltage_functional = true

options = pydict(Dict("thermal" => "lumped"))


parameter_values = pybamm.ParameterValues("Marquis2019")

#make it an 18650
parameter_values["Electrode height [m]"] = 5.8e-2
parameter_values["Electrode width [m]"] = 61.5e-2*2
parameter_values["Ambient temperature [K]"] = 305.
parameter_values["Initial temperature [K]"] = 305.


experiment = pybamm.Experiment(["Discharge at $(28*Np*Ns) W for 75 sec", "Discharge at $(8*Np*Ns) W for 800 sec", "Discharge at $(28*Np*Ns) W for 105 sec", "Rest for 1000 sec"])


model = pybamm.lithium_ion.SPMe(name="DFN", options=options)

netlist = setup_circuit.setup_circuit(Np, Ns, I=curr)  
circuit_graph = setup_circuit.process_netlist_from_liionpack(netlist) 

#Cooling System Parameters
ṁ = 0.1
height = 0.1
width = 0.01
P = (2*height + 2*width)
A = height*width
Dₕ = 4*A/(P)
Tᵢ = 305.0
COP = 2
Δx = 0.04 

#LIQUID GLYCOL
coolant = "Liquid glycol" 

#Coolant Properties
ρ = PyBaMM.coolant_properties[coolant]["Density [kg.m3]"]
cₚ = PyBaMM.coolant_properties[coolant]["Specific heat capacity [J.kg.K]"]
μ = PyBaMM.coolant_properties[coolant]["Dynamic viscosity [Pa-s]"]
κₜ = PyBaMM.coolant_properties[coolant]["Thermal conductivity [W.m-K]"]
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


pybamm_pack = pack.Pack(
    model, 
    circuit_graph, 
    functional=functional, 
    thermals=thermal_pipe, 
    voltage_functional=voltage_functional, 
    input_parameter_order=input_parameter_order,
    operating_mode = experiment,
    parameter_values=parameter_values,
    initial_soc = 1.0
)

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

forcing_function = pybamm_pack.forcing_functions[0]
forcing_converter = pybamm2julia.JuliaConverter(cache_type = "symbolic")
forcing_converter.convert_tree_to_intermediate(forcing_function)
forcing_str = forcing_converter.build_julia_code()
forcing_str = pyconvert(String,forcing_str)

forcing_function = eval(Meta.parse(forcing_str))

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

forcing_function = pybamm_pack.forcing_functions[0]
forcing_converter = pybamm2julia.JuliaConverter(cache_type = "dual")
forcing_converter.convert_tree_to_intermediate(forcing_function)
forcing_str = forcing_converter.build_julia_code()
forcing_str = pyconvert(String,forcing_str)

forcing_function = eval(Meta.parse(forcing_str))

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
prob = ODEProblem(func, jl_vec, (0.0, 1080.0), p)
println("problem created...")

    #u0 = T.(jl_vec)
    prob = ODEProblem(func, jl_vec, (0.0, 1080.0), p)
    println("initializing")
    integrator = init(prob, QNDF(), save_everystep = true, dtmax = 1.0)
    println("done initializing, cycling...")
    #force_out = zeros(1)

    for i in 1:length(experiment.operating_conditions)
        println(i)
        forcing_function = pybamm_pack.forcing_functions[i-1]
        termination_function = pybamm_pack.termination_functions[i-1]

        forcing_converter = pybamm2julia.JuliaConverter(cache_type = "dual")
        forcing_converter.convert_tree_to_intermediate(forcing_function)
        forcing_str = forcing_converter.build_julia_code()
        forcing_str = pyconvert(String,forcing_str)

        forcing_function = eval(Meta.parse(forcing_str))

        #Base.@invokelatest forcing_function(force_out, integrator.u, integrator.t)
    
        termination_converter = pybamm2julia.JuliaConverter(cache_type = "dual")
        termination_converter.convert_tree_to_intermediate(termination_function)
        termination_str = termination_converter.build_julia_code()
        termination_str = pyconvert(String, termination_str)

        termination_function = eval(Meta.parse(termination_str))

        done = false
        start_t = integrator.t
        while !done
            Base.@invokelatest step!(integrator)
            done = any((Base.@invokelatest termination_function(integrator.u, (integrator.t - start_t))).<0)
        end
        savevalues!(integrator)
    end