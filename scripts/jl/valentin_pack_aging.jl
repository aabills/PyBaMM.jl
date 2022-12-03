using PyBaMM
using ProgressMeter

pybamm = PyBaMM.pybamm
pack = PyBaMM.pack
pybamm2julia = PyBaMM.pybamm2julia
setup_circuit = PyBaMM.setup_circuit
sys = PyBaMM.sys

sys.path.append(joinpath(@__DIR__,"/Users/abills/Software/PyBaMM.jl/scripts/py/"))


# operating conditions
charge_rate = 1  # 0.05 to 5 C, linear
discharge_rate = 1  # 0.05 to 5 C, linear
V_max = 4.2  # 3.9 to 4.2 V, linear
V_min = 3.0  # 2.8 to 3.5 V, linear
T = 25  # -5 to 45 degC, linear

# degradation variables
SEI_rate_constant = 1e-15  # 1e-17 to 1e-14, logarithmic
pos_LAM_term = 1e-6  # 1e-8 to 1e-5, logarithmic
neg_LAM_term = 1e-6  # 1e-8 to 1e-5, logarithmic
EC_diffusivity = 1e-18  # 1e-20 to 1e-16, logarithmic

parameter_utils = pyimport("parameter_utils")

get_parameter_values = parameter_utils.get_parameter_values

# NMC532 parameter values with extra parameters for mechanical models
parameter_values = get_parameter_values()
parameter_values.update(
    pydict(Dict(
        "Ambient temperature [K]" => 273.15 + T,
        "SEI kinetic rate constant [m.s-1]" => SEI_rate_constant,
        "EC diffusivity [m2.s-1]" => EC_diffusivity,
    
)))
parameter_values.update(
    pydict(Dict(
        "Positive electrode LAM constant proportional term [s-1]" => pos_LAM_term,
        "Negative electrode LAM constant proportional term [s-1]" => neg_LAM_term,
    )),
    check_already_exists=false,
)


pybamm.set_logging_level("NOTICE")  # comment out to remove logging messages

capacity = pyconvert(Float64, parameter_values["Nominal cell capacity [A.h]"])

charge_rate = charge_rate * capacity
discharge_rate = discharge_rate * capacity
cutoff = capacity/50

Np = 3
Ns = 3
curr = 12
p = nothing 
t = 0.0
functional = true
voltage_functional = true

V_min = V_min*Ns
V_max = V_max*Ns
charge_rate = charge_rate*Np
discharge_rate = discharge_rate*Np



experiment = pybamm.Experiment(
    repeat([
            "Discharge at $charge_rate A until $V_min V",
            "Charge at $discharge_rate A until $V_max V",
            "Hold at $V_max V until $cutoff A",
    ],3)
)


# Load model
spm = pybamm.lithium_ion.DFN(
    pydict(Dict(
        "SEI" => "ec reaction limited",
        "loss of active material" => "stress-driven",
        "thermal" => "lumped"
    ))
)







netlist = setup_circuit.setup_circuit(Np, Ns, I=curr)   
pybamm_pack = pack.Pack(
    spm,
    netlist, 
    functional=functional, 
    thermal=true, 
    parameter_values=parameter_values, 
    initial_soc = nothing, 
    operating_mode=experiment,
    voltage_functional = voltage_functional
)

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

voltageconverter = pybamm2julia.JuliaConverter(cache_type = "dual", inplace=true)
voltageconverter.convert_tree_to_intermediate(pybamm_pack.voltage_func)
voltage_str = voltageconverter.build_julia_code()
voltage_str = pyconvert(String, voltage_str)
voltage_func = eval(Meta.parse(voltage_str))

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
