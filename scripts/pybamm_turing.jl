using PyBaMM
using ProgressMeter
using Statistics
using Turing

#pybamm all lower case is the python module pybamm
pybamm = PyBaMM.pybamm
#pybamm2julia is the python module we will use to convert from pybamm to julia (duh)
pybamm2julia = PyBaMM.pybamm2julia


model = pybamm.lithium_ion.SPMe(name="DFN")
parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.update(
    PyDict(Dict(
        "Positive electrode active material volume fraction" => pybamm.InputParameter("eps_p"),
        "Negative electrode active material volume fraction" => pybamm.InputParameter("eps_n")
    ))
)
sim = pybamm.Simulation(model, parameter_values=parameter_values)

inputs = OrderedDict{String,Float64}(
    "eps_n" => 0.75,
    "eps_p" => 0.5
)

input_parameter_order = [k for k in keys(inputs)]

# Build ODE problem
prob,cbs = get_ode_problem(sim, 100.0, inputs, cache_type="dual");

# Generate data
sol = solve(prob, QNDF(), saveat=1.0);

y = pybamm.StateVector(pyslice(0, 100))
εₑ⁻ = pybamm.InputParameter("eps_n")
εₑ⁺ = pybamm.InputParameter("eps_p")
get_voltage = pybamm2julia.PybammJuliaFunction([y, εₑ⁻, εₑ⁺], sim.built_model.variables["Terminal voltage [V]"], "get_voltage", false)

var_converter = pybamm2julia.JuliaConverter(inplace=false, input_parameter_order=input_parameter_order, cache_type="dual")
var_converter.convert_tree_to_intermediate(get_voltage)
var_str = var_converter.build_julia_code()
get_voltage = runtime_eval(Meta.parse(pyconvert(String,var_str)))

#generate voltage data
p = prob.p
voltage = Array{Float64}(undef, length(sol.t))
for i in 1:length(sol.t)
    voltage[i] = get_voltage(sol.u[i], p)[1]
end


@model function test_turing(data, prob)
    εₑ⁻ ~ Uniform(1e-2, 1.0)
    εₑ⁺ ~ Uniform(1e-2, 1.0)
    p = [εₑ⁻, εₑ⁺]
    predicted = solve(prob, QNDF(), saveat=1.0, p=p)
    for i in 1:length(predicted)
        data[i] ~ Normal(get_voltage(predicted.u[i], p)[1], 0.001)
    end
end

model = test_turing(voltage, prob)


chain = sample(model, NUTS(), MCMCSerial(), 500, 1)

