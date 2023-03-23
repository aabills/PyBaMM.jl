using PyBaMM
using NonlinearSolve
using ProgressMeter
using Statistics
using Turing

#pybamm all lower case is the python module pybamm
pybamm = PyBaMM.pybamm
#pybamm2julia is the python module we will use to convert from pybamm to julia (duh)
pybamm2julia = PyBaMM.pybamm2julia

eps_n = pybamm.InputParameter("eps_n")
eps_p = pybamm.InputParameter("eps_p")
input_parameter_order = ["eps_p", "eps_n"]


parameter_values = pybamm.ParameterValues("Chen2020")
parameter_values.update(
    PyDict(Dict(
        "Positive electrode active material volume fraction" => pybamm.InputParameter("eps_p"),
        "Negative electrode active material volume fraction" => pybamm.InputParameter("eps_n")
    ))
)

param = pybamm.LithiumIonParameters()

Un = param.n.prim.U
Up = param.p.prim.U

model = pybamm.BaseModel()


V_min = parameter_values.evaluate(param.voltage_low_cut)
V_max = parameter_values.evaluate(param.voltage_high_cut)

Q_n = parameter_values.process_symbol(param.n.Q_init)
Q_p = parameter_values.process_symbol(param.p.Q_init)
Q_Li = parameter_values.process_symbol(param.Q_Li_particles_init)

x_100 = pybamm.StateVector(pyslice(0,1))

y_100 = (Q_Li - x_100 * Q_n) / Q_p

Un_100 = Un(x_100, T_ref)
Up_100 = Up(y_100, T_ref)

model.algebraic[x_100] = Up_100 - Un_100 - V_max

x_0 = pybamm.StateVector(pyslice(1,2))

Q = Q_n * (x_100 - x_0)
y_0 = y_100 + Q / Q_p
Un_0 = Un(x_0, T_ref)
Up_0 = Up(y_0, T_ref)
model.algebraic[x_0] = Up_0 - Un_0 - V_min

V_init = 4.1

soc = pybamm.StateVector(pyslice(2,3))
x = x_0 + soc * (x_100 - x_0)
y = y_0 - soc * (y_0 - y_100)
model.algebraic[soc] = Up(y, T_ref) - Un(x, T_ref) - V_init



parameter_values.process_model(model)

rhs = pybamm.numpy_concatenation(model.algebraic[x_100], model.algebraic[x_0], model.algebraic[soc])
y = pybamm.StateVector(pyslice(0,2))

initial_conc = pybamm2julia.PybammJuliaFunction([y, eps_p, eps_n], rhs, "stoich_function", true)

initial_conc_converter = pybamm2julia.JuliaConverter(cache_type="dual", input_parameter_order=input_parameter_order)
initial_conc_converter.convert_tree_to_intermediate(initial_conc)
initial_conc_str = pyconvert(String, initial_conc_converter.build_julia_code())

initial_conc_func = runtime_eval(Meta.parse(initial_conc_str))

eps_p_init = 0.665
eps_n_init = 0.7


p = [eps_p_init, eps_n_init]

test_vals = [0.9106121196114546, 0.026347301451411866, 0.9]

out = zeros(3) .+ 100
y = test_vals

nlp = NonlinearProblem(initial_conc_func, test_vals, p=p)

solve(nlp, NewtonRaphson())

function f(p)
    local_prob = NonlinearProblem(initial_conc_func, test_vals, p=p)
    sol = solve(local_prob, NewtonRaphson())
    return sol.u
end

@model function turing_thing(soc)
    eps_p ~ Uniform(0.0, 1.0)
    eps_n ~ Uniform(0.0, 1.0)
    p = [eps_p, eps_n]
    prob = NonlinearProblem(initial_conc_func, test_vals, p=p)
    predicted = try
        sol = solve(prob, NewtonRaphson())
        predicted = sol.u[end]
    catch
        predicted = 100.
    end
    soc ~ Normal(predicted, 0.001)
    return nothing
end

turingmodel = turing_thing(0.9)
chain = sample(turingmodel, NUTS(), MCMCSerial(), 5000, 1)
    