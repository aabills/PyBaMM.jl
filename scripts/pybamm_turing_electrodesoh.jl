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
x_0, x_100, y_100, y_0 = pybamm.lithium_ion.electrode_soh.get_min_max_stoichiometries(parameter_values)
parameter_values.update(
    PyDict(Dict(
        "Positive electrode active material volume fraction" => pybamm.InputParameter("eps_p"),
        "Negative electrode active material volume fraction" => pybamm.InputParameter("eps_n")
    ))
)


V_init = pybamm2julia.PsuedoInputParameter("test")
V_min = parameter_values.evaluate(param.voltage_low_cut)
V_max = parameter_values.evaluate(param.voltage_high_cut)


soc_model = pybamm.BaseModel()
soc = pybamm.StateVector(pyslice(0,1))

param = pybamm.LithiumIonParameters()

Up = param.p.prim.U
Un = param.n.prim.U
T_ref = parameter_values["Reference temperature [K]"]
x = x_0 + soc * (x_100 - x_0)
y = y_0 - soc * (y_0 - y_100)
soc_model.algebraic[soc] = Up(y, T_ref) - Un(x, T_ref) - V_init

soc_model.initial_conditions[soc] = (V_init - V_min) / (V_max - V_min)
disc = pybamm.Discretisation()
soc_model.variables["soc"] = soc
parameter_values.process_model(soc_model)

rhs = soc_model.algebraic[soc]

pbj_stoichs = pybamm2julia.PybammJuliaFunction([y, soc, V_init], rhs, "stoich_function", true)

pbj_converter = pybamm2julia.JuliaConverter(cache_type="dual")
pbj_converter.convert_tree_to_intermediate(pbj_stoichs)
pbj_str = pyconvert(String, pbj_converter.build_julia_code())

stoich_function = runtime_eval(Meta.parse(pbj_str))

#inject first voltage



