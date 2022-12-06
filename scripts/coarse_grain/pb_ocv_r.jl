using PyBaMM
using ProgressMeter
using Plots
plotly()

pybamm = PyBaMM.pybamm
pack = PyBaMM.pack
pybamm2julia = PyBaMM.pybamm2julia
setup_circuit = PyBaMM.setup_circuit

pv_chen = pybamm.ParameterValues("Chen2020")

ocv_R = pybamm.BaseModel()

#Parameters
R = pybamm.Parameter("Cell resistance [Ohms]")
C = pybamm.Parameter("Heat capacity [J.K-1]")
h = pybamm.Parameter("Total heat transfer coefficient times area [W.K-1]")
con_n_start = pybamm.Parameter("Initial concentration in negative electrode [mol.m-3]")
con_p_start = pybamm.Parameter("Initial concentration in positive electrode [mol.m-3]")
current = pybamm.Parameter("Current function [A]")

pos_elec_cap = pybamm.Parameter("Positive electrode capacity [A.h]")
neg_elec_cap = pybamm.Parameter("Negative electrode capacity [A.h]")
T_amb = pybamm.Parameter("Ambient temperature [K]")
P_OCV = pv_chen["Positive electrode OCP [V]"]
N_OCV = pv_chen["Negative electrode OCP [V]"]


#Variables
terminal_voltage = pybamm.Variable("Terminal voltage [V]")
cell_temperature = pybamm.Variable("Cell temperature [K]")
pos_elec_sto = pybamm.Variable("Positive electrode stoichiometry")
neg_elec_sto = pybamm.Variable("Negative electrode stoichiometry")

#equations
d_p_sto_dt = current/3600/pos_elec_cap
d_n_sto_dt = -current/3600/neg_elec_cap
dTdt = (current*current*R - h*(cell_temperature-T_amb))/C

ocv_R.rhs = pydict(Dict(
    pos_elec_sto => d_p_sto_dt,
    neg_elec_sto => d_n_sto_dt,
    cell_temperature => dTdt
))

#Variables
ocv_R.variables = pydict(Dict(
    "Cell temperature [K]" => cell_temperature,
    "Terminal voltage [V]" => P_OCV(pos_elec_sto) - N_OCV(neg_elec_sto) - current*R
))

pv_chen.update(
    pydict(
        Dict(
            "Cell resistance [Ohms]" => 0.05,
            "Heat capacity [J.K-1]" => 1.5,
            "Total heat transfer coefficient times area [W.K-1]" => 1,
            "Positive electrode capacity [A.h]" => pv_chen["Nominal cell capacity [A.h]"],
            "Negative electrode capacity [A.h]" => pv_chen["Nominal cell capacity [A.h]"],
        )
    ),
    check_already_exists=false
)
ocv_R.initial_conditions = pydict(Dict(pos_elec_sto=> y, neg_elec_sto => x,cell_temperature => 298))


x,y = pybamm.lithium_ion.get_initial_stoichiometries(1.0,pv_chen)



submesh_types = pydict()
var_pts = pydict()
geometry = pydict()

mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

spatial_methods = pydict()
disc = pybamm.Discretisation(mesh, spatial_methods)

param.process_model(ocv_R)
bm = disc.process_model(ocv_R, remove_independent_variables_from_rhs=false)


myconverter = pybamm2julia.JuliaConverter()
myconverter.convert_tree_to_intermediate(bm.concatenated_rhs)
jl_str = pyconvert(String, myconverter.build_julia_code())

