using PyBaMM
using ProgressMeter
using Statistics

#pybamm all lower case is the python module pybamm
pybamm = PyBaMM.pybamm
#pybamm2julia is the python module we will use to convert from pybamm to julia (duh)
pybamm2julia = PyBaMM.pybamm2julia

#load the model just like always. Any python code here will behave identically to normal python code.
model = pybamm.lithium_ion.SPMe(name="DFN")
sim = pybamm.Simulation(model)
sim.build(initial_soc = 1)

####################################### Get Julia Model #######################################

#Begin the conversion by initiating a JuliaConverter object. We have various cache types, including:
#   "symbolic" : enables usage of Symbolics.jl
#   "dual" : enables usage of ForwardDiff automatic differentiation
# and more, but these are the useful ones. Default is Float64.
cellconverter = pybamm2julia.JuliaConverter(cache_type = "dual", inplace=true)

# Now, convert the model. Note that this is only true for explicit ODE models, the 
# following code will need to be modified for DFN models. 
cellconverter.convert_tree_to_intermediate(sim.built_model.concatenated_rhs)

# Finally, the cell_str is a string of julia code that defines a function which can be used with DifferentialEquations.jl
cell_str = cellconverter.build_julia_code()

# Cell string needs to be turned from a python string into a julia string.
cell_str = pyconvert(String, cell_str)

# This takes the string and turns it into a function
cell! = eval(Meta.parse(cell_str))

####################################### Get Initial Conditions #######################################
#don't worry too much about this stuff for now, it's not important for SPMe/SPM
get_consistent_ics_solver = pybamm.CasadiSolver()
get_consistent_ics_solver.set_up(sim.built_model)
get_consistent_ics_solver._set_initial_conditions(sim.built_model,0.0,pydict(Dict()), false)
ics = pybamm.Vector(sim.built_model.y0.full())

#PBJ. This takes ics, which is any arbitrary pybamm model, and converts it to julia code. The first argument
#is the inputs to the functions. In this case, none. second argument is the pybamm model. Third argument is the name
# fourth model is whether or not to make it an inplace function (important for speed). 
u0_pbj = pybamm2julia.PybammJuliaFunction([],ics,"u0",inplace=false)
# Now, we can convert u0_pbj to a julia function (no need to worry about the cache here.)
u0_converter = pybamm2julia.JuliaConverter()
u0_converter.convert_tree_to_intermediate(u0_pbj)
u0_str = u0_converter.build_julia_code()
u0 = eval(Meta.parse(pyconvert(String,u0_str)))

# Run the function and get the initial conditions!
jl_vec = u0()

#finally, solve the model. At this point we have everything we need from pybamm.
prob = ODEProblem(cell!, jl_vec, (0., 3600.))
sol = solve(prob, QNDF())
#note that this won't be particularly fast, this is a simplified version of what we're going to do later on.


####################################### Get Variables #######################################

#to illustrate how this works in a bit more detail, we will now get the voltage from the simulation.
var_name = "Terminal voltage [V]"

#sim.built_model.variables[var_name] is a pybamm expression tree corresponding to a pybamm function.
#to get voltage, all we need is y (the state vector.)
y = pybamm.StateVector(pyslice(0, 100))
get_voltage = pybamm2julia.PybammJuliaFunction([y], sim.built_model.variables[var_name], "get_voltage!", true)

var_converter = pybamm2julia.JuliaConverter()
var_converter.convert_tree_to_intermediate(get_voltage)
var_str = var_converter.build_julia_code()
get_voltage! = runtime_eval(Meta.parse(pyconvert(String,var_str)))

# Evaluate and fill in the vector
var = Array{Float64}(undef, length(sol.t))
out = [0.0]
for i in 1:length(sol.t)
    # Updating 'out' in-place. notice that we have 4 inputs.
    get_voltage!(out, sol.u[i])
    var[i] = out[1]
end
#note that the previous procedure will work for any variable. 


