using PyBaMM
using LinearSolve
using TerminalLoggers
using JLD2
using Plots
plotly()

pybamm = pyimport("pybamm")
pybamm2julia = pyimport("pybamm2julia")
setup_circuit = pyimport("setup_circuit")
pack = pyimport("pack")
coarse_pack = pyimport("coarse_pack")


Np = 3
Ns = 3

curr = 0.6*Np

p = nothing 
t = 0.0
functional = true

options = Dict("thermal" => "lumped")


full_model = pybamm.lithium_ion.DFN(name="DFN", options=options)
reduced_model = pybamm.lithium_ion.SPMe(name="spm", options=options)


#Really all this should be in setup_circuit. And setup_circuit should handle thermals
netlist = setup_circuit.setup_circuit(Np, Ns, I=curr)
full_cells = ["V0"]
voltage_functional = true
    
pybamm_pack = coarse_pack.CoarsePack(
    full_model,
    reduced_model, 
    full_cells, 
    netlist, 
    functional=functional, 
    thermal=true, 
    left_bc = "symmetry", 
    voltage_functional=voltage_functional
)
pybamm_pack.build_pack()


timescale = pyconvert(Float64,pybamm_pack.reduced_timescale.evaluate())
full_timescale = pyconvert(Float64,pybamm_pack.full_timescale.evaluate())
if timescale!=full_timescale
    @error "shit"
end

if functional
    fullcellconverter = pybamm2julia.JuliaConverter(cache_type = "symbolic", inplace=true)
    fullcellconverter.convert_tree_to_intermediate(pybamm_pack.full_cell_model)
    full_cell_str = fullcellconverter.build_julia_code()
    full_cell_str = pyconvert(String, full_cell_str)
    full_cell! = eval(Meta.parse(full_cell_str))
else
    full_cell_str = ""
end


if voltage_functional
    fullvoltageconverter = pybamm2julia.JuliaConverter(cache_type = "symbolic", inplace=true)
    fullvoltageconverter.convert_tree_to_intermediate(pybamm_pack.full_voltage_func)
    full_voltage_str = fullvoltageconverter.build_julia_code()
    full_voltage_str = pyconvert(String, full_voltage_str)
    full_voltage_func = eval(Meta.parse(full_voltage_str))
else
    full_voltage_str = ""
end


if functional
    reducedcellconverter = pybamm2julia.JuliaConverter(cache_type = "symbolic", inplace=true)
    reducedcellconverter.convert_tree_to_intermediate(pybamm_pack.reduced_cell_model)
    reduced_cell_str = reducedcellconverter.build_julia_code()
    reduced_cell_str = pyconvert(String, reduced_cell_str)
    reduced_cell! = eval(Meta.parse(reduced_cell_str))
else
    full_cell_str = ""
end


if voltage_functional
    reducedvoltageconverter = pybamm2julia.JuliaConverter(cache_type = "symbolic", inplace=true)
    reducedvoltageconverter.convert_tree_to_intermediate(pybamm_pack.reduced_voltage_func)
    reduced_voltage_str = reducedvoltageconverter.build_julia_code()
    reduced_voltage_str = pyconvert(String, reduced_voltage_str)
    reduced_voltage_func = eval(Meta.parse(reduced_voltage_str))
else
    reduced_voltage_str = ""
end


myconverter = pybamm2julia.JuliaConverter(cache_type = "symbolic")
myconverter.convert_tree_to_intermediate(pybamm_pack.pack)
pack_str = myconverter.build_julia_code()

icconverter = pybamm2julia.JuliaConverter(override_psuedo = true)
icconverter.convert_tree_to_intermediate(pybamm_pack.ics)
ic_str = icconverter.build_julia_code()

u0 = eval(Meta.parse(pyconvert(String,ic_str)))
jl_vec = u0()

pack_str = pyconvert(String, pack_str)
jl_func = eval(Meta.parse(pack_str))

dy = similar(jl_vec)


println("Calculating Jacobian Sparsity")
jac_sparsity = float(Symbolics.jacobian_sparsity((du,u)->jl_func(du,u,p,t),dy,jl_vec))

if functional
    fullcellconverter = pybamm2julia.JuliaConverter(cache_type = "dual", inplace=true)
    fullcellconverter.convert_tree_to_intermediate(pybamm_pack.full_cell_model)
    full_cell_str = fullcellconverter.build_julia_code()
    full_cell_str = pyconvert(String, full_cell_str)
    full_cell! = eval(Meta.parse(full_cell_str))
else
    full_cell_str = ""
end


if voltage_functional
    fullvoltageconverter = pybamm2julia.JuliaConverter(cache_type = "dual", inplace=true)
    fullvoltageconverter.convert_tree_to_intermediate(pybamm_pack.full_voltage_func)
    full_voltage_str = fullvoltageconverter.build_julia_code()
    full_voltage_str = pyconvert(String, full_voltage_str)
    full_voltage_func = eval(Meta.parse(full_voltage_str))
else
    full_voltage_str = ""
end


if functional
    reducedcellconverter = pybamm2julia.JuliaConverter(cache_type = "dual", inplace=true)
    reducedcellconverter.convert_tree_to_intermediate(pybamm_pack.reduced_cell_model)
    reduced_cell_str = reducedcellconverter.build_julia_code()
    reduced_cell_str = pyconvert(String, reduced_cell_str)
    reduced_cell! = eval(Meta.parse(reduced_cell_str))
else
    full_cell_str = ""
end


if voltage_functional
    reducedvoltageconverter = pybamm2julia.JuliaConverter(cache_type = "dual", inplace=true)
    reducedvoltageconverter.convert_tree_to_intermediate(pybamm_pack.reduced_voltage_func)
    reduced_voltage_str = reducedvoltageconverter.build_julia_code()
    reduced_voltage_str = pyconvert(String, reduced_voltage_str)
    reduced_voltage_func = eval(Meta.parse(reduced_voltage_str))
else
    reduced_voltage_str = ""
end

myconverter = pybamm2julia.JuliaConverter(cache_type = "dual")
myconverter.convert_tree_to_intermediate(pybamm_pack.pack)
pack_str = myconverter.build_julia_code()

pack_voltage_index = Np + 1
pack_voltage = 1.0
jl_vec[1:Np] .=  curr
jl_vec[pack_voltage_index] = pack_voltage

pack_str = pyconvert(String, pack_str)
jl_func = eval(Meta.parse(pack_str))

#build mass matrix.
pack_eqs = falses(pyconvert(Int,pybamm_pack.len_pack_eqs))


full_cell_rhs = trues(pyconvert(Int,pybamm_pack.len_full_cell_rhs))
full_cell_algebraic = falses(pyconvert(Int,pybamm_pack.len_full_cell_algebraic))
full_cell_diff_vars = vcat(full_cell_rhs,full_cell_algebraic)
reduced_cell_rhs = trues(pyconvert(Int,pybamm_pack.len_reduced_cell_rhs))
reduced_cell_algebraic = falses(pyconvert(Int,pybamm_pack.len_reduced_cell_algebraic))
reduced_cell_diff_vars = vcat(reduced_cell_rhs,reduced_cell_algebraic)

arr = falses(size(jl_vec) .- length(pack_eqs))


for batt in pybamm_pack.batteries
    this_offset = pyconvert(Int,pybamm_pack.batteries[batt]["offset"]) - length(pack_eqs) + 1
    if pyconvert(String,batt) in full_cells
        len = pyconvert(Int,pybamm_pack.full_cell_size)
        diff_vars = full_cell_diff_vars
    else
        len = pyconvert(Int,pybamm_pack.reduced_cell_size)
        diff_vars = reduced_cell_diff_vars
    end
    arr[this_offset:(this_offset+len-1)] = diff_vars
end


differential_vars = vcat(pack_eqs,arr)
mass_matrix = sparse(diagm(differential_vars))
func = ODEFunction(jl_func, mass_matrix=mass_matrix,jac_prototype=jac_sparsity)
prob = ODEProblem(func, jl_vec, (0.0, 3600/timescale), nothing)


sol = solve(prob, Trapezoid(linsolve=KLUFactorization(),concrete_jac=true))

