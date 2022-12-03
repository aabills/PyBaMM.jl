using PyBaMM
using LinearSolve
using TerminalLoggers
using JLD2

pybamm = PyBaMM.pybamm
pack = PyBaMM.pack
pybamm2julia = PyBaMM.pybamm2julia
setup_circuit = PyBaMM.setup_circuit



Np = 3
Ns = 3

curr = 1.8

p = nothing 
t = 0.0
functional = true

options = Dict("thermal" => "lumped")


model = pybamm.lithium_ion.DFN(name="DFN", options=options)

netlist = setup_circuit.setup_circuit(Np, Ns, I=curr)
    
pybamm_pack = pack.Pack(model, netlist, functional=functional, thermal=true, implicit=true)
pybamm_pack.build_pack()

timescale = pyconvert(Float64,pybamm_pack.timescale.evaluate())

if functional
    cellconverter = pybamm2julia.JuliaConverter(inplace=true, cache_type = "dual")
    cellconverter.convert_tree_to_intermediate(pybamm_pack.cell_model)
    cell_str = cellconverter.build_julia_code()
    cell_str = pyconvert(String, cell_str)
    cell! = eval(Meta.parse(cell_str))
else
    cell_str = ""
end

packconverter = pybamm2julia.JuliaConverter(override_psuedo=true, cache_type = "dual")
packconverter.convert_tree_to_intermediate(pybamm_pack.pack)
pack_str = packconverter.build_julia_code()

icconverter = pybamm2julia.JuliaConverter(override_psuedo = true)
icconverter.convert_tree_to_intermediate(pybamm_pack.ics)
ic_str = icconverter.build_julia_code()

u0 = eval(Meta.parse(pyconvert(String,ic_str)))
jl_vec = u0()

du0 = deepcopy(jl_vec)

pack_voltage_index = Np + 1
pack_voltage = 1.0
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

prob = DAEProblem(jl_func, du0, jl_vec, (0.0, 3600/timescale), nothing, differential_vars=differential_vars)


#Haven't gotten GMRES to work
@time sol =  solve(prob, IDA())
@time sol = solve(prob, IDA(), save_everystep=false)
