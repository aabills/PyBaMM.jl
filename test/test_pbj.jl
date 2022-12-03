using PyBaMM


function add_5(out,y)
    @. out = y .+ 5
    return nothing
end


@testset "test pure julia function" begin
    pybamm = PyBaMM.pybamm
    pybamm2julia = PyBaMM.pybamm2julia
    
    sv = pybamm.StateVector(pyslice(0,5))
    b =  pybamm.Scalar(5.0)
    c = sv + b
    
    c_jl = pybamm2julia.PybammJuliaFunction([sv],c,"add_pb",true)
    d = pybamm2julia.PybammJuliaFunction([sv],nothing, "add_5",true, shape=(5,1))
    d_o = pybamm2julia.PybammJuliaFunction([sv],d,"add_outer!",true)

    d_converter = pybamm2julia.JuliaConverter()
    d_converter.convert_tree_to_intermediate(d_o)
    jl_d_string = d_converter.build_julia_code()

    c_converter = pybamm2julia.JuliaConverter()
    c_converter.convert_tree_to_intermediate(c_jl)
    jl_c_string = c_converter.build_julia_code()

    jl_d_func = eval(Meta.parse(pyconvert(String,jl_d_string)))
    jl_c_func = eval(Meta.parse(pyconvert(String, jl_c_string)))

    x1 = zeros(5)
    x2 = zeros(5)

    y = ones(5)

    jl_d_func(x1,y)
    jl_c_func(x2,y)

    @test x1 == x2
end