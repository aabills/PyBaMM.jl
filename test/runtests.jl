using PyBaMM
using Test
using SafeTestsets
using SparseArrays, LinearAlgebra

#@safetestset "Compare models with PyBaMM" begin include("test_full_models.jl") end
#@safetestset "Test loss functions" begin include("test_loss.jl") end
#@safetestset "Test Event Handling" begin include("test_events.jl") end
#@safetestset "Test caches" begin include("test_caches.jl") end
#@safetestset "Test jacobians" begin include("test_jacobian.jl") end
@safetestset "Test pack modeling" begin include("test_pack.jl") end
@safetestset "Test Interface for Pure Julia Functions" begin include("test_pbj.jl") end