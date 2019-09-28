using Test
#using MATLAB
#using SSST
include("../src/SSSTensor.jl")


# define common variables usable in all tests here

n = 5
valid_edges = [([1,2,3],1.0),([1,2,2],-1.0),([3,3,3],1)]
unordered_edge = [([2,3,1],1.0)]
negative_index = [([-1,2,3],1.0)]

include("SSSTensor_Tests.jl")
include("CoreOperations_Tests.jl")
include("Helpers_Tests.jl")
include("HyperGraphAlgos_Tests.jl")
include("NumericalRoutines_Tests.jl")