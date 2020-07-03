using Test
#using MATLAB
#using SSST
using LinearAlgebra

# work around until added to General Registry
include("../src/SparseSymmetricTensors.jl")
using Main.SparseSymmetricTensors

function set_up(n,ord,nnz)
    x = rand(n)
    indices = rand(1:n,nnz,ord)
    for i = 1:nnz
        indices[i,:] = indices[i,sortperm(indices[i, :])]
    end
    vals = rand(nnz)
    x, indices, vals
end

# define common variables usable in all tests here

n = 5
valid_edges = [([1,2,3],1.0),([1,2,2],-1.0),([3,3,3],1)]
unordered_edge = [([2,3,1],1.0)]
negative_index = [([-1,2,3],1.0)]

# ------------      used in CoreOperations    ------------    #
 ord = 5
 tol = 1e-9
 nnz = 10
tempath = "tmp_tensor.ssten"

#include("SparseSymmetricTensors_Tests.jl")
#include("CoreOperations_Tests.jl")
include("Helpers_Tests.jl")
#include("HyperGraphAlgos_Tests.jl")
#include("NumericalRoutines_Tests.jl")