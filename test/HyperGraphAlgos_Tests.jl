using Test
using LinearAlgebra
include("../src/SSSTensor.jl")

 n = 5
 ord = 5
 tol = 1e-9
 nnz = 10

function set_up(n,ord,nnz)
    x = rand(n)
    indices = rand(1:n,nnz,ord)
    for i = 1:nnz
        indices[i,:] = indices[i,sortperm(indices[i, :])]
    end
    vals = rand(nnz)
    x, indices, vals
end


@testset "HyperGraphAlgos Tests" begin

    @testset "find_edge_incidence tests" begin
#         _, indices, vals = set_up(n,ord,nnz)
         indices = [1 1 2; 1 2 2; 3 3 3]
         vals = [1.0,2.0,3.0]
         edges = [(indices[i,:],vals[i]) for i in 1:3]


         DictA = ssten.SSSTensor(edges,n)
         COOA  = ssten.COOTen(indices,vals,n)
         DictA_incidence = ssten.find_edge_incidence(DictA)
         COOA_incidence = ssten.find_edge_incidence(DictA)

         @test COOA_incidence == DictA_incidence
         @test COOA_incidence[1] == COOA_incidence[2]
         @test COOA_incidence[3] == Set([edges[3]])

    end


end
