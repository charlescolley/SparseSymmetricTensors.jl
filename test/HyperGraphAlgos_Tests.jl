using Test
using LinearAlgebra
include("../src/SSSTensor.jl")

 n = 5
 ord = 5
 tol = 1e-9
 nnz = 10
 tmppath = "tempLargestComp"

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


    @testset "Largest Connected Component Tests" begin

        _, indices, vals = set_up(n,ord,nnz)
        indices = cat(indices,(n+1)*ones(Int,ord)',1)

        A = ssten.SSSTensor([(indices[i,:],vals[i]) for i in 1:nnz])

        get_largest_component(A::SSSTensor,filepath=tmppath)


    end

    #do we need order tests?
    @testset "Connected Components Tests" begin
        indices = [1 1 2 3; 1 2 2 2; 1 3 3 3] #fully connected
        vals = [1.0,2.0,3.0]

        DictA = ssten.SSSTensor([(indices[i,:],vals[i]) for i in 1:3],3)
        COOA  = ssten.COOTen(indices,vals,3)

        assignment, sizes, _ = ssten.connected_components(DictA)
        @test assignment == ones(3)
        @test sizes[1] == 3

        COOA  = ssten.COOTen(indices,vals,3)
        assignment, sizes, orders = ssten.connected_components(COOA)
        @test assignment == ones(3)
        @test sizes[1] == 3

    end

    @testset "find_edge_incidence Tests" begin

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
