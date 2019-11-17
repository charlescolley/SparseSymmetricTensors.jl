
using Test
using LinearAlgebra
include("../src/SSSTensor.jl")

 n = 5
 ord = 5
 tol = 1e-9
 nnz = 10
tempath = "tmp_tensor.ssten"

function set_up(n,ord,nnz)
    x = rand(n)
    indices = rand(1:n,nnz,ord)
    for i = 1:nnz
        indices[i,:] = indices[i,sortperm(indices[i, :])]
    end
    vals = rand(nnz)
    x, indices, vals
end

@testset "CoreOperations Tests" begin

    @testset "Sub Tensor Tests" begin
        indices = [1 1 2 2; 1 2 2 2; 1 3 3 3;2 2 2 3] #fully connected
        vals = [1.0,2.0,3.0,4.0]

        DictA = ssten.SSSTensor([(indices[i,:],vals[i]) for i in 1:4],3)
        COOA  = ssten.COOTen(indices,vals,3)

        sub_DictA = ssten.get_sub_tensor(DictA,[1,2],false)
        sub_COOA  = ssten.get_sub_tensor(COOA,[1,2],false)

        @test sub_COOA == ssten.COOTen(indices[1:2,:],vals[1:2],2)
    end


    @testset "save/load tests" begin
        _, indices, vals = set_up(n,ord,nnz)

        A = ssten.COOTen(indices,vals,n)
        ssten.save(A, tempath)
        B = ssten.load(tempath,true,"COOTen")
        @test A == B

        A = ssten.SSSTensor([(indices[i,:],vals[i]) for i in 1:nnz],n)
        ssten.save(A, tempath)
        B = ssten.load(tempath,true,"DICTen")
        @test A == B
        #check for .ssten file extension functionality

        rm(tempath)
    end

    @testset "flatten" begin
        x = rand(n)
        A = ssten.SSSTensor([([2,3,3],1.0),([1,2,4],-1.0),([3,4,5],1.0)])
        flattened_A = ssten.flatten(A)
        kron_x = reduce(kron,[x for _ in 1:(ssten.order(A)-1)])
        kron_mul = flattened_A*kron_x
        @test LinearAlgebra.norm(ssten.contract(A,x,2) -kron_mul)/LinearAlgebra.norm(kron_mul) < tol
    end


end