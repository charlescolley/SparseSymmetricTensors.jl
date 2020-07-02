using Test
import LinearAlgebra.norm
include("../src/SSSTensor.jl")

 n = 3
 ord = 5
 tol = 1e-9
 nnz = 4

function set_up(n,ord,nnz)
    x = rand(n)
    indices = rand(1:n,nnz,ord)
    for i = 1:nnz
        indices[i,:] = indices[i,sortperm(indices[i, :])]
    end
    vals = rand(nnz)
    x, indices, vals
end


@testset "Contraction Tests" begin

  @testset "COOTen contraction tests" begin
    x, indices, vals = set_up(n,ord,nnz)

    DICTen_A = ssten.SSSTensor([(indices[i,:],vals[i]) for i in 1:nnz],n)
    COOTen_A  = ssten.COOTen(indices,vals,n)

    DICTen_contract = ssten.contract(DICTen_A,x,ord-1)
    COOTen_contract  = ssten.contract_k_1(COOTen_A,x)

    println(DICTen_contract)
    println(COOTen_contract)

    @test norm(DICTen_contract - COOTen_contract)/norm(COOTen_contract) < tol

    COOTen_contract = zeros(n)
    ssten.contract_k_1!(indices,vals,x,COOTen_contract)

    @test norm(DICTen_contract - COOTen_contract)/norm(COOTen_contract)< tol
  end

  @testset "inner product tests" begin
    _, indices, vals = set_up(n,ord,nnz)
    indices = [1 2 3; 2 3 4; 1 3 4; 1 2 4; 2 3 5; 1 2 5]
    vals = [.5,2.3,3.2,3.2,4.3,4.3]

    COOTen_A  = ssten.COOTen(indices,vals)

    test_vals = copy(vals)
    test_vals = test_vals.^2
    test_vals .*= 6

    @test ssten.inner_product(COOTen_A ,COOTen_A) == sum(test_vals)

  end
  #=
  @testset "in place low mem contraction tests" begin
    x = rand(n)
    indices = rand(1:n,nnz,ord)
    for i =1:nnz
        indices[i,:] = indices[i,sortperm(indices[i, :])]
    end
    vals = rand(nnz)

    A = ssten.SSSTensor([(indices[i,:],vals[i]) for i in 1:nnz])

    dictTen_contract = ssten.contract(A,x,ord-1)
    matTen_contract = zeros(n)
    ssten.contract_k_1!(indices,vals,x,matTen_contract)

    @test norm(dictTen_contract - matTen_contract)/norm(matTen_contract)< tol
  end
  =#
end