

using Test
import LinearAlgebra.norm
include("../src/SSSTensor.jl")

 n = 3
 ord = 3
 tol = 1e-9
 nnz = 4

@testset "Contraction Tests" begin

  @testset "low mem contraction tests" begin
    x = rand(n)
    indices = rand(1:n,nnz,ord)
    for i =1:nnz
        indices[i,:] = indices[i,sortperm(indices[i, :])]
    end
    vals = rand(nnz)

    A = ssten.SSSTensor([(indices[i,:],vals[i]) for i in 1:nnz])

    dictTen_contract = ssten.contract(A,x,ord-1)
    matTen_contract = ssten.contract_k_1(indices,vals,n,x)

    @assert norm(dictTen_contract - matTen_contract)/norm(matTen_contract)< tol
  end

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

    @assert norm(dictTen_contract - matTen_contract)/norm(matTen_contract)< tol
  end
end