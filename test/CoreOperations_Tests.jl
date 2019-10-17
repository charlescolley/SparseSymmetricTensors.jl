
using Test
using LinearAlgebra
include("../src/SSSTensor.jl")

 n = 5
 tol = 1e-9

@testset "CoreOperations Tests" begin

  @testset "save/load tests" begin
    A = SSSTensor(valid_edges)
  end

  @testset "flatten" begin
    x = rand(n)
    A = ssten.SSSTensor([([2,3,3],1.0),([1,2,4],-1.0),([3,4,5],1)])
    flattened_A = ssten.flatten(A)
    kron_x = reduce(kron,[x for _ in 1:(ssten.order(A)-1)])
    kron_mul = flattened_A*kron_x
    @assert LinearAlgebra.norm(ssten.contract(A,x,2) -kron_mul)/LinearAlgebra.norm(kron_mul) < tol
  end

end