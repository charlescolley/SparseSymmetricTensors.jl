using Test
#using MATLAB
#using SSST
include("../src/SSSTensor.jl")



n = 5

@testset "Contructor Tests" begin
  valid_edges = [([1,2,3],1.0),([1,2,2],-1.0),([3,3,3],1)]
  unordered_edge = [([2,3,1],1.0)]
  negative_index = [([-1,2,3],1.0)]
  @testset "Hyperedge List Constructor" begin
    @test_throws ErrorException SSSTensor(unordered_edge) #unsorted indices
    @test_throws ErrorException SSSTensor(unordered_edge,n)
    @test_throws ErrorException SSSTensor(unordered_edge,1) # too small cubical dim
    @test_throws ErrorException SSSTensor(valid_edges,1)
    @test_throws ErrorException SSSTensor(negative_index,1)
    @test_throws ErrorException SSSTensor(negative_index,n) #neg index
    @test_throws ErrorException SSSTensor(negative_index)
  end

  @testset "Dense Tensor Constructor" begin
    non_sym_tensor = rand(2,2,2,2)
    sym_tensor = ones(2,2,2,2)
    @test_throws ErrorException SSSTensor(non_sym_tensor)
    @test_throws ErrorException SSSTensor(sym_tensor,1) # too small cubical dim
    @test_throws ErrorException SSSTensor(non_sym_tensor,1)

  end

  @testset "Dictionary Constructor" begin
    D_valid_edges = Dict(valid_edges)
    D_unordered_edge = Dict(unordered_edge)
    D_negative_index = Dict(negative_index)

    @test_throws ErrorException SSSTensor(D_unordered_edge) #unsorted indices
    @test_throws ErrorException SSSTensor(D_unordered_edge,n)
    @test_throws ErrorException SSSTensor(D_unordered_edge,1) # too small cubical dim
    @test_throws ErrorException SSSTensor(D_valid_edges,1)
    @test_throws ErrorException SSSTensor(D_negative_index,1)
    @test_throws ErrorException SSSTensor(D_negative_index,n) #neg index
    @test_throws ErrorException SSSTensor(D_negative_index)
  end
  @test_throws ErrorException SSSTensor()

end

@testset "multiplicity_tests" begin
    @test multiplicity_factor([1,1,1,1,1]) == 1
    @test multiplicity_factor([1,2,1,1,1]) == 5
    @test multiplicity_factor([2,2,1,1,1]) == 10
end

#=
"""-----------------------------------------------------------------------------
                     Compute Tensor eigenvectors in MATLAB
-----------------------------------------------------------------------------"""
function MATLAB_eigs(A::SSST.SSSTensor)

    data = Array{Array{Any,1}}(undef,length(A.edges))
    i = 1
    for (indices, val) in A.edges
        data[i] = [indices,SSST.multiplicity_factor(indices),val]
        i += 1
    end

    xs = mxarray(data)

    n = A.cubical_dimension

    mat"""
        disp(xs)
    """

    #=

    mat"""

    %build homogenous polynomial

    mpol('x',$n)
    mpol('f',1)

    f = $xs{1}{2}*$xs{1}{3}*prod(x($xs{1}{1}));

    for i =2:length(xs)
        f = f + $xs{i}{2}*$xs{i}{3}*prod(x($xs{i}{1}));
    end

    %call AReig solver

    [$lmd, $eigvec,  $info] = AReigSTensors(f, [], x, 3, 2)

    """
    println(lmd)
    println(eigvec)

    =#

end
=#