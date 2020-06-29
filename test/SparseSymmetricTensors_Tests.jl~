
using Test
#using MATLAB

include("../src/SSSTensor.jl")
using .ssten




@testset "Contructor Tests" begin

@testset "DictTen Constructor Tests" begin

    n = 5
    valid_edges = [([1,2,3],1.0),([1,2,2],-1.0),([3,3,3],1.0)]
    unordered_edge = [([2,3,1],1.0)]
    negative_index = [([-1,2,3],1.0)]

    @testset "Hyperedge List Constructor" begin

        @test_throws ErrorException ssten.SSSTensor(unordered_edge) #unsorted indices
        @test_throws ErrorException ssten.SSSTensor(unordered_edge,n)
        @test_throws ErrorException ssten.SSSTensor(unordered_edge,1) # too small cubical dim
        @test_throws ErrorException ssten.SSSTensor(valid_edges,1)
        @test_throws ErrorException ssten.SSSTensor(negative_index,1)
        @test_throws ErrorException ssten.SSSTensor(negative_index,n) #neg index
        @test_throws ErrorException ssten.SSSTensor(negative_index)

    end

    @testset "Dense Tensor Constructor" begin

        non_sym_tensor = rand(2,2,2,2)
        sym_tensor = ones(2,2,2,2)
        @test_throws ErrorException ssten.SSSTensor(non_sym_tensor)
        @test_throws ErrorException ssten.SSSTensor(sym_tensor,1) # too small cubical dim
        @test_throws ErrorException ssten.SSSTensor(non_sym_tensor,1)

    end

    @testset "Dictionary Constructor" begin

        D_valid_edges = Dict(valid_edges)
        D_unordered_edge = Dict(unordered_edge)
        D_negative_index = Dict(negative_index)

        @test_throws ErrorException ssten.SSSTensor(D_unordered_edge) #unsorted indices
        @test_throws ErrorException ssten.SSSTensor(D_unordered_edge,n)
        @test_throws ErrorException ssten.SSSTensor(D_unordered_edge,1) # too small cubical dim
        @test_throws ErrorException ssten.SSSTensor(D_valid_edges,1)
        @test_throws ErrorException ssten.SSSTensor(D_negative_index,1)
        @test_throws ErrorException ssten.SSSTensor(D_negative_index,n) #neg index
        @test_throws ErrorException ssten.SSSTensor(D_negative_index)

    end

    @test_throws ErrorException ssten.SSSTensor()
end

@testset "COOTEN" begin

    valid_indices = [1 2 3; 2 2 3; 1 2 4]
    unsorted_indices = [1 2 3; 2 2 3; 4 2 1]
    zero_indexed_indices = [1 2 3; 2 2 3; 0 0 0]

    valid_values = [1.0,2.0,3.0]


    @testset "COOTEN Constructor" begin


        A = ssten.COOTen(valid_indices,rand(3),10)
        @test A.cubical_dimension == 10

         A = ssten.COOTen(valid_indices,rand(3))
        @test A.cubical_dimension == maximum(valid_indices)

        #all params
        @test_throws ErrorException ssten.COOTen(valid_indices, rand(2),10) #mis-matched dimensions
        @test_throws ErrorException ssten.COOTen(unsorted_indices, valid_values,10) #unsorted rows
        @test_throws ErrorException ssten.COOTen(zero_indexed_indices, valid_values,10) #zero indexed

        @test_throws ErrorException ssten.COOTen(zero_indexed_indices,valid_values,2)#invalid cubicaldim

        #without cubical dimension
        @test_throws ErrorException ssten.COOTen(valid_indices, rand(2)) #mis-matched dimensions
        @test_throws ErrorException ssten.COOTen(unsorted_indices, valid_values) #unsorted rows
        @test_throws ErrorException ssten.COOTen(zero_indexed_indices, valid_values) #zero indexed

        #without weights
        @test_throws ErrorException ssten.COOTen(unsorted_indices) #unsorted rows
        @test_throws ErrorException ssten.COOTen(zero_indexed_indices) #zero indexed

    end

    @test_throws ErrorException ssten.COOTen()

    @testset "COOTEN iterator" begin
        valid_indices = [1 2 3; 2 2 3; 1 2 4]
        valid_values = [1.0,2.0,3.0]

    end


end


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