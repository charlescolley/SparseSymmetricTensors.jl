using Test
using MATLAB
using SSST
include("SSSTensor.jl")

"""------------------------------------#----------------------------------------
                               SSSTensor Test Suite"
-----------------------------------------------------------------------------"""
                                Constructor Tests
#------------------------------------------------------------------------------#
@testset "Contructor Tests" begin
    @test_throws Error

end
@test_throws

"""-----------------------------------------------------------------------------
                                Add Edges Tests
-----------------------------------------------------------------------------"""


"""-----------------------------------------------------------------------------
                             Vector Contraction Tests
-----------------------------------------------------------------------------"""


"""-----------------------------------------------------------------------------
                                Multiplicity Tests
-----------------------------------------------------------------------------"""
@testset "multiplicity_tests" begin
    @Test multiplicity_factor([1,1,1,1,1]) = 1
    @Test multiplicity_factor([1,2,1,1,1]) = 5
    @Test multiplicity_factor([2,2,1,1,1]) = 10
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