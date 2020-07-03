


@testset "helper.jl tests" begin

@testset "multiplicity tests" begin
    @test SparseSymmetricTensors.multiplicity_factor([1,1,1,1,1]) == 1
    @test SparseSymmetricTensors.multiplicity_factor([1,2,1,1,1]) == 5
    @test SparseSymmetricTensors.multiplicity_factor([2,2,1,1,1]) == 10
end

@testset "remap_indices tests" begin
    x = [([2,3,4],1.0)]
    SparseSymmetricTensors.remap_indices!(x)
    @test x == [([1,2,3],1.0)]

    x = [([2,3,4],1.0),([3,4,5],-2.3)]
    SparseSymmetricTensors.remap_indices!(x)
    @test x == [([1,2,3],1.0),([2,3,4],-2.3)]
end

end