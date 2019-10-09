


@testtet "helper.jl tests" begin

@testset "multiplicity tests" begin
    @test multiplicity_factor([1,1,1,1,1]) == 1
    @test multiplicity_factor([1,2,1,1,1]) == 5
    @test multiplicity_factor([2,2,1,1,1]) == 10
end

@testset "remap_indices tests" begin
    @test remap_indices!([([2,3,4],1.0)]) == [([1,2,3],1.0)]
    @test remap_indices!([([2,3,4],1.0),([3,4,5],-2.3)]) = [([1,2,3],1.0),([2,3,4],-2.3)]
end

end