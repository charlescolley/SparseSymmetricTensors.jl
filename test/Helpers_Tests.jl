


@testtet "helper_tests" begin

@testset "multiplicity_tests" begin
    @test multiplicity_factor([1,1,1,1,1]) == 1
    @test multiplicity_factor([1,2,1,1,1]) == 5
    @test multiplicity_factor([2,2,1,1,1]) == 10
end

end