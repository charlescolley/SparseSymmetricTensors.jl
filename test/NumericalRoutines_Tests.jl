using Test
import LinearAlgebra.norm
include("../src/SSSTensor.jl")

 n = 3
 ord = 5
 tol = 1e-9
 nnz = 10

@testset "NumericalRoutines Tests" begin


    @testset "Low Mem SSHOPM Tests" begin
        #make tensors
        x = rand(n)
        indices = rand(1:n,nnz,ord)
        for i =1:nnz
            indices[i,:] = indices[i,sortperm(indices[i, :])]
        end
        vals = rand(nnz)
        dictA = ssten.SSSTensor([(indices[i,:],vals[i]) for i in 1:nnz])
        COOA = ssten.COOTen(indices,vals)

        max_iter = 500
        x_0 = randn(n)

        #no shift test
        shift =0.0
        COOHOPM_x, COOHOPM_λ, COOHOPM_iter =
          ssten.SSHOPM(COOA,x_0,shift,max_iter,tol)

        dictHOPM_x, dictHOPM_λ, dictHOPM_iter =
           ssten.SSHOPM(dictA, x_0,shift,max_iter,tol)

        @test dictHOPM_iter == COOHOPM_iter
        @test norm(COOHOPM_x - dictHOPM_x)/norm(COOHOPM_x) < tol
        @test abs(COOHOPM_λ - dictHOPM_λ) < tol


        #postive shift test
        shift = rand()

        COOHOPM_x, COOHOPM_λ, COOHOPM_iter =
          ssten.SSHOPM(COOA,x_0,shift,max_iter,tol)

        dictHOPM_x, dictHOPM_λ, dictHOPM_iter =
           ssten.SSHOPM(dictA, x_0,shift,max_iter,tol)

        @test dictHOPM_iter == COOHOPM_iter
        @test norm(COOHOPM_x - dictHOPM_x)/norm(COOHOPM_x) < tol
        @test abs(COOHOPM_λ - dictHOPM_λ) < tol

        #negative shift test
        shift = -rand()

        COOHOPM_x, COOHOPM_λ, COOHOPM_iter =
          ssten.SSHOPM(COOA,x_0,shift,max_iter,tol)

        dictHOPM_x, dictHOPM_λ, dictHOPM_iter =
           ssten.SSHOPM(dictA, x_0,shift,max_iter,tol)

        @test dictHOPM_iter == COOHOPM_iter
        @test norm(COOHOPM_x - dictHOPM_x)/norm(COOHOPM_x) < tol
        @test abs(COOHOPM_λ - dictHOPM_λ) < tol

    end


end