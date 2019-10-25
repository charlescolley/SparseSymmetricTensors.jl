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
        A = ssten.SSSTensor([(indices[i,:],vals[i]) for i in 1:nnz])

        max_iter = 500
        x_0 = randn(n)

        #no shift test
        shift =0.0
        matHOPM_x, matHOPM_λ, matHOPM_iter =
           ssten.SSHOPM(indices,vals,n, x_0,shift,max_iter,tol)

        dictHOPM_x, dictHOPM_λ, dictHOPM_iter =
           ssten.SSHOPM(A, x_0,shift,max_iter,tol)
       println("og SSHOPM iterates : $(dictHOPM_iter)")
       println("new SSHOPM iterates : $(matHOPM_iter)")

       @test norm(matHOPM_x - dictHOPM_x) < tol
       @test abs(matHOPM_λ - dictHOPM_λ) < tol


       #postive shift test
       shift = rand()
        matHOPM_x, matHOPM_λ, matHOPM_iter =
           ssten.SSHOPM(indices,vals,n, x_0,shift,max_iter,tol)

        dictHOPM_x, dictHOPM_λ, dictHOPM_iter =
           ssten.SSHOPM(A, x_0,shift,max_iter,tol)
       println("og SSHOPM iterates : $(dictHOPM_iter)")
       println("new SSHOPM iterates : $(matHOPM_iter)")

       @test norm(matHOPM_x - dictHOPM_x) < tol
       @test abs(matHOPM_λ - dictHOPM_λ) < tol

       #negative shift test
       shift = -rand()
        matHOPM_x, matHOPM_λ, matHOPM_iter =
           ssten.SSHOPM(indices,vals,n, x_0,shift,max_iter,tol)

        dictHOPM_x, dictHOPM_λ, dictHOPM_iter =
           ssten.SSHOPM(A, x_0,shift,max_iter,tol)
       println("og SSHOPM iterates : $(dictHOPM_iter)")
       println("new SSHOPM iterates : $(matHOPM_iter)")

       @test norm(matHOPM_x - dictHOPM_x) < tol
       @test abs(matHOPM_λ - dictHOPM_λ) < tol
    end


end