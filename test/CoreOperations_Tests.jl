

@testset "CoreOperations Tests" begin

    @testset "Sub Tensor Tests" begin
        indices = [1 1 2 2; 1 2 2 2; 1 3 3 3;2 2 2 3] #fully connected
        vals = [1.0,2.0,3.0,4.0]

        DictA = SSSTensor([(indices[i,:],vals[i]) for i in 1:4],3)
        COOA  = COOTen(indices,vals,3)

        sub_DictA = get_sub_tensor(DictA,[1,2],false)
        sub_COOA  = get_sub_tensor(COOA,[1,2],false)

        @test sub_COOA == COOTen(indices[1:2,:],vals[1:2],2)
    end


    @testset "save/load tests" begin
        _, indices, vals = set_up(n,ord,nnz)

        A = COOTen(indices,vals,n)
        save(A, tempath)
        B = load(tempath;enforceFormatting = true,type = "COOTen")
        @test A == B

        A = SSSTensor([(indices[i,:],vals[i]) for i in 1:nnz],n)
        save(A, tempath)
        B = load(tempath;enforceFormatting = true,type ="DICTen")
        @test A == B
        #check for .ssten file extension functionality

        rm(tempath)
    end


    @testset  "Data Representation" begin

        @testset "flatten" begin
            x = rand(n)
            A = SSSTensor([([2,3,3],1.0),([1,2,4],-1.0),([3,4,5],1.0)])
            flattened_A = flatten(A)
            kron_x = reduce(kron,[x for _ in 1:(order(A)-1)])
            kron_mul = flattened_A*kron_x
            @test LinearAlgebra.norm(SparseSymmetricTensors.contract(A,x,2) -kron_mul)/LinearAlgebra.norm(kron_mul) < tol
        end

        @testset "dense" begin
            @test ones(2,2,2,2) == dense(SSSTensor(ones(2,2,2,2)))
        end
    end


end