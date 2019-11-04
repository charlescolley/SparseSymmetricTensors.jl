using BenchmarkTools
using Dates
using JSON
include("../src/SSSTensor.jl")

log_root = "./logs/"
log_folder = log_root *"Contraction-logs/"

if !isdir(log_root)
    mkdir(log_root)
end
if !isdir(log_folder)
    mkdir(log_folder)
end

 scales = 5
 n_0 = 100000
 ord = 3
 tol = 1e-9
 sparsity = 10

root_name = log_folder*
  "Contraction-profiling_$(Dates.month(now()))-$(Dates.day(now()))-"*
   "$(Dates.year(now()))_$(Dates.hour(now())):$(Dates.minute(now()))"


#initialize dictionaries
Trials = BenchmarkGroup()


if false  #fulll contraction tests

Trials["matSST_contract_k_1"] = BenchmarkGroup()
Trials["dictSST_contract_k_1"] = BenchmarkGroup()
Trials["dictSST_contract"] = BenchmarkGroup()

for i = 1:scales
    n = n_0^i

    nnz = Int(floor(sparsity*n))
    m = ord -1

    x = rand(n)
    y = Array{Float64,1}(undef,n)

    indices = rand(1:n,nnz,ord)
    for i =1:nnz
        indices[i,:] = indices[i,sortperm(indices[i, :])]
    end
    vals = rand(nnz)

    A = ssten.SSSTensor([(indices[i,:],vals[i]) for i in 1:nnz],n)

#    Trials["matSST_contract_k_1"]["n:$(n)"]  = @benchmark ssten.contract_k_1($indices,$vals,$n, $x)
    Trials["dictSST_contract_k_1"]["n:$(n)"] =  @benchmark ssten.contract_k_1($A, $x)
#    Trials["dictSST_contract"]["n:$(n)"] =      @benchmark ssten.contract($A, $x, $m)

    for k in keys(Trials)
        println("Results for Trial: $(k) for n = $(n)")
        display(Trials[k]["n:$(n)"])
        print("\n")
    end

end

BenchmarkTools.save(root_name*".json", Trials)
end



if true  #edge wise contraction operations
ord_0 = 3

Trials["row_edge_contract_k_1"] = BenchmarkGroup()
Trials["tuple_contract_k_1"] = BenchmarkGroup()
Trials["tuple_contract"] = BenchmarkGroup()

n = n_0

for i = 1:scales


    ord = ord_0^i

    m = ord -1

    x = rand(n)
    y = zeros(n)

    indices = rand(1:n,ord)
    sort!(indices)
    val = rand()

    e = (indices,val)

    Trials["row_edge_contract_k_1"]["ord:$(ord)"]  = @benchmark ssten.contract_edge_k_1!($indices,$val,$x,$y)
    Trials["tuple_contract_k_1"]["ord:$(ord)"] =  @benchmark ssten.contract_edge_k_1($e,$x)
    Trials["tuple_contract"]["ord:$(ord)"] =      @benchmark ssten.contract_edge($e,$x,$m)

    for k in keys(Trials)
        println("Results for Trial: $(k) for ord = $(ord)")
        display(Trials[k]["ord:$(ord)"])
        print("\n")
    end

end

end

# unprofiled
#contract(A,x,m)
#contract(A::Array{N,k}, x::Array{M,1},m::Int64)
#contract_k_1(A::SSSTensor, x::Array{N,1})
#contract_multi(A::SSSTensor, Vs::Array{N,2})
#contract(A::SSSTensor,v::Array{N,1},u::Array{N,1})
#contract_k_1!(indices::Array{Int,2},nnz::Array{N,1}, x::Array{N,1},y::Array{N,1})
