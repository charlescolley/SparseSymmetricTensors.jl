#=------------------------------------------------------------------------------
------------------------------------------------------------------------------=#
module SSST
using Combinatorics
using Base.Cartesian
using Printf



mutable struct SSSTensor
    edges::Dict{Array{Int,1},Number}
    cubical_dimension::Int
    SSSTensor(e,n) =
       SSSTensor_verifier(e,n) ? new(reduce_edges(e),n) : error("invalid indices")
    function SSSTensor(e)
        indices_valid, n = SSSTensor_verifier(e)
        if indices_valid
            new(reduce_edges(e),n)
        else
            error("invalid indices")
        end
    end

    function SSSTensor(A::Array{N,k}) where {N <: Number,k}
        edge_dict, n = SSSTensor_from_Array(A,zeros(Int,repeat([0],k-1)...))
        new(edge_dict,n)
    end

end



@generated function SSSTensor_from_Array(A::Array{N,k},B::Array{Int,p}) where {N<:Number,k,p}
    quote

        n = size(A)[1]
    #    @assert all((x)->x == n, size(A)) #cubical tensor

        y = Dict{Array{Int64,1},N}()
        c = $k
        @show c
        string_as_varname_function(@sprintf("i_%d",c+1),n)
        @nloops $k i d-> 1:i_{d+1} begin
            indices = @ntuple $k i
            add_perm!(y,A,$k,indices)
        end
        return y,n
    end
end

"""
helper function for testing and adding in the permutation as an edge into the
dictionary used to build the symmetric tensor from the

"""
function add_perm!(edges,A,order,indices)
    val = A[indices...]
    sum = 0.0

    tol = 1e-12
    for p in permutations(indices)
        sum += A[p...]
        if abs(A[p...] - val) > tol
            error("tensor is not symmetric")
            #would be good to print the permutation which raises error
        end
    end
    #add in the average of the values of the permutations of the tensor
    if sum > 0.0
        edges[collect(indices)] = sum/multiplicity_factor(collect(indices))
    end
end

#helper function which creates a new variable, bounds the last loop
function string_as_varname_function(s::AbstractString, v::Any)
   s = Symbol(s)
   @eval (($s) = ($v))
end

"""-----------------------------------------------------------------------------
    order(A)

This function returns the order of the tensor passed in.

Input
-----
* A -(SSSTensor):
    the tensor in question

Outputs
-------
* order - (Int)
    the order of the tensor
-----------------------------------------------------------------------------"""
function order(A::SSSTensor)
    for (indices,_) in A.edges
        return length(indices)
    end
end


"""-----------------------------------------------------------------------------
    matrix_to_dictionary()

This function takes in a matrix containing the indices of hyperedges as columns
 and converts it into a dictionary linking the sorted indices to values. If no
 values are passed in, then the weights of each hyperedge are assumed to be 1.0.

Inputs
------
A - (k x n Array{Int,2}):
  each column corresponds to the indices of a hyper edges.

Outputs
-------
D - (Dictionary{Array{Int,1},Float64}):
  The dictionary used to create the super symmetric tensor.

Note
----
Add in weights to link to hyper edges.
-----------------------------------------------------------------------------"""
function matrix_to_dictionary(A::Array{Int,2})
    D = Dict{Array{Int,1},Float64}()
    _,n = size(A)

    for j in 1:n
        sorted_indices = sort(A[:,j])
        if !haskey(D,sorted_indices)
            D[sorted_indices] = 1.0
        end
    end

    return D
end

"""-----------------------------------------------------------------------------
    reduce_edges(edges)

This function takes in the edges to passed to create a new SSSTensor and reduces
the list into a dictionary where the edges that have the same edges are added
together.

Inputs:
-------
* edges -(Array{{Array{Int,1},Number},1}):
  2 Tuples which contain the sorted indices and an edge weight associated with
  it.

Outputs:
--------
* edge_dict - (Dict{Array{Int,1},Number}):
  The resulting dictionary which has the edges aggregated together.
-----------------------------------------------------------------------------"""
function reduce_edges(edges::Array{Tuple{Array{Int,1},N},1}) where N <: Number
    edge_dict = Dict()

    for (indices, weight) in edges
        if haskey(edge_dict,indices)
            edge_dict[indices] += weight
        else
            edge_dict[indices] = weight
        end
    end
    return edge_dict
end

function reduce_edges(edges::Dict{Array{Int,1},N}) where N <: Number
    return edges
end

function reduce_edges!(edge_dict::Dict{Array{Int,1},N},
                       edges::Array{Tuple{Array{Int,1},N},1}) where N <: Number
    for (indices, weight) in edges
        if haskey(edge_dict,indices)
            edge_dict[indices] += weight
        else
            edge_dict[indices] = weight
        end
    end
    return edge_dict
end


"""-----------------------------------------------------------------------------
    SSSTensor_verifier(edges,n)

This function takes a list of edges and a cubical dimension and checks whether
or not the edges are appropriate for a a super symmetric cubical tensor of n
dimensions.

Inputs:
-------
* edges - (Array{Tuple{Array{Int,1},Float}}):
  An array which contains Tuples of index arrays and Tupleed edge values
  associated. The indices must be sorted
* n  - (Int):
  An iteger indicating the desired dimension of the cubical tensor.

Outputs:
--------
is_valid - (Bool)
  An integer indicating whether or not the edges are appropriate for the tensor
  specified.
-----------------------------------------------------------------------------"""
function SSSTensor_verifier(edges::Union{Array{Tuple{Array{Int,1},N},1},Dict{Array{Int64,1},N}}
                            ,n::Int) where N <: Number

    indices_are_valid, max_index = SSSTensor_verifier(edges)

    @show indices_are_valid, max_index
    return indices_are_valid && max_index <= n
end

#UNTESTED
"""-----------------------------------------------------------------------------
    SSSTensor_verifier(edges)

This function takes in a list of edges and checks whether or not the edges are
appropriate for a super symmetric tensor. Finds the largest index and over all
the edges and sets that as the cubical dimension of the tensor. Used as a helper
 function for the SSSTensor constructors.

Inputs
------
* edges - (Dict{Array{Int64,1},Number}):
  An array which contains Tuples of index arrays and paired edge values
  associated. The indices must be sorted

Outputs
-------
* are_valid - (Bool):
    a bool indicating whether or not the edges all have positive indices and
    the indices are sorted, and have same number of indices.
* max_index - (Int):
    an integer indicating the maximum index, returns 0 if an edge is found not
    to be sorted.
-----------------------------------------------------------------------------"""
function SSSTensor_verifier(edges::Dict{Array{Int64,1},N}) where N <: Number
    max_index = -Inf
    order = -1
    for (indices,_) in edges
        if order == -1
            order = length(indices)
        else
            if length(indices) != order
                error("edge is wrong order")
            end
        end

        if !issorted(indices) || any(x -> x < 1,indices)
            return false, 0
        end
        if indices[end] > max_index
            max_index = indices[end]
        end
    end

    return true, max_index
end


"""-----------------------------------------------------------------------------
    SSSTensor_verifier(edges)

This function takes in a dictionary of edges and a cubical dimension and checks
whether or not the edges are appropriate for a super symmetric tensor with
dimension n. Used as a helper function for the SSSTensor constructors.

Inputs
------
* edges - (Array{Tuple{Array{Int,1},Float}}):
  An array which contains Tuples of index arrays and paired edge values
  associated. The indices must be sorted

Outputs
-------
* are_valid - (Bool):
    a bool indicating whether or not the edges all have positive indices and
    the indices are sorted, and have same number of indices.
* max_index - (Int):
    an integer indicating the maximum index, returns 0 if an edge is found not
    to be sorted.
-----------------------------------------------------------------------------"""
function SSSTensor_verifier(edges::Array{Tuple{Array{Int,1},N},1}) where N <: Number
    max_index = -Inf
    order = -1
    for (edge,_) in edges
        if order == -1
            order = length(edge)
        else
            if length(edge) != order
                error("edge is wrong order")
            end
        end

        if !issorted(edge) || any(x -> x < 1,edge)
            return false, 0
        end
        if edge[end] > max_index
            max_index = edge[end]
        end
    end

    return true, max_index
end

#UNTESTED
"""-----------------------------------------------------------------------------
    add_edges!(A,edges)

This function takes in a list of edges and adds them into the SSSTensor. If an
edge is already present in the tensor, and the value is added in at that index.

Inputs:
-------
* A - (SSSTensor)
* edges - (Array{Tuple{Array{Int,1},Float},1})
  an array of pairs which contain the indices in the first element, and the
  value in the second element. Note each edge's indices must be in range
-----------------------------------------------------------------------------"""
function add_edges!(A::SSSTensor,edges::Array{Tuple{Array{Int,1},N},1}) where N <: Number
    #check edges' validity
    for (indices,_) in edges
        sort!(indices)
        @assert indices[1] > 0
        @assert indices[end] <= A.cubical_dimension
    end

    for (indices,v) in edges
        if haskey(A.edges, indices)
            A.edges[indices] += v
        else
            A.edges[indices] = v
        end
    end
end

#UNTESTED
"""-----------------------------------------------------------------------------
    dense(A)

This function returns a dense representation of the SSSTensor passed in.
-----------------------------------------------------------------------------"""
function dense(A::SSSTensor)
    B = zeros(eltype(A.edges.vals),Tuple(repeat([A.cubical_dimension],order(A))))

    for (indices,_) in A.edges
        for p in permutations(indices)
            B[CartesianIndex(Tuple(p))] = A.edges[indices]
        end
    end

    return B
end

"""-----------------------------------------------------------------------------
    contract_edge(e,x,k)

This function takes in an edge of a super symmetric tensor and computes the
resulting edges which result from contracting the the edge along k modes with
the vector x.

Inputs:
-------
* e -(Tuple(Array{Int,1},Float)):
    a list of indices paired with an edge value. Note that the list of indices
    corresponds to multiple sets of indices because we consider all
    permutations.
* x -(Array{Float,1})
    The vector to contract with.
* k - (Int):
    a positive integer which corresponds to the number of modes to contract
    along, must be greater than 0, and less than or equal to the cardinality
    of the edge.
-----------------------------------------------------------------------------"""
function contract_edge(e::Tuple{Array{Int,1},M},x::Array{N,1},k::Int) where {N <: Number, M<:Number}
    order = length(e)

    (indices,val) = e
    condensed_dict = Dict{Array{Int,1},N}()
    visited_sub_indices = Dict{Array{Int,1},Dict{Array{Int,1},N}}()

    for i in 1:length(indices)
        sub_edge = deleteat!(copy(indices),i)
        if !haskey(visited_sub_indices,sub_edge)
            if k == 1
                condensed_dict[sub_edge] = val*x[indices[i]]
            else
                visited_sub_indices[sub_edge] = contract_edge((sub_edge,val*x[indices[i]]),x,k-1)
            end
        end
    end

    if k != 1
        for (_,sub_dict) in visited_sub_indices
            reduce_dictionaries!(condensed_dict,sub_dict)
        end
    end
    return condensed_dict
end


"""-----------------------------------------------------------------------------
    contract_edge_k_1(e,x)

This function takes in an edge of a super symmetric tensor and computes the
resulting edges which result from contracting the the edge along k-1 modes with
the vector x, where k is the order of the hyper edge.

Inputs:
-------
* e -(Tuple(Array{Int,1},Number)):
    a list of sorted indices paired with an edge value. Note that the list of
    indices corresponds to multiple sets of indices because we consider all
    permutations.
* x -(Array{Number,1})
    The vector of the same dimenionality of the tensor, to contract with.
-----------------------------------------------------------------------------"""
function contract_edge_k_1(e::Tuple{Array{Int,1},N},x::Array{N,1}) where N <: Number
    (indices,val) = e
    order = length(indices)

    visited_sub_indices = Set{Array{Int,1}}()
    contraction_vals = Array{Tuple{Array{Int,1},N}}(undef,0)

    for i in 1:order
        sub_edge = deleteat!(copy(indices),i)
        if !in(sub_edge,visited_sub_indices)#haskey(scaling_factors,sub_edge)
            scaling = multiplicity_factor(sub_edge)
            push!(visited_sub_indices,sub_edge)
            push!(contraction_vals,([indices[i]],scaling*val*prod(x[sub_edge])))
        end
    end
    return contraction_vals
end

"""-----------------------------------------------------------------------------
    reduce_dictionaries!(D1,D2)

This function takes in two dictionaries assumed to have the same value type and
reduces the contents of the second dicionary into the first. When two keys exist
in both dictionaries, their values are summed together.
-----------------------------------------------------------------------------"""
function reduce_dictionaries!(D1::Dict{Array{Int,1},N},
                              D2::Dict{Array{Int,1},N}) where N <: Number

    for (key,val) in D2
        if haskey(D1,key)
            D1[key] += val
        else
            D1[key] = val
        end
    end
end

"""-----------------------------------------------------------------------------
    contract(A,x,m)

This function contracts the tensor along the

Inputs
------
* A -(SSSTensor):
  the tensor to contract.
* x - (Array{Number,1}):
  a vector of numbers to contract with.
* k - (Int)
  the number of modes to contract A with x along.
Outputs
-------
* y - (Array{Number,1}):
  the output vector of Ax^k.
-----------------------------------------------------------------------------"""
function contract(A::SSSTensor, x::Array{N,1},k::Int) where {N <: Number}
    @assert length(x) == A.cubical_dimension
    order = SSST.order(A)
    @assert 0 < k <= order

    new_edges = Dict{Array{Int,1},N}()
    #compute contractions
    for edge in A.edges
        new_e = contract_edge(Tuple(edge),x,k)
        reduce_dictionaries!(new_edges,new_e)
    end

    if order == k
        for (_,v) in new_edges
            return v
        end
    elseif order - k == 1
        y = zeros(length(x))
        for (e,v) in new_edges
            y[e[1]] = v
        end
        return y
    else
        return SSSTensor(new_edges)
    end
end


"""-----------------------------------------------------------------------------
    contract_k_1(A,x)

This function contracts the tensor along k-1 modes to produce a vector. This
will produce the same result as contract(A,x,k-1), but runs in a much faster
time.

Inputs
------
* A -(SSSTensor):
  the tensor to contract.
* x - (Array{Number,1}):
  a vector of numbers to contract with.
Outputs
-------
* y - (Array{Number,1}):
  the output vector of Ax^{k-1}.
-----------------------------------------------------------------------------"""
function contract_k_1(A::SSSTensor, x::Array{N,1}) where {N <: Number}
    @assert length(x) == A.cubical_dimension
    order = SSST.order(A)

    new_edges = Array{Tuple{Array{Int,1},N}}(undef,0)
    y = zeros(A.cubical_dimension)

    #compute contractions
    for edge in A.edges
        contracted_edges = contract_edge_k_1(Tuple(edge),x)
        push!(new_edges,contracted_edges...)
    end
    #reduce edges and copy into new vector
    edge_dict = reduce_edges(new_edges)

    for (i,v) in edge_dict
        y[i[1]] = v
    end
    return y
end

"""-----------------------------------------------------------------------------
    multiplicity_factor(indices)

This function takes in a list of indices and returns the multinomial coefficient
computed by the frequency of the values in the indices. Works as a helper
function for computing the number of non-zeros the edge represents in the
vector contraction routines.

Inputs
------


* indices -(Array{Int,1}):
  the indices associated with the hyper edge

Output
------
* multinomial_factor - (Int)
   the number of non-zeros this edge represents in the original tensor.

Note
----
Change to include tuples too.
-----------------------------------------------------------------------------"""
function multiplicity_factor(indices::Array{Int,1})
    multiplicities = Dict()

    for index in indices

        if haskey(multiplicities,index)
            multiplicities[index] += 1
        else
            multiplicities[index] = 1
        end
    end

    #copy into format that can be passed to multinomial
    final_counts = zeros(Int,length(indices))
    i = 1
    for (_,val) in multiplicities
        final_counts[i] = Int(val)
        i += 1
    end

    return multinomial(final_counts...)
end


"""-----------------------------------------------------------------------------
    dense_contraction(A, x, m)

This function computes a m mode contraction for a dense kth order cubical
tensor representation, with a vector of the appropriate dimension. Note that
this function uses Base.Cartesian, and thus in order to generate the loops with
a variable used a trick which instantiates empty arrays of length 0, and passes
them to another function which can pull the orders out to generate the loops.

Inputs
------
* A - (Array{Float64,k}):
  a kth order cubical tensor stored as a multidimensional array.
* x - (Array{Float,1}):
  an array corresponding to the vector to contract A with.
* m - (Int)
  an integer indicating the number of modes to contract along

Outputs
-------
* y - (Array{Float64,k-m}):
  the result of the m mode contraction.
-----------------------------------------------------------------------------"""
function dense_contraction(A::Array{N,k}, x::Array{M,1},m::Int64) where {M <: Number,N <: Number,k}

    return dense_contraction(A,x,zeros(Int,repeat([0],m)...),
                             zeros(Int,repeat([0],k-m)...))
end

@generated function dense_contraction(A::Array{N,k}, x::Array{M,1},
                                      B::Array{Int,m}, C::Array{Int,p}) where
                                      {M<:Number,N<:Number,k,m,p}
    quote
        n = size(A)[1]
        @assert n == length(x)
        @assert $k >= m

        y = zeros(N,repeat([n],$k - $m)...)

        @nloops $k i A begin
            xs = prod(x[collect(@ntuple $m j-> i_{j+$p})])
            (@nref $p y i) += xs*(@nref $k A i)
        end
        return y
    end
end

"""-----------------------------------------------------------------------------
    find_nnz(A)

Finds the non-zeros in a k-dimensional array and returns the list of the indices
associated along with a count of the non-zeros.
-----------------------------------------------------------------------------"""
@generated function find_nnz(A::Array{N,k}) where {N<:Number,k}
    quote
        n = size(A)[1]
        y = Array{Tuple{Array{Int,1},N},1}(undef,n^k)
        nnz = 1

        @nloops $k i A begin
            val = @nref $k A i
            if val != 0.0
                y[nnz] = (collect(@ntuple $k i),val)
                nnz += 1
            end
        end
        nnz -= 1 #take care of overcount
        return y[1:nnz], nnz
    end
end



end #module end
