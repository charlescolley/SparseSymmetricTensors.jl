#=------------------------------------------------------------------------------
   Helper Functions used throughout the package. This is a good place for code
   that shouldn't be exposed to the user.

  Operations Glossary
  -------------------
    * reduce_dictionaries - ()
    * multiplicity_factor - ()
    * add_perm! - ()
    * string_as_varname_function - ()
    * matrix_to_dictionary - ()
    * reduce_edges - ()

------------------------------------------------------------------------------=#


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
    multiplicity_factor(indices)

  This function takes in a list of indices and returns the multinomial
coefficient computed by the frequency of the values in the indices. Works as a
helper function for computing the number of non-zeros the edge represents in the
vector contraction routines.

Input:
------
* indices -(Array{Int,1}):

  The indices associated with the hyper edge.

Output:
-------
* multinomial_factor - (Int)

   The number of non-zeros this edge represents in the original tensor.

Note
----

TODO: Change to include tuples too.
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

"""
helper function for testing and adding in the permutation as an edge into the
dictionary used to build the symmetric tensor from the

"""
function add_perm!(edges,A,order,indices)
    val = A[indices...]
    sum = 0.0

    tol = 1e-12
    for p in unique(permutations(indices))
        sum += A[p...]
        if abs(A[p...] - val) > tol
            error(string("tensor is not symmetric: index ",p," has different values"))
        end
    end
    #add in the average of the values of the permutations of the tensor
    if sum > 0.0
        edges[collect(indices)] = sum/multiplicity_factor(collect(indices))
    end
end

#creates a new variable, bounds the last loop, used in meta code generators
function string_as_varname_function(s::AbstractString, v::Any)
   s = Symbol(s)
   @eval (($s) = ($v))
end


"""-----------------------------------------------------------------------------
    matrix_to_dictionary()

  This function takes in a matrix containing the indices of hyperedges as
columns and converts it into a dictionary linking the sorted indices to values.
If no values are passed in, then the weights of each hyperedge are assumed to be
1.0.

Inputs:
-------
* A - (k x n Array{Int,2}):

    Each column corresponds to the indices of a hyper edges.

Output:
-------
* D - (Dictionary{Array{Int,1},Float64}):

    The dictionary used to create the super symmetric tensor.

Note:
-----

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

Input:
------
* edges -(Array{{Array{Int,1},Number},1}):

    2 Tuples which contain the sorted indices and an edge weight associated with
    it.

Output:
-------
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