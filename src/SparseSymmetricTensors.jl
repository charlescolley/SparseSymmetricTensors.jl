module SparseSymmetricTensors
#=------------------------------------------------------------------------------
   Main File of the COOTensor class, Only type definitions and Constructor
   functions should be placed here.
------------------------------------------------------------------------------=#

using Base.Cartesian
using Printf
using SparseArrays
using DataStructures
#using StaticArrays

import Combinatorics.permutations, Combinatorics.multinomial
import LinearAlgebra.eigen, LinearAlgebra.norm, LinearAlgebra.dot
import Arpack.eigs, Arpack.svds
import Base: ==


export COOTen
export SSSTensor


#TODO: SSSTensor([i,j,k]) returns abnormal result, this should be altered to set
# default hyperedge weight to 1.

#TODO: should raise an error message for an empty iterator in the contstuctor
#TODO: possible to create a zero tensor from Dense Array, but not list constructor
#      need to standardize this.
#TODO: possible to condense constructor definitions with default params
#TODO: Remove triangles_iterator constructor


abstract type AbstractSSTen end

#TODO: Determine if we should make mutable in a later version
#TODO: Can make this parametric, but only supporting real floats currently.
struct COOTen <: AbstractSSTen

	cubical_dimension::Int
	order::Int
	unique_nnz::Int
	indices::Array{Int,2} #must be sorted lexigraphically for efficient Inner Products
	vals::Array{Float64,1}

	#edges::Vector{Tuple{Array{Int,1},Float64}}

	function COOTen(indices::Array{Int,2};nocheck::Bool=false)
		if nocheck
			unique_nnz, order = size(indices)
			return new(maximum(indices),order,unique_nnz,indices, ones(unique_nnz))
	    else
			cubical_dimension,unique_nnz,order,edges = COOTenVerifier(indices)
		    return new(cubical_dimension,order,unique_nnz,sort(indices,dims=1),ones(unique_nnz))
		end
	end

	function COOTen(indices::Array{Int,2},values::Array{N,1};
		            nocheck::Bool=false) where N <: AbstractFloat
		if nocheck
			unique_nnz, order = size(indices)
			return new(maximum(indices),order,unique_nnz,indices, values)
	    else
			cubical_dimension,unique_nnz,order,indices,values = COOTenVerifier(indices,values)
			perm = sort(1:unique_nnz, by=i->indices[i,:])
	   	    return new(cubical_dimension,order,unique_nnz,indices[perm,:],values[perm])
		end
	end

	function COOTen(indices::Array{Int,2},values::Array{N,1},n::Int;
		            nocheck::Bool=false) where N <: AbstractFloat
		unique_nnz, order = size(indices)
		if nocheck
			return new(n,order,unique_nnz,indices,values)
	    else
    		perm = sort(1:unique_nnz, by=i->indices[i,:])
			_,unique_nnz,order,indices,values = COOTenVerifier(indices,values,n)
			return new(n,order,unique_nnz,indices[perm,:],values[perm])
		end
	end

	COOTen() = error("empty hyper-edge list.")
end


#iterators for COOTen
Base.iterate(A::COOTen, state=1) =
    state > length(A) ? nothing : ((A.indices[state,:],A.vals[state]),state +1)
Base.length(A::COOTen) = A.unique_nnz


#TODO: rename to DICTen
mutable struct SSSTensor <: AbstractSSTen
    edges::Dict{Array{Int,1},AbstractFloat}
    cubical_dimension::Int

    function SSSTensor(e::Union{Array{Tuple{Array{Int,1},N},1},
							    Dict{Array{Int64,1},N}},
					   n::Int;nocheck::Bool=false) where {N <: AbstractFloat}
	     if nocheck
    		 new(reduce_edges(e),n)
		 else

			 if SSSTensor_verifier(e,n)
				 new(reduce_edges(e),n)
			 else
				 error("Cubical dimension $(n) too small.")
			 end
		 end
       # Need to adjust the error message on this constructor
	end

	#Edge List constructor
	function SSSTensor(e::Union{Array{Tuple{Array{Int,1},N},1},
							    Dict{Array{Int64,1},N}};
					   nocheck::Bool=false) where {N <: AbstractFloat}
		n = SSSTensor_verifier(e,nocheck)
		new(reduce_edges(e),n)
	end

	SSSTensor() = error("empty hyper-edge list.")

	#Dense tensor constructor
	function SSSTensor(A::Array{N,k},n::Int=typemax(Int)) where {N <: AbstractFloat,k}

		if any(size(A) .> n)
		    error("input Tensor size ",size(A),
			      " has dimension too large for input cubical dimension ",n)
		else
			edge_dict, n = SSSTensor_from_Array(A)
			new(edge_dict,n)
		end
	end

	#function SSSTensor(path_file::String)
	#TODO: load .ssten file
end

#write iterators for COOTen
Base.iterate(A::SSSTensor, state=1) = Base.iterate(A.edges,state)
Base.length(A::SSSTensor) = Base.length(A.edges)

# Include the files from the other functions
include("Helpers.jl") # makes helpers accessible to other files
include("CoreOperations.jl")
include("Contraction.jl")
include("HyperGraphAlgos.jl")
include("NumericalRoutines.jl")


@generated function SSSTensor_from_Array(A::Array{N,k}) where {N<:AbstractFloat,k,p}
    quote

        shape = size(A)
        n = shape[1]
        for i in 2:length(shape)
            if shape[i] != n
                error(string("input tensor of shape ",shape," is not cubical"))
            end
        end

        y = Dict{Array{Int64,1},N}()
        string_as_varname_function(@sprintf("i_%d",$k+1),n)
        @nloops $k i d-> 1:i_{d+1} begin
            indices = @ntuple $k i
            add_perm!(y,A,$k,indices)
        end
        return y,n
    end
end

#=------------------------------------------------------------------------------
						          Input Verifiers
--------------------------------------------------------------------------------
						          SSTen Verifier
------------------------------------------------------------------------------=#
"""-----------------------------------------------------------------------------
    SSSTensor_verifier(edges,n)

  This function takes a list of edges and a cubical dimension and checks whether
or not the edges are appropriate for a a super symmetric cubical tensor of n
dimensions.

Inputs:
-------
* edges - (Array{Tuple{Array{Int,1},Float}}):

  An array which contains Tuples of index arrays and Tupleed edge values
  associated. The indices must be sorted.
* n  - (Int):

  An iteger indicating the desired dimension of the cubical tensor.

Outputs:
--------
* is_valid - (Bool)

    An integer indicating whether or not the edges are appropriate for the
    tensor specified.
-----------------------------------------------------------------------------"""
function SSSTensor_verifier(edges::Union{Array{Tuple{Array{Int,1},N},1},
                                         Dict{Array{Int64,1},N}},
										 n::Int) where N <: AbstractFloat
    max_index = SSSTensor_verifier(edges,false)
    return max_index <= n
end

#UNTESTED
"""-----------------------------------------------------------------------------
    SSSTensor_verifier(edges,nocheck)

  This function takes in a list of edges and checks whether or not the edges are
appropriate for a super symmetric tensor. Finds the largest index and over all
the edges and sets that as the cubical dimension of the tensor. Used as a helper
function for the SSSTensor constructors.

Input:
------
* edges - (Dict{Array{Int64,1},AbstractFloat}):

    An array which contains Tuples of index arrays and paired edge values
    associated. The indices must be sorted.

* nocheck - (optional Bool):


Output:
-------
* max_index - (Int):

    An integer indicating the maximum index, returns 0 if an edge is found not
    to be sorted.
-----------------------------------------------------------------------------"""
function SSSTensor_verifier(edges::Dict{Array{Int64,1},N},
                            nocheck::Bool=false) where N <: AbstractFloat
    max_index = -Inf
    order = -1
    for (indices,_) in edges
        if order == -1
            order = length(indices)
        end

		if !nocheck  #assure input meets invariant standards
			if length(indices) != order
                error("hyperedge $(indices) must be order $(order)")
            end
			if !issorted(indices)
			  error("$(indices) must be sorted ascendingly")
			end
			if any(x -> x < 1,indices)
			  error("$(indices) has an index < 1")
			end
		end

		if indices[end] > max_index
			max_index = indices[end]
		end
    end

    return max_index
end
#=
"""-----------------------------------------------------------------------------
    SSSTensor_verifier(edges)

  This function takes in a dictionary of edges and a cubical dimension and checks
whether or not the edges are appropriate for a super symmetric tensor with
dimension n. Used as a helper function for the SSSTensor constructors.

Input:
------
* edges - (Array{Tuple{Array{Int,1},Float}}):

    An array which contains Tuples of index arrays and paired edge values
    associated. The indices must be sorted.

Output:
-------
* max_index - (Int):

    An integer indicating the maximum index, returns 0 if an edge is found not
    to be sorted.
-----------------------------------------------------------------------------"""
=#
function SSSTensor_verifier(edges::Array{Tuple{Array{Int,1},N},1},
                            nocheck::Bool=false) where N <: AbstractFloat
  max_index = -Inf
  order = -1
  for (indices,_) in edges
    if order == -1
	  order = length(indices)
    else
	  if length(indices) != order
	    error(string("hyperedge ",indices," must be order ",order))
	  end
    end

    if !issorted(indices)
	  error(string(indices, " must be sorted ascendingly"))
    end
    if any(x -> x < 1,indices)
	  error(string(indices," has an index < 1"))
    end
    if indices[end] > max_index
	  max_index = indices[end]
    end
  end

  return max_index
end

#=------------------------------------------------------------------------------
						          COOTen Verifiers
------------------------------------------------------------------------------=#


"""-----------------------------------------------------------------------------
    COOTen_verifier(indices, values)

  This function takes in a 2-D array and checks to see that the indices in each
row are sorted, that the sizes of the passed in values are correct and if the
invariants are preserved, returns the order, cubical_dimension, and unique
non-zero count back for the constructor.

Input:
------
* indices - (Array{Int64,2}):

    Array storing all the indices of the tensor to be formed. Rows must be
    sorted in increasing order.

* values - (optional Array{Float64,1}):

    Array storing the non-zero values of the tensor to be formed. Must have the
    same number of rows as indices as each row of each correspond to the same
    hyperedge. If none are passed in, one is constructed containing all ones.

* n - (optional Int)

    The desired cubical dimension, allows for the construction of tensors with
    zero subtensors. Throws an error if not the largest
Output:
-------

* cubical_dimension

  The dimension of each mode for the cubical tensor formed.

* unique_nnz

  The number of unique non-zeros in the tensor up to symmetry.

* order

  The number of modes of the tensor.

* indices - (Array{Int,2})

  A static matrix made from the indices passed in.

* values - (Array{AbstractFloat,1})

  A static vector made from the values passed in.
-----------------------------------------------------------------------------"""
function COOTenVerifier(indices::Array{Int,2},
    					values::Union{Array{Float64,1},Nothing}=nothing,
                        n::Union{Int,Nothing}=nothing)

    rows, order = size(indices)
	if values === nothing
		values = ones(rows)
	end

	if n === nothing
		n = Inf
	end
	unique_nnz = length(values)

	if rows != unique_nnz
		error("Indices and values have mis-matched sizes. indices has $(rows)",
		      " rows and values has $(unique_nnz) entries. These must be the ",
			  "same.")
	end


	cubical_dimension = 0

    for i in 1:rows
		#check for sorted rows
		if !issorted(indices[i,:])
			error("row $(i) has unsorted indices: $(indices[i,:]). Each row ",
			      "must be sorted in increasing order.")
		end

		#check for valid indices
		if indices[i,1] < 1
			error("row $(i) has index less than 1 in entry $(indices[i,:]), ",
			      "Julia is indexed by 1.")
		elseif indices[i,order] > cubical_dimension

			if cubical_dimension > n
				error("indices contain vertex index,$(indices[i,order]), ",
				      "greater than specified cubical dimension, n:$(n), ",
					  "specified.")
			end
			cubical_dimension = indices[i,order]
		end
	end

	return cubical_dimension,unique_nnz,order,indices, values
	#       [((indices[i,:]),values[i]) for i in 1:unique_nnz]

end

end #module end
