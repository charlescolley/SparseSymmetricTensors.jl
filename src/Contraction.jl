#=
Functions:
  contract_edge(e,x,k)
  contract_edge_k_1(e,x)
  contract(A,x,m)
  contract_k_1(A,x)
  contract_k_1!(A(type needs to be determined),x,k,y)
  contract_multi(A,vs)
  contract_edge_k_1!(indices,val,x,res)
#------------------------------------------------------------------------------
						           Contraction
------------------------------------------------------------------------------=#

"""-----------------------------------------------------------------------------
    contract_edge(e,x,k)

This function takes in an edge of a super symmetric tensor and computes the
resulting edges which result from contracting the the edge along k modes with
the vector x.

Input:
------
* e -(Tuple(Array{Int,1},Float)):

    A list of indices paired with an edge value. Note that the list of indices
    corresponds to multiple sets of indices because we consider all.
    permutations.
* x -(Array{Float,1})

    The vector to contract with.
* k - (Int):

    A positive integer which corresponds to the number of modes to contract
    along, must be greater than 0, and less than or equal to the cardinality
    of the edge.

Output:
-------
* condensed_dict - (Dict{Array{Int,1},AbstractFloat}

    The hyper edges in the lower order tensor which are formed by contracting
    the vector along the hyperedge e.
-----------------------------------------------------------------------------"""
function contract_edge(e::Tuple{Array{Int,1},M},x::Array{N,1},k::Int) where {N <: AbstractFloat, M<:AbstractFloat}
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

Input:
------
* e -(Tuple(Array{Int,1},AbstractFloat)):

    a list of sorted indices paired with an edge value. Note that the list of
    indices corresponds to multiple sets of indices because we consider all
    permutations.
* x -(Array{AbstractFloat,1})

    The vector of the same dimenionality of the tensor, to contract with.

Output:
-------
* contraction_vals - (Array{Tuple{Array{Int,1},AbstractFloat}})

    The hyper edges in the lower order tensor which are formed by contracting
    the vector along the hyperedge e.
-----------------------------------------------------------------------------"""
function contract_edge_k_1(e::Tuple{Array{Int,1},N},x::Array{N,1}) where N <: AbstractFloat
    (indices,val) = e
    order = length(indices)

    visited_sub_indices = Set{Array{Int,1}}()
    contraction_vals = Array{Tuple{Array{Int,1},N}}(undef,1)

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
    contract(A,x,m)

  This function contracts the tensor along m modes. Note that when the tensor is
dense this function uses Base.Cartesian, and thus in order to generate the loops
with a variable used a trick which instantiates empty arrays of length 0, and
passes them to another function which can pull the orders out to generate the
loops.

Input:
------
* A -(SSSTensor or Array{AbstractFloat,k}):

    The tensor to contract.
* x - (Array{AbstractFloat,1}):

    A vector of numbers to contract with.
* m - (Int)

    The number of modes to contract A with x along.

Output:
-------
* y - (SSSTensor or CSC Matrix or Array{Float64,k-m}):

    The output vector of Ax^m. THe output will be sparse if the input tensor is
    sparse, and dense otherwise. When the output is second order, and A is
    sparse, then the output will be a sparse matrix.
-----------------------------------------------------------------------------"""
function contract(A::SSSTensor, x::Array{N,1},m::Int) where {N <: AbstractFloat}
    @assert length(x) == A.cubical_dimension
    k = order(A)
    @assert 0 < m <= k

    new_edges = Dict{Array{Int,1},N}()
    #compute contractions
    for edge in A.edges
        new_e = contract_edge(Tuple(edge),x,m)
        reduce_dictionaries!(new_edges,new_e)
    end

    if k == m
        for (_,v) in new_edges
            return v
        end
    elseif k - m == 1
        y = zeros(length(x))
        for (e,v) in new_edges
            y[e[1]] = v
        end
        return y
    elseif k - m == 2
	  index = 0
	  nnzs = length(new_edges)
	  I = zeros(Int64,2*nnzs)
	  J = zeros(Int64,2*nnzs)
	  V = zeros(N,2*nnzs)

	  for (e,val) in new_edges
	     i,j = e
		 if i == j
		   index += 1
		   I[index] = i
		   J[index] = j
		   V[index] = val
	     else
		   index += 2
 	       I[index-1] = i
		   J[index-1] = j
		   I[index] = j
		   J[index] = i
		   V[index-1] = val
		   V[index] = val
		 end
	  end
	  return sparse(I[1:index],J[1:index],V[1:index],A.cubical_dimension,A.cubical_dimension)
	else
        return SSSTensor(new_edges,A.cubical_dimension)
    end
end

#Dense Case
function contract(A::Array{N,k}, x::Array{M,1},m::Int64) where {M <: AbstractFloat,N <: AbstractFloat,k}

    return dense_contract(A,x,zeros(Int,repeat([0],m)...),
	                      zeros(Int,repeat([0],k-m)...))
end

@generated function dense_contract(A::Array{N,k}, x::Array{M,1},
                                   B::Array{Int,m}, C::Array{Int,p}) where
								   {M<:AbstractFloat,N<:AbstractFloat,k,m,p}
    quote
        n = size(A)[1]
        @assert n == length(x)
        @assert $k >= m

        y = zeros(N,repeat([n],$k - $m)...)

        @nloops $k i A begin
            xs = prod(x[collect(@ntuple $m j-> i_{j+$p})])
            (@nref $p y i) += xs*(@nref $k A i)
        end
		if $k == m
	  		for val in y
	    		return val
	  		end
		else
          return y
        end
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

    The tensor to contract.
* x - (Array{AbstractFloat,1}):

    A vector of numbers to contract with.
Outputs
-------
* y - (Array{AbstractFloat,1}):

    The output vector of Ax^{k-1}.
-----------------------------------------------------------------------------"""
function contract_k_1(A::Ten, x::Array{N,1}) where {N <: AbstractFloat,Ten <: AbstractSSTen}
    @assert length(x) == A.cubical_dimension

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

function contract_k_1!(A::Ten, x::Array{N,1},y::Array{N,1}) where {N <: AbstractFloat,Ten <: AbstractSSTen}
    @assert length(x) == A.cubical_dimension

    new_edges = Array{Tuple{Array{Int,1},N}}(undef,0)
	@inbounds for i in 1:A.cubical_dimension
		y[i] = zero(N)
	end

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
    contract_multi(A,vs)

  This function computes the result of contracting the tensor A by the columns
of the array Vs.

-----------------------------------------------------------------------------"""
function contract_multi(A::SSSTensor, Vs::Array{N,2}) where N <: AbstractFloat
  k = order(A)
  n,m = size(Vs)
  @assert m <= k
  @assert n == A.cubical_dimension

  i = 1
  while true
    if k - i >= 2
      global A_sub = contract(A,Vs[:,i],1)
    elseif k - i == 1
	  global A_sub = A_sub*Vs[:,i]
    elseif k - i == 0
	  global A_sub = dot(A_sub,Vs[:,i])
    end
	i += 1

	if i > m #no more vectors
	  return A_sub
    end
  end
end

#TODO: make this take in an arbitrary number of vectors.
function contract(A::SSSTensor,v::Array{N,1},u::Array{N,1}) where N <: AbstractFloat
  return contract_multi(A,hcat(v,u))
end

"""-----------------------------------------------------------------------------
    contract_edge_k_1!(indices,val,x,res)

  This function computes the value of contracting one hyper edge with an input
vector x to all but 1 mode, and saves the result in the appropriate locations
in res.

Inputs:
-------
* indices - (Array{Int,2}):

    2D array with all the indices of the hyper edges in the tensor. Note that
    the number of rows is not necesarily equal to the non-zeros in the tensors.
    The indices of each row is assumed to be sorted in increasing order.
* nnz_val -  (N <: AbstractFloat):

    The non-zero associated with the hyper edges.
* x - (Array{N <: AbstractFloat, 1}):

    The vector to contract the hyper edge with.
* res_val - (Array{N <:AbstractFloat, 1}):

    The location to update the solution to. res is assumed to be initialized and
    so the values are simply added to the entries of y, this is done because
    this function is considered a helper function to the contract function o
    verloaded for the dense array representation of symmetric tensors.
-----------------------------------------------------------------------------"""
@inline function contract_edge_k_1!(edge::Tuple{Array{Int,1},N},
                                    x::Array{N,1},res::Array{N,1},
									sub_edge::Array{Int,1},
									multiplicities::Dict{Int,Int},
									final_counts::Array{Int,1},
									ord::Int)where {M,T, N <: AbstractFloat}
	(indices, nnz_val) = edge
    #ord = length(indices)
	prev_index = -1
	#sub_edge = Array{Int,1}(undef,ord - 1)

	for j = 1:ord
		#only compute contraction once per index
		if prev_index != indices[j]

			#form sub_edge
			i = 1
			for k in Base.Iterators.flatten((1:j-1,j+1:ord))
				sub_edge[i] = indices[k]
				i += 1
			end

			val = nnz_val * multiplicity_factor(sub_edge,multiplicities,final_counts)
			for k in sub_edge
				val *= x[k]
			end
			res[indices[j]] += val
		end
		prev_index = indices[j]
	end

end


"""-----------------------------------------------------------------------------
    contract_k_1(A,x,k)

  Contracts the vector x with the tensor to all but one mode

-----------------------------------------------------------------------------"""
function contract_k_1(A::COOTen,x::Array{N,1}) where {N <: AbstractFloat}
	@assert A.cubical_dimension == length(x)

	y = zeros(A.cubical_dimension) #resulting vector

	if A.order == 3
		#need a guarantee of unique indices in the hyper edges.
		tri_contract!(A,x,y)
	else
		#preallocate helper functions
		sub_edge = Array{Int,1}(undef,A.order - 1)
		multiplicities = Dict{Int,Int}()
		final_counts = zeros(Int,A.order)

		for edge in A
			contract_edge_k_1!(edge,x,y,sub_edge,multiplicities,final_counts,A.order)
		end

	end
	return y
end

"""-----------------------------------------------------------------------------
    contract_k_1(A,x,k)

  Contracts the vector x with the tensor to all but one mode

-----------------------------------------------------------------------------"""
function contract_k_1!(A::COOTen,x::Array{N,1},y::Array{N,1}) where {N <: AbstractFloat}
	@assert A.cubical_dimension == length(x)

	@inbounds for i in 1:A.cubical_dimension
		y[i] = zero(N)
	end

	if A.order == 3
		tri_contract!(A,x,y)
	else

		#preallocate helper functions
		sub_edge = Array{Int,1}(undef,A.order - 1)
		multiplicities = Dict{Int,Int}()
		final_counts = zeros(Int,A.order)

		for edge in A
			contract_edge_k_1!(edge,x,y,sub_edge,multiplicities,final_counts,A.order)
		end
	end

end

function tri_contract!(A::ssten.COOTen,x::Array{N,1},y::Array{N,1}) where {N <: AbstractFloat}

	@assert A.order == 3
	#y = zeros(Float64,A.cubical_dimension)

	@inbounds for i in 1:A.unique_nnz
		ci = A.indices[i,1]
        cj = A.indices[i,2]
        ck = A.indices[i,3]
		y[ci] += 2*x[cj]*x[ck]*A.vals[i]
        y[cj] += 2*x[ci]*x[ck]*A.vals[i]
        y[ck] += 2*x[cj]*x[ci]*A.vals[i]
	end
end
