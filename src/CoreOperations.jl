#=------------------------------------------------------------------------------
   Core Operations associated with the COOTensor Class. This includes the
   getters/setters, addition, and contraction operations. More sophisticated
   numerical routines are saved for the NumericalRoutines.jl file.

  Operations Glossary
  -------------------
    * order - ()
    * add_edges! - (UNTESTED)
    * dense - (UNTESTED)
    * contract_edge_k_1 - ()
    * contract  - ()
    * dense_contract - ()
    * contract_k_1 - ()
    * contract_multi - ()
    * get_sub_tensor - ()
    * find_nnz - ()

------------------------------------------------------------------------------=#
#=------------------------------------------------------------------------------
						           File I/O
------------------------------------------------------------------------------=#
"""-----------------------------------------------------------------------------
    save(A, filepath)

  Saves the the file in the desired location as a ssten file. The ssten file
  format is a generalization of the smat format to the sparse symmetric tensor
  case.  The first line of the file is tab separated line indicated the order of
  the tensor, the cubical dimension, and the number of unique non-zeros in the
  symmetric tensor all stored in that order, as ints:

      (order::Int)\\t(cubical_dimension::Int)\\t(non-zero count)\\n  .

  The subsequent lines are tab separated lines which store the unique non-zeros
  which occur in the tensor. Each line contains the indices of the hyperedge in
  sorted order as ints, and the weight as a floating point number:

      '(v_{i_1}::Int)\\t...(v_{i_k}::Int)\\t(edgeweight::Float)\\n'  .

  If the filepath doesn't end in ".ssten", it will be appended to the filepath.

  Inputs:
  -------
  * A - (SSSTensor):
    Instance of the sparse symmetric tensor we wish to save.

  * filepath - (string):
    The location to save the file.
----------------------------------------------------------------------------"""
function save(A::SSSTensor, filepath::String)
    file = open(filepath, "w")

    header = "\t$(order(A))\t$(A.cubical_dimension)\t(length(A.edges))\n"
    write(file,header);

    for (edge,weight) in A.edges
		edge_string=""
        for v_i in edge
		   edge_string *= string(v_i,"\t")
        end
		edge_string *= string(weight,"\n")
		write(file,edge_string)
    end

	close(file)
end

"""-----------------------------------------------------------------------------
    load(filepath,NoChecks)

  Loads in a sparse symmetric tensor from a .ssten file. See save for expected
  .ssten file formatting specifications. The routine will check to see if the
  vertex indices are 0 indexed or not, and will increment each by 1 if it is to
  adhere to Julia's use of indexing by 1.

  Inputs:
  -------
  * filepath - (String):
    The .ssten file to load from. Throws an error if not .ssten formatting.
  * enforceFormatting - (Optional Bool):
    Sorts each of the hyperedge indices before storing them. Can be used as a
    quick fix for a improperly formatted file. Not recommended as it will add
    to memory usage and run time.

  Outputs:
  --------
  * A - (SSSTensor):
    The Sparse symmetric tensor stored in the file.
-----------------------------------------------------------------------------"""
function load(filepath::String,enforceFormatting::Bool=false)
	#check path validity
	@assert filepath[end-5:end] == ".ssten"

    open(filepath) do file
		#preallocate from the header line
		order, n, m =
			[parse(Int,elem) for elem in split(chomp(readline(file)),'\t')]

		hyperedges = Array{Tuple{Array{Int64,1},Float64},1}(undef, m)
		i = 1
		for line in eachline(file)
			entries = split(chomp(line),'\t')
			hyperedges[i] = ([parse(Int,elem) for elem in entries[1:end-1]],
			   				 parse(Float64,entries[end]))
			i += 1
		end

		#check for 0 indexing
		zero_indexed = false

		for (indices,_) in hyperedges
			for v_i in indices
				if v_i == 0
    				zero_indexed = true
    				break
				end
			end
		end

		if zero_indexed
			redo_indexing!(hyperedges)
		end

		return SSSTensor(hyperedges)
	end
end

#=------------------------------------------------------------------------------
						           Getters
------------------------------------------------------------------------------=#

"""-----------------------------------------------------------------------------
 	  This function returns the order of the tensor passed in.

    Input:
    ------
    * A -(SSSTensor):

      The tensor in question.

    Output:
    -------
    * order - (Int)

      The order of the tensor.
-----------------------------------------------------------------------------"""
function order(A::SSSTensor)

    for (indices,_) in A.edges
        return length(indices)
    end
end

"""-----------------------------------------------------------------------------
    find_nnz(A)

  Finds the non-zeros in a k-dimensional array and returns the list of the
indices associated along with a count of the non-zeros.
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

"""-----------------------------------------------------------------------------
    get_sub_tensor(A,indices)

  This function produced a subtensor which only containes the vertices specified
in the array indices. Note that there must be hyper edges shared between the
vertices in indices, otherwise the SSSTensor constructor will throw an error
when given an empty list of hyperedges.

Input:
------
* A - (SSSTensor)

    The tensor to produce a subtensor from.
* indices - (Array{Int,1} or Set{Int})

    The indices to build a subtensor from.
* remap - (optional bool)

    Indicates whether or not to remap the vertices from 1 to length(indices).
  Useful if the desired subtensor is to be used as a standalone tensor.
Output:
-------
* sub_tensor - (SSSTensor)

    The subtensor of A which only contains the hyperedges of A which all include
    each index in indices.
-----------------------------------------------------------------------------"""
function get_sub_tensor(A::SSSTensor,indices::T,
                        remap::Bool=false) where T <: Union{Array{Int,1},Set{Int}}
  @assert 0 < length(indices) <= A.cubical_dimension
  @assert all(indices .> 0)

  if T == Array{Int,1}
    indices = Set(indices)
  end

  incident_edges = find_edge_incidence(A)
  sub_tensor_edges = Array{Tuple{Array{Int,1},Number}}(undef,0)

  for v_i in indices
    for (V,val) in get(incident_edges,v_i,[])
	  edge = (V,val)
	  #only include edge if all indices in hyperedge are desired
	  if all(x -> x in indices,V)
	    push!(sub_tensor_edges,edge)
      end
	  #remove all other edges in incidence
	  for v_j in V
	    delete!(get(incident_edges,v_j,[]),edge)
	  end
	end
  end

  if remap
	remap_indices!(sub_tensor_edges)
  end

  SSSTensor(sub_tensor_edges)
end

#=------------------------------------------------------------------------------
						             Setters
------------------------------------------------------------------------------=#

#UNTESTED
"""-----------------------------------------------------------------------------
    add_edges!(A,edges)

  This function takes in a list of edges and adds them into the SSSTensor. If an
edge is already present in the tensor, and the value is added in at that index.

Input:
------
* A - (SSSTensor)

    The tensor to add hyper edges to.
* edges - (Array{Tuple{Array{Int,1},Float},1})

    An array of pairs which contain the indices in the first element, and the
    value in the second element. Note each edge's indices must be in range.
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

#=------------------------------------------------------------------------------
						   Data Representation Converters
------------------------------------------------------------------------------=#

#UNTESTED
"""-----------------------------------------------------------------------------
    dense(A)

  This function returns a dense representation of the SSSTensor passed in.

Inputs:
-------
* A - (SSSTensor)

    The sparse tensor to be converted.

Output:
-------
* B - (Array{Number,k})

    The corresponding dense tensor representation of A.
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

#=------------------------------------------------------------------------------
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
* condensed_dict - (Dict{Array{Int,1},Number}

    The hyper edges in the lower order tensor which are formed by contracting
    the vector along the hyperedge e.
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

Input:
------
* e -(Tuple(Array{Int,1},Number)):

    a list of sorted indices paired with an edge value. Note that the list of
    indices corresponds to multiple sets of indices because we consider all
    permutations.
* x -(Array{Number,1})

    The vector of the same dimenionality of the tensor, to contract with.

Output:
-------
* contraction_vals - (Array{Tuple{Array{Int,1},Number}})

    The hyper edges in the lower order tensor which are formed by contracting
    the vector along the hyperedge e.
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
    contract(A,x,m)

  This function contracts the tensor along m modes. Note that when the tensor is
dense this function uses Base.Cartesian, and thus in order to generate the loops
with a variable used a trick which instantiates empty arrays of length 0, and
passes them to another function which can pull the orders out to generate the
loops.

Input:
------
* A -(SSSTensor or Array{Number,k}):

    The tensor to contract.
* x - (Array{Number,1}):

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
function contract(A::SSSTensor, x::Array{N,1},m::Int) where {N <: Number}
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
function contract(A::Array{N,k}, x::Array{M,1},m::Int64) where {M <: Number,N <: Number,k}

    return dense_contract(A,x,zeros(Int,repeat([0],m)...),
	                      zeros(Int,repeat([0],k-m)...))
end

@generated function dense_contract(A::Array{N,k}, x::Array{M,1},
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
* x - (Array{Number,1}):

    A vector of numbers to contract with.
Outputs
-------
* y - (Array{Number,1}):

    The output vector of Ax^{k-1}.
-----------------------------------------------------------------------------"""
function contract_k_1(A::SSSTensor, x::Array{N,1}) where {N <: Number}
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

"""-----------------------------------------------------------------------------
    contract_multi(A,vs)

  This function computes the result of contracting the tensor A by the columns
of the array Vs.

-----------------------------------------------------------------------------"""
function contract_multi(A::SSSTensor, Vs::Array{N,2}) where N <: Number
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

function contract(A::SSSTensor,v::Array{N,1},u::Array{N,1}) where N <: Number
  return contract_multi(A,hcat(v,u))
end
