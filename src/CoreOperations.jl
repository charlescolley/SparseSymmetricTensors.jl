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

    header = "$(order(A))\t$(A.cubical_dimension)\t$(length(A.edges))\n"
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

		i = 0
		for line in eachline(file)
			i += 1
			entries = split(chomp(line),'\t')

			hyperedges[i] = ([parse(Int,elem) for elem in entries[1:end-1]],
			                 parse(Float64,entries[end]))
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


"""-----------------------------------------------------------------------------
    lowMemLoad(filepath)

  Loads in the components of a sparse symmetric tensor from a .ssten file.
  Useful when memory contraints restrict using dictionaries for the core data
  structure for the hyperedges. The routine will check to see if the vertex
  indices are 0 indexed or not, and will increment each by 1 if it is to
  adhere to Julia's use of indexing by 1.

  Inputs:
  -------
  * filepath - (String):
    The .ssten file to load from. Throws an error if not .ssten formatting.

  Outputs:
  --------
  * indices - (Array{Int,2}):
    The indices of the hyper edges, rows are expected to be sorted in ascending
    order from the ssten format.
  * vals - (Array{Float64,1}):
    The values for each hyperedge, the ith row of vals corresponds to the ith
    row of indices.
  * n - (Int):
    The cubical dimension of the tensor, found in the .ssten file.

  Notes:
  ------
  * See save for expected .ssten file formatting specifications.
-----------------------------------------------------------------------------"""
function lowMemLoad(filepath::String)

	#check path validity
	@assert filepath[end-5:end] == ".ssten"

    open(filepath) do file
		#preallocate from the header line
		order, n, m =
			[parse(Int,elem) for elem in split(chomp(readline(file)),'\t')]

		indices = Array{Int,2}(undef,m,order)
		vals = Array{Float64,1}(undef,m)

		i = 0
		for line in eachline(file)
			i += 1
			entries = split(chomp(line),'\t')

			indices[i,:] = [parse(Int,elem) for elem in entries[1:end-1]]
			vals[i] = parse(Float64,entries[end])

		end

		#check for 0 indexing
		zero_indexed = false

		for row = 1:i
			for col=1:order
				if indices[row,col] == 0
    				zero_indexed = true
    				break
				end
			end
		end

		indices = indices[1:i,:] #truncate array

		if zero_indexed
			n += 1
			indices += ones(Int,i,order)
		end

		#TODO: There's a problem with this routine, or perhaps the saving routine.

		return indices, vals, maximum(indices)
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
    get_sub_tensor(A,indices,remap=false)

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
function get_sub_tensor(A::SSSTensor,indices::T;
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

#UNTESTED
"""-----------------------------------------------------------------------------
    flatten(A,colIndex)

  This function returns a sparse matrix from flattening k-2 modes into the
  columns of the matrix.  Defaults to col major formatting.

  TODO: add colIndex function to parameters.

Inputs:
-------
* A - (SSSTensor)

    The sparse tensor to be converted.
*
Output:
-------
* B - (Array{Number,k})

    The corresponding dense tensor representation of A.
-----------------------------------------------------------------------------"""
function flatten(A::SSSTensor)

	ord = order(A)
	nnz = length(A.edges)*factorial(ord)
	I = Array{Int,1}(undef,nnz)
	J = Array{Int,1}(undef,nnz)
	V = Array{Float64,1}(undef,nnz)

	i = 0
	for (edge,val) in A.edges

		for indices in unique(permutations(edge))
			i+=1
			I[i] = indices[1]
			V[i] = val

			#adjust back to 0 indexing
			J[i] = foldl((x,y)-> x+A.cubical_dimension*(y-1),indices[2:end])

		end
	end

	return SparseArrays.sparse(I[1:i],J[1:i],V[1:i],A.cubical_dimension,A.cubical_dimension^(ord-1))
end