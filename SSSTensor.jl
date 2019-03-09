#=------------------------------------------------------------------------------
------------------------------------------------------------------------------=#
#module SSST

using Base.Cartesian
using Printf
using SparseArrays
using DataStructures

import MatrixNetworks.triangles_iterator
import Combinatorics.permutations, Combinatorics.multinomial
import LinearAlgebra.eigen, LinearAlgebra.norm, LinearAlgebra.dot
import Arpack.eigs

#TODO: SSSTensor([i,j,k]) returns abnormal result, this should be altered to set
# default hyperedge weight to 1.


mutable struct SSSTensor
  edges::Dict{Array{Int,1},Number}
  cubical_dimension::Int

  SSSTensor(e,n) =
    SSSTensor_verifier(e,n) ? new(reduce_edges(e),n) : error("invalid indices")
    # Need to adjust the error message on this constructor

  function SSSTensor(e)
    n = SSSTensor_verifier(e)
    new(reduce_edges(e),n)
  end

  function SSSTensor(A::Array{N,k}) where {N <: Number,k}
    edge_dict, n = SSSTensor_from_Array(A)
	new(edge_dict,n)
  end

  function SSSTensor(T::triangles_iterator)
    edges = Array{Tuple{Array{Int64,1},Float64}}(undef,0)
    for t in T
      push!(edges,([t.v1,t.v2,t.v3],1.0))
    end
	SSSTensor(edges)
  end

  function SSSTensor(T::triangles_iterator,n)
    edges = Array{Tuple{Array{Int64,1},Float64}}(undef,0)
    for t in T
      push!(edges,([t.v1,t.v2,t.v3],1.0))
    end
	SSSTensor(edges,n)
  end

end


@generated function SSSTensor_from_Array(A::Array{N,k}) where {N<:Number,k,p}
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

#helper function which creates a new variable, bounds the last loop
function string_as_varname_function(s::AbstractString, v::Any)
   s = Symbol(s)
   @eval (($s) = ($v))
end

"""-----------------------------------------------------------------------------
    order(A)

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
function SSSTensor_verifier(edges::Union{Array{Tuple{Array{Int,1},N},1},Dict{Array{Int64,1},N}}
                            ,n::Int) where N <: Number

    max_index = SSSTensor_verifier(edges)

    return max_index <= n
end

#UNTESTED
"""-----------------------------------------------------------------------------
    SSSTensor_verifier(edges)

  This function takes in a list of edges and checks whether or not the edges are
appropriate for a super symmetric tensor. Finds the largest index and over all
the edges and sets that as the cubical dimension of the tensor. Used as a helper
function for the SSSTensor constructors.

Input:
------
* edges - (Dict{Array{Int64,1},Number}):
    An array which contains Tuples of index arrays and paired edge values
    associated. The indices must be sorted.

Output:
-------
* max_index - (Int):
    An integer indicating the maximum index, returns 0 if an edge is found not
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
function SSSTensor_verifier(edges::Array{Tuple{Array{Int,1},N},1}) where N <: Number
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

"""-----------------------------------------------------------------------------
    permute_tensor!(A,p)

  This function applies a permutation to the indices in the sparse tensor
defined by the array p. Iterates have the permutation applied and indices are
resorted.

Input:
------
* A - (SSSTensor)
    The tensor to apply the permutation to.
* p - (Array{Int,1})
    The permutation array, where p[i] = j implies that vertex j is mapped to
    index i.
Note:
-----
  Originally planned to be named permute!, but Base.permute! must be overloaded
-----------------------------------------------------------------------------"""
function permute_tensor!(A::SSSTensor,p::Array{Int,1})
  @assert length(p) == A.cubical_dimension
  @assert Set(1:length(p)) == Set(p)  #check for a proper permutation

  permuted_edges = Dict{Array{Int,1},Number}()
  for (indices,val) in A.edges
    permuted_edges[sort(map(i->p[i],indices))] = val
  end
  A.edges = permuted_edges
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
Output:
-------
* sub_tensor - (SSSTensor)
    The subtensor which c
-----------------------------------------------------------------------------"""
function get_sub_tensor(A::SSSTensor,indices::T) where T <: Union{Array{Int,1},Set{Int}}
  @assert 0 < length(indices) <= A.cubical_dimension
  @assert all(indices .> 0)

  if T == Array{Int,1}
    indices = Set(indices)
  end

  incident_edges = find_edge_incidence(A)

  sub_tensor_edges = Dict{Array{Int,1},Number}()

  for v_i in indices

  end
end

"""-----------------------------------------------------------------------------
    Dynamical_System_Solver(A,x0,h,tol,m=1,update=0)

  This function takes in a sparse symmetric tensor and computes the tensor
eigenvector by solving the dynamical system formed by contracting the tensor
A into a vector, and then computing the mth largest eigenvector of the matrix
Ax^{k-1}. With this we form the dynamical system
					dxdt = :Lambda_m(Ax^{k+1})
which we solve with a forward Euler scheme with a step size of h, starting at
the point x0. We solve this until the norm of dxdt reaches a specified
tolerance.

Input:
------
* A - (SSSTensor or Array{Number,k})
    The symmetric tensor to compute the eigenvector of. Functions are
    overloaded to handle appropriate type.
* x0 - (Array{Number,1})
    The initial starting point for solving the dynamical system.
* h - (Float64)
    The step size for running the forward Euler scheme.
* tol - (Float64)
    The tolerance to solve the dynamical system up to, stops when
    norm(dxdt) < tol.
* m - (Int)
    The eigenvector to find when computing the dynamical system. Default is the
    largest eigenvector of the matrix.
* update - (Int)
    Indicates how many steps the program should update the user by, default
    value is 0.

Output:
-------
* x - (Array{Number,1}
    The resulting eigenvector computed by the dynamical system method.
* lambda - (Number)
    The resulting eigenvalue computed by the dynamical system method.
-----------------------------------------------------------------------------"""
function Dynamical_System_Solver(A::SSSTensor,x0::Array{N,1},h::Float64,
                                 tol::Float64,m::Int64 = 1, update::Int = 0) where
								 N <: Number
                                 #start with strictly positive random vector
                                 #check the monotonic dereasing property
  n = A.cubical_dimension
  @assert m <= n
  @assert length(x0) == n
  k = order(A)
  x = copy(x0)/norm(x0)
  step = 1

  while true
    A_x_k_2 = contract(A,x,k-2)
    _,V,_ = eigs(A_x_k_2,nev=m) # check eigs
    dxdt = sign(V[1,m])*V[:,m] - x
	#x /= norm(x)

	if norm(dxdt) <= tol
      return x, x'*A_x_k_2*x
    else
      x += h*dxdt
    end

	if update > 0
	  if step % update == 0
	    z = A_x_k_2*x
		lambda = dot(x,z)
		residual = z - lambda*x
	    @printf("step %5d: norm(dxdt) = %.16f | lambda = %.16f | res norm = %.16f \n",
		       step, norm(dxdt),lambda, norm(residual))
	  end
	end
    step += 1
  end
end

#figure out if it's faster to do an internal check than two overloaded functions

function Dynamical_System_Solver(A::Array{N,k},x0::Array{N,1},h::Float64,
                                 tol::Float64,m::Int64 = 1) where {N <: Number,k}
  #k = length(size(A))
  n = size(A)[1]
  @assert m <= n
  @assert length(x0) == n
  x = copy(x0)

  while true
    _,V = eigen(contract(A,x,k-2))
    dxdt = sign(real(V[1,m]))*real(V[:,m]) - x
	x /= norm(x)

	if norm(dxdt) <= tol
      return x
    else
      x += h*dxdt
    end
  end
end

"""-----------------------------------------------------------------------------
    SSHOPM(A,x_0,shift,max_iter,tol)

  This function runs the shifts symmetric higher order power method for a super
symmetric tensor with the passed in shift, up to a tolerance or up until a
maximum iteration.

Input:
------
* A - (SSSTensor):
    An instance of the super symmetric tensor class.
* x_0 - (Array{Number,1}):
    An initial vector to start the algorithm with.
* shift - (Number)
    The shift for the algorithm, can be predetermined to ensure convergence of
    the method.
* max_iter - (Int)
    The maximum number of iterations to run the routine for, prints a warning if
    the method hasn't converged by then.
* tol - (Float)
    The tolerance in difference between subsequent approximate eigenvalues to
    solve the routine up to.

Output:
-------
* z - (Array{Number,1})
    The final vector produced by the SSHOPM routine.
* lambda_k - (Number)
    The final approximate eigenvalue at the last iteration.
-----------------------------------------------------------------------------"""
function SSHOPM(A::SSSTensor, x_0::Array{N,1},shift::N,max_iter,tol) where
                N <: Number
    @assert A.cubical_dimension == length(x_0)

    x = x_0/norm(x_0)
    iterations = 0
    lambda_k_1 = Inf

    while true

        if shift > 0
            z = contract_k_1(A,x) + shift*x
        elseif shift == 0
            z = contract_k_1(A,x)
        else
            z = -(contract_k_1(A,x) + shift*x)
        end

        lambda_k = x'*z

        #normalize
        z /= norm(z)

        iterations += 1
        @show z
        @printf("lambda_k = %f\n",lambda_k)
        @printf("lambda diff = %f\n",abs(lambda_k - lambda_k_1))
        if abs(lambda_k - lambda_k_1) < tol || iterations >= max_iter
            if iterations >= max_iter
                @warn("maximum iterations reached")
            end
            return z, lambda_k
        else
            lambda_k_1 = lambda_k
            x = z
        end
    end
end


"""-----------------------------------------------------------------------------
    find_shift_for_convergence(A,use_fro)

  This function takes in a tensor and computes a shift bound to ensure
convergence of the SSHOPM. If the contracted tensor is large, this can be
computed with the frobenius norm.

Input:
------
* A - (SSSTensor):
    An instance of a super symmetric tensor class.
* use_fro - (bool):
    Indicates whether or not to use the frobenius norm.

Output:
-------
* shift_bound - (Float)
    A float indicating the lower bound for which the method is guaranteed to
    converge.
-----------------------------------------------------------------------------"""
function find_shift_for_convergence(A::SSSTensor)
    error("unfinished")
    shift_bound = (order(A) -1)
    if use_fro
        shift_bound *= contract(A)
    end

end


"""-----------------------------------------------------------------------------
    connected_components(A)

  This function computes the connected components of the undirected hypergraph
associated with the tensor A. The algorithm runs a breadth first search until
all the vertices are found and returns an array with the component number for
each vertex, and an array which stores the visting order.

Input:
------
* A - (SSSTensor)
    The tensor to find the conncected components of.
* v0 - (Int)
    Optional starting vertex of bfs tree, default will be 1.
Output:
-------
* component_assignment - (Array{Int,1})
    The array storing the components each vertex belongs too.
* component_sizes - (Array{Int,1})
    An array which keeps track of the size of each component.
* visting_order - (Array{Int,1})
    The order the vertices were visited in, used for permuting into block
    diagonal structures.
-----------------------------------------------------------------------------"""
function connected_components(A::SSSTensor,v0::Int = 1)
  n = A.cubical_dimension

  #initialize variables
  component_count = 1
  visited_vertex_count = 1

  unvisited_vertices = Set(1:n)
  q = Queue{Int}()
  enqueue!(q,v0)
  delete!(unvisited_vertices,v0)

  visiting_order = zeros(n)
  component_assignment = zeros(n)
  component_sizes = [0]

  edge_incidence = find_edge_incidence(A)

  while length(q.store) != 0

	v = dequeue!(q)

	#record information about v
	visiting_order[v] = visited_vertex_count
    component_assignment[v] = component_count
	component_sizes[component_count] += 1
	visited_vertex_count += 1


	#add neighbors to queue
	for (indices,_) in get(edge_incidence,v,[])
	  for neighbor in unique(indices)
	    if neighbor in unvisited_vertices
		  enqueue!(q,neighbor)
		  delete!(unvisited_vertices,neighbor)
		end
	  end

	end

	#check for diconnected components
 	if (length(q.store) == 0) && (length(unvisited_vertices) > 0)
 	  #add random unvisited vertex
       enqueue!(q,pop!(unvisited_vertices))
 	  component_count += 1
	  push!(component_sizes,0)
     end
  end

  return component_assignment, component_sizes, visiting_order
end
"""-----------------------------------------------------------------------------
    find_edge_incidence(A)

  This function creates a dictionary linking each vertex to each of the hyper
edges associated with that edge.

 Input:
 ------
* A - (SSSTensor)
    The tensor to find the hyper edge association of.

 Output:
 -------
* edge_incidence - (Dict{Int,Array{Tuple{Array{Int,1},Number}},1})
    The dictionary which links all vertices to the hyper edges they're contained
    within.

 TODO:
 -----
 edge_indicence value type should be converted to a set for better usage.
-----------------------------------------------------------------------------"""
function find_edge_incidence(A::SSSTensor)
  edge_incidence = Dict{Int,Array{Tuple{Array{Int,1},Number},1}}()

  for (indices,val) in A.edges
    prev_v = -1
    for v in indices
	  if prev_v == v
	    continue
	  else
  	    if !haskey(edge_incidence,v)
	      edge_incidence[v] = [(indices,val)]
	    else
	      push!(edge_incidence[v],(indices,val))
	    end
	  end
	  prev_v = v
	end
  end

  return edge_incidence
end
#end #module end
