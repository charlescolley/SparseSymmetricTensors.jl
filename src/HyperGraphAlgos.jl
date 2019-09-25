#=------------------------------------------------------------------------------
   Functions typically associated with hypergraph operations, functions placed
   here should relate to how to operate on the abstract representation of the
   tensor. Note that these may be common operations in multi-linear algebra too.

  Operations Glossary
  -------------------
    * permute_tensor - ()
    * connected_components - ()
    * find_edge_incidence - ()

------------------------------------------------------------------------------=#

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
* edge_incidence - (Dict{Int,Set{Tuple{Array{Int,1},Number}},1})

    The dictionary which links all vertices to the hyper edges they're contained
    within.
-----------------------------------------------------------------------------"""
function find_edge_incidence(A::SSSTensor)
  edge_incidence = Dict{Int,Set{Tuple{Array{Int,1},Number}}}()

  for (indices,val) in A.edges
    edge = (indices,val)

	prev_v = -1
    for v in indices
	  if prev_v == v
	    continue
	  else
  	    if !haskey(edge_incidence,v)
	      edge_incidence[v] = Set([edge])
	    else
	      push!(edge_incidence[v],edge)
	    end
	  end
	  prev_v = v
	end
  end

  return edge_incidence
end
