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

  permuted_edges = Dict{Array{Int,1},AbstractFloat}()
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
function connected_components(A::Ten,v0::Int = 1) where {Ten <: AbstractSSTen}
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
    get_largest_component(A,filepath)

  Computes the largest connected component (lcc) of the hypergraph. Will either
  compute it directly or load/save it it from/to an lcc file built from the
  filepath. The filepath's postfix is replaced from a .lcc of the formatting

  (largest_component_size::Int)\t(total vertex count::Int)\n
  (vertex_index_1::Int)\n
            ⋮
  (vertex_index_n::Int)\n

Inputs:
-------
* A - (SSSTensor)
  The hypergraph adjacency tensor to find the largest component of.
* filepath - (Optional String)
  The .ssten filepath associated with the tensor, if passed in, then .lcc files
  are created so that the largest connected component need not be computed
  multiple times.

Outputs:
--------
* subtensor - (SSSTensor)

  The subtensor associated with the largest component of the orignal hypergraph.
  Vertices are remapped so that they're indexed from 1:sizeof(lcc)
* lcc_indices - (Array{Int,1})

  Array with the indices of the vertices from the original graph which comprise
  the largest connected component.
-----------------------------------------------------------------------------"""
function get_largest_component(A::Ten,filepath::String="") where {Ten <: AbstractSSTen}

    if !isempty(filepath) #if path is passed in, look for lcc file
    	lcc_file = alterFilename(filepath,".lcc",keep_postfix=false)
    	if isfile(lcc_file)
    		open(lcc_file, "r") do f
				println("opened lcc file")
				lcc_size,full_size =
				    [parse(Int,elem) for elem in split(chomp(readline(f)),'\t')]
				if lcc_size == full_size
    				subtensor = A
					lcc_indices = 1:A.cubical_dimension
				else
					lcc_indices = Array{Int,1}(undef,lcc_size)
					for (i,line) in zip(1:lcc_size,eachline(f))
					  lcc_indices[i] = parse(Int,chomp(line))
					end
					subtensor = get_sub_tensor(A,lcc_indices,true)
				end
				return subtensor,lcc_indices
			end
		end
	end

	comps, comp_sizes, _ = connected_components(A)

	if length(comp_sizes) > 1
		largest_comp = findall(x->x==maximum(comp_sizes),comp_sizes)[1]
		lcc_indices = findall(x->x==largest_comp,comps)
	else
		#graph is connected, return self
		return A, 1:A.cubical_dimension
	end

	if !isempty(filepath) #save lcc file if one doesn't exist
		lcc_file = alterFilename(filepath,".lcc",keep_postfix=false)
		open(lcc_file,"w") do f
			header="$(length(lcc_indices))\t$(A.cubical_dimension)\n"
			write(f,header)
			if lcc_indices == A.cubical_dimension
				return
			end
			for v_i in lcc_indices
				write(f,"$(v_i)\n")
			end
		end
		println("made it")
	end
	get_sub_tensor(A,lcc_indices,true),lcc_indices
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
* edge_incidence - (Dict{Int,Set{Tuple{Array{Int,1},AbstractFloat}},1})

    The dictionary which links all vertices to the hyper edges they're contained
    within.

TODO: May be good to create a version where you can request the incidence for a
particular vertex
-----------------------------------------------------------------------------"""
function find_edge_incidence(A::Ten) where {Ten <: AbstractSSTen}
  edge_incidence = Dict{Int,Set{Tuple{Array{Int,1},AbstractFloat}}}()

  for (indices,val) in A

	prev_v = -1
    for v in indices
	  if prev_v == v
	    continue
	  else
  	    if !haskey(edge_incidence,v)
	      edge_incidence[v] = Set([(indices,val)])
	    else
	      push!(edge_incidence[v],(indices,val))
	    end
	  end
	  prev_v = v
	end
  end

  return edge_incidence
end
