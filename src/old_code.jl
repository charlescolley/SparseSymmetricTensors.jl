#=------------------------------------------------------------------------------
  A storage file for old code that may be useful in the future.
------------------------------------------------------------------------------=#

#=-------------------------------Old Constructors-----------------------------=#
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
