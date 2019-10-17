#=------------------------------------------------------------------------------
   Numerical Routines that are useful for tensors but are not a part of the
   core functionality. Though one could create a seperate package implementing
   these, it would be more effective to store common functions in this package
   that can be optimized for the COOTensor package specifically.

  Operations Glossary
  -------------------
    * Dynamical
------------------------------------------------------------------------------=#

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
* A - (SSSTensor or Array{Number,k > 2})

    The symmetric tensor to compute the eigenvector of. Dense tensors have
    limited support, primary support is for constructors to the sparse cases.

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
      return x, x'*A_x_k_2*x , step
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
* iterations - (Integer)

    The number of iterations the algorithm ran for.
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
#        @show z
#        @printf("lambda_k = %f\n",lambda_k)
#        @printf("lambda diff = %f\n",abs(lambda_k - lambda_k_1))
        if abs(lambda_k - lambda_k_1) < tol || iterations >= max_iter
            if iterations >= max_iter
                @warn("maximum iterations reached")
            end
            return z, lambda_k, iterations
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

function HOSVD(A::SSSTensor,k::Int)



end