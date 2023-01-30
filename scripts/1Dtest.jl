# using OrdinaryDiffEq, LinearAlgebra

const N = 512
const M= 1
const xyd_brusselator = range(0,stop=5,length=N)
limit(a, N) = a == N+1 ? 1 : a == 0 ? N : a

# function δUδσ3(σ::T; h::T= 0.0f0, σ0::T =0.1f0) where {T}
# 	return σ * (σ^2 - σ0^2) - h
# 	# return σ * σ0 - h
# end

function δUδσ(σ::T; h::T = 0.0f0, σ0::T = 0.1f0) where {T}
	return σ * (σ^2 - σ0^2) - h
	# return σ * σ0 - h
end

function δUδσ(σ; h= 0.0f0, σ0 = 0.1f0) where {T}
	return σ * (σ^2 - σ0^2) - h
	# return σ * σ0 - h
end


function kernel_1D2!(du, u, α,dx, II, I, t)
    i, j = Tuple(I)
    ip1 = limit(i+1, N); im1 = limit(i-1, N)
    du[II[i,j]] = α/(dx^2)*(u[II[im1,j]] + u[II[ip1,j]] - 2u[II[i,j]])+δUδσ(u[II[i,j]])
  end
function brusselator_1d!(du, u, p, t)
    @inbounds begin
      α,dx = p
      II = LinearIndices((N, M))
      kernel_1D2!.(Ref(du), Ref(u), α,dx, Ref(II), CartesianIndices((N, M)), t)
      return nothing
    end
  end
p = Float32.((0.5, step(xyd_brusselator)))

function init_brusselator_1d(xyd)
  N = length(xyd)
  u = zeros(N, M)
  for I in CartesianIndices((N, M))
    x = xyd[I[1]]
    y = xyd[I[2]]
    u[I] = randn()
  end
  u
end

function g(du, u, p, t)
        return du .= 0.01f0
end
u0 = init_brusselator_1d(xyd_brusselator)
prob_ode_brusselator_1d = ODEProblem(brusselator_1d,u0,(0.,11.5),p)
prob_ode_brusselator_1d_cuda = SDEProblem(brusselator_1d!, g, cu(u0), (0f0,11.5f0), p)
@time solve(prob_ode_brusselator_1d_cuda,SKenCarp(),save_everystep=false)

du0=similar(cu(u0))
brusselator_1d!(du0, cu(u0), p, 0f0)
