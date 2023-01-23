# using OrdinaryDiffEq, LinearAlgebra

const N = 512
const M= 1
const xyd_brusselator = range(0,stop=5,length=N)
limit(a, N) = a == N+1 ? 1 : a == 0 ? N : a
@inline function kernel_1D!(du, u, α, II, I, t)
    i, j = Tuple(I)
    ip1 = limit(i+1, N); im1 = limit(i-1, N)
    du[II[i,j]] = α/(dx^2)*(u[II[im1,j]] + u[II[ip1,j]] - 2u[II[i,j]])
  end
function brusselator_1d!(du, u, p, t)
    @inbounds begin
      α = p[1]
      II = LinearIndices((N, M))
      kernel_1D!.(Ref(du), Ref(u), α, Ref(II), CartesianIndices((N, M)), t)
      return nothing
    end
  end
p = Float32.((0.1, step(xyd_brusselator)))

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
SDEProblem(brusselator_1d, g, cu(u0), (0f0,110.5f0), p)
prob_ode_brusselator_1d_cuda = SDEProblem(brusselator_1d, g, cu(u0), (0f0,11.5f0), p)
@time solve(prob_ode_brusselator_1d_cuda,SKenCarp(),save_everystep=false)

testf(x,y...)=y[1]

testf(1.0,2.0,3.0)
