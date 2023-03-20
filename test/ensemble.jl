using CUDA,DifferentialEquations,LinearAlgebra


const D = 100.0
const N = 8
const Mx = Array(Tridiagonal([1.0 for i in 1:(N - 1)], [-2.0 for i in 1:N],
                             [1.0 for i in 1:(N - 1)]))
const My = copy(Mx)
Mx[2, 1] = 2.0
Mx[end - 1, end] = 2.0
My[1, 2] = 2.0
My[end, end - 1] = 2.0

u0=u0 = randn(N, N)

const gMx = CuArray(Float32.(Mx))
const gMy = CuArray(Float32.(My))
# const gα₁ = CuArray(Float32.(α₁))
gu0 = CuArray(Float32.(u0))

const gMyA = CuArray(zeros(Float32, N, N))
const AgMx = CuArray(zeros(Float32, N, N))
const gDA = CuArray(zeros(Float32, N, N))
function gf(du, u, p, t)
    A = @view u[:, :]
    dA = @view du[:, :]
    mul!(gMyA, gMy, A)
    mul!(AgMx, A, gMx)
    @. gDA = D * (gMyA + AgMx)
    @. dA = gDA
end

function g(du, u, p, t)
    du .= 0.1f0
end


prob2 = ODEProblem(gf, gu0, (0.0, 0.1))
CUDA.allowscalar(false) # makes sure none of the slow fallbacks are used
@time sol = solve(prob2, ROCK2(), progress = true, dt = 0.01, save_everystep = false,
                  save_start = false,abstol=1e0,reltol=1e0);


prob = SDEProblem(gf, g, gu0, (0.0, 0.1))
@time sol = solve(prob, SRIW1(),progress = true, dt = 0.01, save_everystep = false,
                  save_start = false,abstol=2e0,reltol=2e0);

GC.gc(true)
CUDA.reclaim()
