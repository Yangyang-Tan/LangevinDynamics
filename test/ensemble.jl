using CUDA, DifferentialEquations, LinearAlgebra


const D = 100.0
const N = 8
const Mx =
    Array(Tridiagonal([1.0 for i = 1:(N-1)], [-2.0 for i = 1:N], [1.0 for i = 1:(N-1)]))
const My = copy(Mx)
Mx[2, 1] = 2.0
Mx[end-1, end] = 2.0
My[1, 2] = 2.0
My[end, end-1] = 2.0

u0 = u0 = randn(N, N)

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
@time sol = solve(
    prob2,
    ROCK2(),
    progress = true,
    dt = 0.01,
    save_everystep = false,
    save_start = false,
    abstol = 1e0,
    reltol = 1e0,
);


prob = SDEProblem(gf, g, gu0, (0.0, 0.1))
@time sol = solve(
    prob,
    SRIW1(),
    progress = true,
    dt = 0.01,
    save_everystep = false,
    save_start = false,
    abstol = 2e0,
    reltol = 2e0,
);

GC.gc(true)
CUDA.reclaim()



"""
    u, uprev, unew = wavepropagate!(u, cΔtΔx⁻¹=0.9/√N, uprev=copy(u), unew=copy(u))

Time-evolve `u` according to `timesteps` timesteps of the equation ∂²u/∂t² = c²∇²u,
using centered-difference approximations in space and time with discretization
size Δx and Δt, respectively, given the ratio cΔtΔx⁻¹(which must be < 1/√N
for CFL stability).

Dirichlet boundary conditions are used, based on the initial boundary values in `u`.
The `uprev` array should contain `u` at the previous timestep, defaulting to `copy(u)`
(corresponding to initial conditions at rest).   The `unew` array is used for storing
the subsequent timestep and should be of the same size and type as `u`.
"""
function wavepropagate!(
    u::AbstractArray{T,N},
    timesteps::Integer,
    cΔtΔx⁻¹::Real = 0.9 / √N,
    uprev::AbstractArray{T,N} = copy(u),
    unew::AbstractArray{T,N} = copy(u),
) where {T<:Number,N}
    cΔtΔx⁻¹ < 1 / √N ||
        throw(ArgumentError("cΔtΔx⁻¹ = $cΔtΔx⁻¹ violates CFL stability condition"))
    c²Δt²Δx⁻² = (cΔtΔx⁻¹)^2
    # construct tuple of unit vectors along N directions, for constructing nearest neighbors
    # ... using ntuple with Val(N) makes this type-stable with hard-coded loops
    unitvecs = ntuple(i -> CartesianIndex(ntuple(==(i), Val(N))), Val(N))

    I = CartesianIndices(u)
    Ifirst, Ilast = first(I), last(I)
    I1 = oneunit(Ifirst)

    for t = 1:timesteps

        # use the "ghost cell" technique for boundary conditions:
        # only timestep in the interior, and update boundaries separately (if at all)
        @inbounds for i = Ifirst+I1:Ilast-I1
            # compute a discrete (center-difference) Laplacian of u
            ∇²u = -2N * u[i]
            for uvec in unitvecs
                ∇²u += u[i+uvec] + u[i-uvec]
            end

            # update u via center-difference approximation of ∂²u/∂t² = c²∇²u
            unew[i] = 2u[i] - uprev[i] + c²Δt²Δx⁻² * ∇²u
        end

        # here, you would update the boundary "pixels" of xnew based on whatever
        # boundary condition you want.   not updating them, as we do here,
        # corresponds to Dirichlet boundary conditions

        u, uprev, unew = unew, u, uprev # cycle the arrays
    end

    return u, uprev, unew
end



using StochasticDiffEq, DiffEqNoiseProcess, Test, DiffEqDevTools
using LinearAlgebra
u0 = CUDA.ones(Float32, 32, 32, 32, 5*10^0)
v0 = CUDA.ones(Float32,32, 32, 32, 5*10^0)
u02 = CUDA.ones(Float32, 32, 32, 24, 2^15)
v02 = CUDA.ones(Float32, 32, 32, 32, 2^15)
sizeof(u02)/1024^3
supertype(supertype(typeof(u0)))

isa(u0,AbstractArray{Float32,})

prob.p
isa(f1_harmonic_iip,Function)
function f1_harmonic_iip(dP, P, Q, p, t)
   @. dP = -2 * sin(Q)
end
prob
f2_harmonic_iip(du, v, u, p, t) = du .= v
g_iip(du, u, p, t) = du .= 2.0f0
prob = DynamicalSDEProblem(f1_harmonic_iip, f2_harmonic_iip, g_iip, v0, u0, (0.0f0, 50.0f0);)
prob2 = DynamicalSDEProblem(f1_harmonic_iip, f2_harmonic_iip, g_iip, v02, u02, (0.0f0, 50.0f0);)
prob2 =
    SimpleSDEProblem(f1_harmonic_iip, v02, u02, (0.0f0, 50.0f0);)


prob2.u0.x[1]
@time sol1 = solve(
    prob,
    BAOAB(gamma = 1.0f0);
    dt = 0.1f0,
    save_everystep = false,
    save_start = false,
    save_end = true,
    dense = false,
)

using AverageShiftedHistograms
asdata = ash(vec(Array(sol1.u[end].x[2])), m = 8, rng = -60:0.002:60)
asdata2 = ash(vec(Array(u02)), m = 8, rng = -60:0.002:60)

plot(asdata; hist = false)
plot!(asdata2; hist = false)
sol1.u[end].x[2]
@time sol2 = solve(
    prob2,
    SimpleBAOABGPU(eta = 1.0f0,noise=2f0);
    dt = 0.1f0,
)


CUDA.@time f1_harmonic_iip(u02, v02, v02, 1f0, 1f0)
sol1 = 0
sol2=0
u02=0;
v02=0;
u0_1=0;
v0_1=0;
u0=0;
v0=0;
prob=0;
prob2=0;
a=0
@test sol1[:] ≈ sol2[:]
GC.gc(true)
CUDA.reclaim()
sol2=0
CUDA.@time randn4!(u02)
u02[1,1,1]
using CUDA
1
CUDA.versioninfo()
a = CUDA.zeros(2^10, 2^10)
CUDA.@time CUDA.randn!(a)
CUDA.versioninfo()
function Random.randn2!(rng::RNG, A::AnyGPUArray{T}) where {T<:Number}
    threads = (length(A) - 1) ÷ 2 + 1
    length(A) == 0 && return
    gpu_call(A, rng.state; elements = threads) do ctx, a, randstates
        idx = 2 * (linear_index(ctx) - 1) + 1
        U1 = gpu_rand(T, ctx, randstates)
        U2 = gpu_rand(T, ctx, randstates)
        Z0 = sqrt(T(-2.0) * log(U1)) * cos(T(2pi) * U2)
        Z1 = sqrt(T(-2.0) * log(U1)) * sin(T(2pi) * U2)
        @inbounds a[idx] = Z0
        idx + 1 > length(a) && return
        @inbounds a[idx+1] = Z1
        return
    end
    A
end



curand
import DifferentialEquations.OrdinaryDiffEq:
    @muladd,
    @unpack,
    @cache,
    @..

struct SimpleEMGPU <: DiffEqBase.AbstractSDEAlgorithm end
@muladd function DiffEqBase.solve(
    prob::SDEProblem{uType,tType,true},
    alg::SimpleEMGPU,
    args...;
    dt = error("dt required for SimpleEM"),
) where {uType,tType}
    f = prob.f
    g = prob.g
    u0 =prob.u0
    tspan = prob.tspan
    p = prob.p
    ftmp = zero(u0)
    gtmp = zero(u0)
    # dW = zero(u0)
    @inbounds begin
        n = Int((tspan[2] - tspan[1]) / dt) + 1
        t = [tspan[1] + i * dt for i = 0:(n-1)]
        sqdt = sqrt(dt)
    end

    @inbounds for i = 2:n
        tprev = t[i-1]
        f(ftmp, u0, p, tprev)
        CUDA.randn!(gtmp)
        g(gtmp, u0, p, tprev)
        @. gtmp = sqdt * gtmp
        @. u0 = u0 + ftmp * dt + gtmp
    end
    return u0
    # sol = DiffEqBase.build_solution(prob, alg, t, u, calculate_error = false)
end
