using DiffEqNoiseProcess
f(u, p, t, rv) = sin(rv * t)
t0 = 0.0
f1(u,p,t) = 1.0
W = NoiseFunction(0.0, f1)

W = NoiseTransport(t0, f, randn)
W_1 = WienerProcess(t0, 0.0, 0.0;)
μ = 1.0
σ = 2.0
W = WienerProcess(t0, 0.0, 0.0;)
prob = NoiseProblem(W, (0.0, 2.0))
sol = solve(prob; dt = 0.01)
enprob = EnsembleProblem(prob)
sol = solve(enprob, LambaEulerHeun(), EnsembleThreads(); dt = 0.1, trajectories = 1000000)


plot(x->sol(x)[1],0,2)
plot(sol.u[1].t, mean(stack(sol.u), dims = 2)[:, 1], label = "Mean")
plot(sol.u[1].t, var(stack(sol.u), dims = 2)[:, 1], label = "Mean")
heatmap(
    cov(stack(sol.u)'[2:end, 2:end], corrected = false),
    yflip = true,
    aspect_ratio = 1,
    size = (500, 500),
)
[min(x, y)^2 / 2 for x = 0.1:0.1:2, y = 0.1:0.1:2]

cov(stack(sol.u)'[2:end, 2:end], corrected = false)
heatmap(
    [min(x, y)/2 for x = 0.1:0.1:2, y = 0.1:0.1:2],
    yflip = true,
    aspect_ratio = 1,
    size = (500, 500),
)


plot(sol.t, sol.u)
tspan=(0f0,1f0)
Γ_cache=[1f0/((x1-x2)^2f0+(y1-y2)^2f0+1f0) for x1=0.1f0:0.1f0:2f0, y1=0.1f0:0.1f0:2f0, x2=0.1f0:0.1f0:2f0, y2=0.1f0:0.1f0:2f0]
Γ_cache = [
    1.0f0 / ((x1 - x2)^2.0f0 + 1.0f0) for x1 = 0.001f0:0.01f0:30.0f0,
    x2 = 0.001f0:0.01f0:30.0f0
]

CorrelatedWienerProcess!(Γ_cache, tspan[1], zeros(20), zeros(20))
γ_cache.U * Diagonal(sqrt.(γ_cache.S))

γ_cache=svd(Γ_cache)
γ_cache.S
randlist1=DiffEqNoiseProcess.wiener_randn(MersenneTwister(1234),fill(1f0,2))
randlist2 = DiffEqNoiseProcess.wiener_randn(MersenneTwister(1234), fill(1.0f0, 2))

mean(randlist)

plot(randn(100))
using CUDA
a=CUDA.randn(10)
CUDA.spae
