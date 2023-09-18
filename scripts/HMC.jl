using AdvancedHMC,ForwardDiff,CUDA,LogDensityProblems,ReTest,LinearAlgebra,StatsBase
include("common.jl")
GC.gc(true)
CUDA.reclaim()
n_chains = 10000
n_samples = 500
dim = 1
T=Float32
m, s, θ₀ = zeros(T, dim), ones(T, dim), rand(T, dim, n_chains)
m, s, θ₀ = CuArray(m), CuArray(s), CuArray(θ₀)

target = Gaussian(m, s)
metric = UnitEuclideanMetric(T, size(θ₀))
ℓπ, ∇ℓπ = get_ℓπ(target), get_∇ℓπ(target)

hamiltonian = Hamiltonian(metric, hf, gf)
integrator = Leapfrog(one(T) / 200)
proposal = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(200)))
samples, stats = sample(hamiltonian, proposal, θ₀, n_samples;progress=false)


function hf(x)
    -dropdims(mean(x.*x,dims=[1]),dims=1)./2
end
function gf(x)
    return (hf(x),x)
end

gf=x->(hf(x),x)

hf(CUDA.randn(10,10))

initial_θ = CUDA.rand(D)
metric = UnitEuclideanMetric(T, size(initial_θ))
hf=x -> -dot(x,x) / 2
gf=x->(hf(x),x)
hamiltonian = Hamiltonian(metric,hf , gf)
# integrator = Leapfrog(one(T) / 5)
#     proposal = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(5)))
gf(initial_θ)

integrator = Leapfrog(one(T) / 5)
proposal = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(5)))

n_samples=10^4

samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples)



initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
integrator = Leapfrog(0.00625f0)

T = Float32
# Define an HMC sampler, with the following components
#   - multinomial sampling scheme,
#   - generalised No-U-Turn criteria, and
#   - windowed adaption for step-size and diagonal mass matrix
proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

# Run the sampler to draw samples from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample
n_samples, n_adapts = 200_00, 1000
samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=true)
interrupt()
using AverageShiftedHistograms
Array(vec(stack(samples)))
temash = ash(
    Array(vec(stack(samples)));
    # kernel = Kernels.gaussian,
    m = 5,
    rng = -5:0.01:5,
)
temash = ash(
    vec(randn(2*10^6));
    # kernel = Kernels.gaussian,
    m = 5,
    rng = -5:0.02:5,
)
plot(temash, labels = "Sims ini",hist=false)
plot(temash, labels = "Sims ini")

1
using StatsBase,Plots
(1-cumulant(vec(stack(samples)),2))/cumulant(vec(stack(samples)),2)
(1-cumulant(vec(stack(randn(20000))),2))/cumulant(vec(stack(randn(20000))),2)
cumulant(vec(stack(randn(20000))),4)
cumulant(vec(stack(randn(20000))),4)
