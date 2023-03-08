using AdvancedHMC,ForwardDiff
D = 32^3; initial_θ = CUDA.rand(D)
metric = UnitEuclideanMetric(T, size(initial_θ))
hf=x -> -dot(x,x) / 2
gf=x->(hf(x),x)
hamiltonian = Hamiltonian(metric,hf , gf)
# integrator = Leapfrog(one(T) / 5)
#     proposal = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(5)))
gf(initial_θ)

integrator = Leapfrog(one(T) / 5)
proposal = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(5)))

samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples)


1
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
temash = ash(
    vec(stack(randn(20000)));
    # kernel = Kernels.gaussian,
    m = 50,
    rng = -3:0.01:3,
)

temash = ash(
    vec(stack(samples));
    # kernel = Kernels.gaussian,
    m = 50,
    rng = -3:0.01:3,
)
plot(temash, labels = "Sims ini")
1
using StatsBase
(1-cumulant(vec(stack(samples)),2))/cumulant(vec(stack(samples)),2)
(1-cumulant(vec(stack(randn(20000))),2))/cumulant(vec(stack(randn(20000))),2)
cumulant(vec(stack(randn(20000))),4)
cumulant(vec(stack(randn(20000))),4)
