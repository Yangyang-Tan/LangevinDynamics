function O1_3d_SDE_Simple_prob(;
    u0 = error("u0 not provided"),
    γ = 1.0f0,
    tspan = myT.((0.0, 15.0)),
    T = 5.0f0,
    dt = 0.1f0,
    args...,
)
    # u0 = init_langevin_2d(xyd_brusselator)

    # mσ = sqrt(abs(para.λ1) + 2 * para.ρ0 * para.λ2)
    # Ufun = funout_cut2(para)
    # Ufun = funout(para)
    function Ufun(x)
        (T-1)*x+x^3
    end
    # function Ufun(x)
    #     x*ifelse(abs(x)>0.4f0,x^2-0.4f0^2,0)
    # end
    ODEfun_tex(dσ, σ) = langevin_3d_loop_simple_GPU(dσ, σ, Ufun)
    # println("noise=", sqrt(mσ * coth(mσ / (2 * T))))
    function g(du, u, p, t)
        du[:, :, :, :] .= 0.001f0*sqrt(2  * T/γ)
        #du .= sqrt(mσ *γ* coth(mσ / (2 * T)))
    end
    u0_GPU = CuArray(u0)
    GC.gc(true)
    CUDA.reclaim()
    function ODEfun_tex(dσ, σ, p, t)
        CUDA.@sync langevin_3d_loop_simple_GPU(dσ, σ, Ufun)
        CUDA.@sync dσ .= (1 / γ) .* dσ
    end
    #println("noise=", sqrt(T))
    sdeprob = SDEProblem(ODEfun_tex,g, u0_GPU, tspan)
    saved_values = SavedValues(Float32, Any)
    cb = SavingCallback(
        (x, t, integrator) -> begin
            x1 = Array(abs.(mean(x, dims = [1, 2, 3])[1, 1, 1, :]))
            #return mean(var(x,dims=[1,2,3]))
             return mean(x1)
        end,
        saved_values;
        # saveat=0.0:0.1:100.0,
        save_everystep = true,
        save_start = true,
    )
    sol=solve(
        sdeprob,
        # PCEuler(ggprime),
        DRI1NM(),
        dt = dt,
        save_start = false,
        save_everystep = false,
        # save_end = false,
        abstol = 1e-1,
        reltol = 1e-1,
        callback = cb,
    )
    [saved_values.t, saved_values.saveval, sol]
end





u0_1 = fill(0.6f0, 32, 32, 32, 2^1)

u0_1 = randn(32, 32, 32, 2^8)
mean(var(u0_1,dims=[1,2,3]))
myT=Float32
sol_O1=O1_3d_SDE_Simple_prob(;
    u0 = u0_1,
    γ = 20.0f0,
    tspan = myT.((0.0, 2000.0)),
    T = 0.99f0,
    dt=0.05f0
)
sol_O1[1]
plot(sol_O1[1],sol_O1[2])

for Tem in 0.5f0:0.01f0:1.0f0
    sol_O1=O1_3d_SDE_Simple_prob(;
    u0 = u0_1,
    γ = 20.0f0,
    tspan = myT.((0.0, 2000.0)),
    T = Tem,
    dt=0.05f0
    )
    writedlm("sims/eqcd_relax_phase/relax_time_GL_gamma20_no/Tem=$Tem.dat",stack(sol_O1[1:2]))
end


for Tem in 0.5f0:0.01f0:1.0f0
    sol_O1=O1_3d_SDE_Simple_prob(;
    u0 = u0_1,
    γ = 20.0f0,
    tspan = myT.((0.0, 50.0)),
    T = Tem,
    dt=0.05f0
    )
    writedlm("sims/eqcd_relax_phase/relax_time_GL_gamma5/Tem=$Tem.dat",stack(sol_O1[1:2]))
end




sol_O1=modelA_3d_SDE_Simple_tex_prob(;
    u0 = u0_1,
    γ = 1.0f0,
    tspan = myT.((0.0, 15.0)),
    T = 5.0f0,
    dt = 0.05f0,
    x_grid=-4f0:0.01f0:4f0,  
    δUδσ_grid=Ufun1.(-4f0:0.01f0:4f0),
)


function Ufun1(x)
    x*ifelse(abs(x)>1,x^2-1,0)
end
plot!(Ufun1,-3:0.01:3)
sol_O1[1,:]
plot(sol_O1[:,1],sol_O1[:,2])
sol_O1[3].u[1]
using AverageShiftedHistograms
his_v=Array(mean(sol_O1[3].u[1],dims=[1,2,3])[1,1,1,:])
his_v=Array(sol_O1[3].u[1])
his_ash = ash(his_v, rng = -3:.02:3, m = 2)
plot!(his_ash)



using AdvancedHMC, ForwardDiff
using LogDensityProblems
using LinearAlgebra

# Define the target distribution using the `LogDensityProblem` interface
struct LogTargetDensity
    dim::Int
end
sum(abs2,θ)

@inline @fastmath function myf(x)
    (x^2/2-1)^2/2
end
LogDensityProblems.logdensity(p::LogTargetDensity, θ) = -sum(myf,θ)  # standard multivariate normal
LogDensityProblems.dimension(p::LogTargetDensity) = p.dim
LogDensityProblems.capabilities(::Type{LogTargetDensity}) = LogDensityProblems.LogDensityOrder{0}()

# Choose parameter dimensionality and initial parameter value
D = 16*16; initial_θ = rand(D)
ℓπ = LogTargetDensity(D)

# Set the number of samples to draw and warmup iterations
n_samples, n_adapts = 2_000, 1_000

# Define a Hamiltonian system
metric = DiagEuclideanMetric(D)
hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)

# Define a leapfrog solver, with the initial step size chosen heuristically
initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
integrator = Leapfrog(initial_ϵ)

# Define an HMC sampler with the following components
#   - multinomial sampling scheme,
#   - generalised No-U-Turn criteria, and
#   - windowed adaption for step-size and diagonal mass matrix
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

# Run the sampler to draw samples from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample
samples, stats = sample(hamiltonian, kernel, initial_θ, n_samples, adaptor, n_adapts; progress=true)
samples


his_v=stack(samples)
his_v=mean.(samples)
his_ash = ash(his_v, rng = -5:.001:5, m = 20)
plot(his_ash)




