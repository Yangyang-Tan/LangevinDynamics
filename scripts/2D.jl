using DrWatson, Plots, DifferentialEquations,CUDA
@time @quickactivate :LangevinDynamics
langevin_2d_ODE_prob()

CUDA.reclaim()
GC.gc(true)

@time sol_GPU = solve(
    langevin_2d_ODE_prob(;
        η=100.0,
        σ0=0.2,
        h=0.2,
        tspan=(0.0f0, 10.0f0),
        # u0fun=x -> CUDA.fill(5.0f0,LangevinDynamics.N, LangevinDynamics.N, LangevinDynamics.M),
        u0fun=x ->
            CUDA.randn(Float32, LangevinDynamics.N, LangevinDynamics.N, LangevinDynamics.M),
    ),
    Tsit5(),
    EnsembleSerial();
    trajectories=1,
    saveat=0.0:0.2:10.0,
    save_everystep=false,
    abstol=1e-5,
    reltol=1e-5,
)
saved_values = SavedValues(Float32, Any)
# cb = SavingCallback(
#     (u, t, integrator) -> reshape(
#         mapslices(x -> cumulant.(Ref(abs.(x)), [1, 2, 3, 4]), Array(u); dims=[1, 2]),
#         (4, LangevinDynamics.M),
#     ),
#     saved_values;
#     # saveat=0.0:2.0:1500.0,
# )

cb = SavingCallback(
    (u, t, integrator) -> begin
        # u_c=Array(u);
        ϕ=[mean(u[:,:,i,1]) for i in 1:size(u)[3]];
        return mean(ϕ)
    end,
    saved_values;
    saveat=0.0:0.2:800.0,
)

@time sol_SDE = solve(
    langevin_2d_SDE_prob(;
        γ=0.3f0,
        m2=-0.5f0,
        λ=1.0f0,
        J=0.0f0,
        tspan=(0.0f0, 400.0f0),
        T=2.0f0,
        # u0fun=x ->
        #     CUDA.fill(5.0f0, LangevinDynamics.N, LangevinDynamics.N, LangevinDynamics.M,2),
        u0fun=x ->1.0f0.+CUDA.randn(LangevinDynamics.N, LangevinDynamics.N, LangevinDynamics.M,2),
    ),
    [SOSRA(),ImplicitEM(),SImplicitMidpoint(),ImplicitRKMil(),SKSROCK(),SRA3(),SOSRI()][1],
    EnsembleSerial();
    dtmax=0.1,
    trajectories=1,
    # dt=0.1f0,
    # saveat=0.0:0.2:8.0,
    save_everystep=false,
    save_start=true,
    save_end=true,
    dense=false,
    callback=cb,
    abstol=1e-1,
    reltol=1e-1,
)

function getM(T)
    saved_values = SavedValues(Float32, Any)
    cb = SavingCallback(
    (u, t, integrator) -> begin
        # u_c=Array(u);
        ϕ=[mean(u[:,:,i,1]) for i in 1:size(u)[3]];
        return mean(ϕ)
    end,
    saved_values;
    saveat=0.0:0.2:800.0,
)
sol_SDE = solve(
    langevin_2d_SDE_prob(;
        γ=0.3f0,
        m2=-0.5f0,
        λ=1.0f0,
        J=0.0f0,
        tspan=(0.0f0, 800.0f0),
        T=T,
        # u0fun=x ->
        #     CUDA.fill(5.0f0, LangevinDynamics.N, LangevinDynamics.N, LangevinDynamics.M,2),
        u0fun=x ->1.0f0.+CUDA.randn(LangevinDynamics.N, LangevinDynamics.N, LangevinDynamics.M,2),
    ),
    [SOSRA(),ImplicitEM(),SImplicitMidpoint(),ImplicitRKMil(),SKSROCK(),SRA3(),SOSRI()][1],
    EnsembleSerial();
    dtmax=0.1,
    trajectories=1,
    # dt=0.1f0,
    # saveat=0.0:0.2:8.0,
    save_everystep=false,
    save_start=true,
    save_end=true,
    dense=false,
    callback=cb,
    abstol=1e-1,
    reltol=1e-1,
)
return saved_values.saveval
end

Nx=LangevinDynamics.N
testf=[1.0]
params
datadir("sims", savename(params, "jld2"))
wsave("/home/tyy/Documents/LangevinDynamics/data/sims/Nx=64.jld2", testf)
tempv=getM(2.0f0)

@tagsave(datadir("sims", savename(params, "jld2")), (@strdict Nx))

datadir("sims", savename(params, "jld2"))
params = @strdict Nx

saved_values.saveval[1]
temash = ash(vec(sol_GPU[:, :, :, 1, :]); kernel=Kernels.gaussian, m=5, rng=-0.5:0.001:0.5)
plot([mean(sol_GPU[:, :, :, i, :]) for i in 1:50])
plot([mean(sol_SDE[:, :, :, i, :]) for i in 1:50])
heatmap(sol_SDE[:, :,1, 1, 2,1],size=(512,512),aspect_ratio=1,right_margin = 8Plots.mm)
plot(log.(abs.(map(x->mapslices(mean,x,dims=2)[1],saved_values.saveval)[10:end])))
plot!(saved_values.saveval[100:end])
vcat(saved_values.saveval...)
saved_values.saveval
sol_SDE[:,:,:,:,:,:]
mean(saved_values.saveval.^2)-mean(saved_values.saveval)^2
mean(saved_values.saveval.^2)-mean(saved_values.saveval)^2
mean(saved_values.saveval.^2)-mean(saved_values.saveval)^2
mean(saved_values.saveval[2500:end].^2)-mean(saved_values.saveval[2500:end])^2
mean(saved_values.saveval[2500:end].^2)-mean(saved_values.saveval[2500:end])^2
ini_SDE.=cu(sol_SDE[:, :, :, :, 2,1])
ini_SDE|>Array
