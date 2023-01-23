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
    (u, t, integrator) -> mean(u),
    saved_values;
    saveat=40.0:20.0:150.0,
)

sol_SDE = solve(
    langevin_2d_SDE_prob(;
        η=100.0,
        σ0=0.1,
        h=0.0,
        tspan=(0.0f0, 150.0f0),
        u0fun=x ->
            CUDA.fill(1.0f0, LangevinDynamics.N, LangevinDynamics.N, LangevinDynamics.M),
        # u0fun=x ->
        #     CUDA.randn(Float32, LangevinDynamics.N, LangevinDynamics.N, LangevinDynamics.M),
    ),
    SOSRA(),
    EnsembleSerial();
    trajectories=1,
    dt=0.1,
    # saveat=0.0:0.2:10.0,
    save_everystep=false,
    save_start=false,
    save_end=true,
    dense=false,
    callback=cb,
    abstol=1e-1,
    reltol=1e-1,
)
saved_values.saveval[1]
temash = ash(vec(sol_GPU[:, :, :, 1, :]); kernel=Kernels.gaussian, m=5, rng=-0.5:0.001:0.5)
plot([mean(sol_GPU[:, :, :, i, :]) for i in 1:50])
plot([mean(sol_SDE[:, :, :, i, :]) for i in 1:50])
heatmap(sol_SDE[:, :, 1, 1, 1],size=(512,512),aspect_ratio=1)
plot(log.(abs.(map(x->mapslices(mean,x,dims=2)[1],saved_values.saveval)[10:end])))
plot(map(x->mapslices(mean,x,dims=2)[1],saved_values.saveval))
plot!(saved_values.saveval)
vcat(saved_values.saveval...)
saved_values.saveval
