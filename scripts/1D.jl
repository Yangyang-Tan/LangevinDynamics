using DrWatson, Plots, DifferentialEquations
@time @quickactivate :LangevinDynamics


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
    saveat=20.0:20.0:1500.0,
)


@time sol_1D_SDE = solve(
    langevin_1d_SDE_prob(;
        η=100.0,
        σ0=0.1,
        h=0.0,
        tspan=(0.0f0, 1500.0f0),
        u0fun=x ->
            CUDA.fill(1.0f0, LangevinDynamics.N, LangevinDynamics.M),
        # u0fun=x ->
        #     CUDA.randn(Float32, LangevinDynamics.N, LangevinDynamics.M),
    ),
    SRA3(),
    # dt=0.01,
    EnsembleSerial();
    trajectories=1,
    dt=0.1,
    # saveat=0.0:0.2:10.0,
    save_everystep=false,
    save_start=true,
    save_end=true,
    dense=false,
    callback=cb,
    abstol=1e-1,
    reltol=1e-1,
)
plot(sol_1D_SDE[:,1,2,1])
plot(saved_values.saveval)
