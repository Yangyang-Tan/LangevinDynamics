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
        # ϕ = [mean(u[:, :, i, 1]) for i = 1:size(u)[3]]
        return u[:, :, 1, 1]
    end,
    saved_values;
    saveat = 0.0:50.0:1000.0,
)

@time sol_SDE = solve(
    langevin_2d_SDE_prob(;
        γ = 0.0f0,
        m2 = -1.0f0,
        λ = 1.0f0,
        J = 0.0f0,
        tspan = (0.0f0, 1000.0f0),
        T = 0.5f0,
        # u0fun=x ->
        #     CUDA.fill(5.0f0, LangevinDynamics.N, LangevinDynamics.N, LangevinDynamics.M,2),
        u0fun = x ->
            0.0f0 .+
            CUDA.randn(512, 512, 1, 2),
    ),
    [
        SOSRA(),
        ImplicitEM(),
        SImplicitMidpoint(),
        ImplicitRKMil(),
        SKSROCK(),
        SRA3(),
        SOSRI(),
    ][1],
    EnsembleSerial();
    dtmax = 0.05,
    trajectories = 1,
    # dt=0.1f0,
    # saveat=0.0:0.2:8.0,
    save_everystep = false,
    save_start = true,
    save_end = true,
    dense = false,
    callback = cb,
    abstol = 1e-1,
    reltol = 1e-1,
)

saved_values.t
heatmap(Array(saved_values.saveval[2]), size = (512, 512), aspect_ratio = 1)
