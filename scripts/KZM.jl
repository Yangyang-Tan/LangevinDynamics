saved_values = SavedValues(Float32, Any)
# cb = SavingCallback(
#     (u, t, integrator) -> reshape(
#         mapslices(x -> cumulant.(Ref(abs.(x)), [1, 2, 3, 4]), Array(u); dims=[1, 2]),
#         (4, LangevinDynamics.M),
#     ),
#     saved_values;
#     # saveat=0.0:2.0:1500.0,
# )
# cb = SavingCallback(
#     (u, t, integrator) -> begin
#         # u_c=Array(u);
#         # ϕ = [mean(u[:, :, i, 1]) for i = 1:size(u)[3]]
#         ϕ = abs.(mean(u, dims = [1,2])[1,1,:,1])
#         return [mean(ϕ), var(ϕ)]
#     end,
#     saved_values;
#     saveat = 0.0:2.0:4000.0,
# )
cb = SavingCallback(
    (u, t, integrator) -> begin
        # u_c=Array(u);
        # ϕ = [mean(u[:, :, i, 1]) for i = 1:size(u)[3]]
        return mean(u, dims = [3])[:,:,1,1]
    end,
    saved_values;
    saveat = 0.0:2:2000.0,
)


Random.seed!(1234)
CUDA.seed!(1234)

@time sol_SDE = solve(
    langevin_2d_SDE_prob(;
        γ = 0.3f0,
        m2 = -1.0f0,
        λ = 1.0f0,
        J = 0.0f0,
        tspan = (0.0f0, 2000.0f0),
        T = 4.0f0,
        # u0fun=x ->
        #     CUDA.fill(0.1f0, 512, 512, 1,2),
        u0fun = x ->
            0.1f0 .+
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
    dtmax = 0.2,
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
CUDA.reclaim()
GC.gc(true)
stack(saved_values.saveval)[1,:]

saved_values.t
plot(saved_values.t, stack(saved_values.saveval)[1, :]./4.5)
heatmap(Array(saved_values.saveval[1001]), size = (512, 512), aspect_ratio = 1)

20.0f0 * (1 - tanh((500 - 500) / 50))
mapreduce
a=CUDA.randn(512,512)
b=CUDA.randn(512,512)
DSP.conv(a,b)
