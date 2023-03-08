sol_1D_SDE = solve(
    langevin_1d_tex_SDE_prob(;
        γ = 2.2f0,
        m2 = -1.0f0,
        λ = 5.0f0,
        J = 0.0f0,
        tspan = (0.0f0, 10.0f0),
        T = 0.4f0,
        tex=δUδσTextureTex,
        # u0fun=x ->
        #     CUDA.fill(0.0f0, LangevinDynamics.N,LangevinDynamics.M,2),
        u0fun = x -> 0.1f0 *CUDA.randn(LangevinDynamics.N, LangevinDynamics.M, 2),
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
    dtmax = 0.02,
    trajectories = 1,
    # dt=0.01f0,
    saveat = 0.0:0.1:10.0,
    save_everystep = false,
    # save_start=true,
    # save_end=false,
    dense = false,
    # callback=cb,
    abstol = 1e-1,
    reltol = 1e-1,
)

mean(mapslices(x -> cumulant(x, 2), sol_1D_SDE[1][:, :, 1, :], dims = 1), dims = 2)[1, 1, :]
heatmap(
    sol_1D_SDE[1][:, :, 1, 3],
    size = (512, 512),
    aspect_ratio = 1,
    # right_margin = 8Plots.mm,
)

plot!(
    mean(mapslices(x -> cumulant(x, 4)/cumulant(x, 2)^2, sol_1D_SDE[1][:, :, 1, :], dims = 1), dims = 2)[
        1,
        1,
        :,
    ],
)

plot(readdlm("data/phi.dat")[1:2500,1],readdlm("data/V.dat")[1:2500,1].-readdlm("data/V.dat")[1,1])
