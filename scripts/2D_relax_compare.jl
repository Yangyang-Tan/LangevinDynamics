using DrWatson, Plots, DifferentialEquations, CUDA
@time @quickactivate :LangevinDynamics

phiconfig_T50=wload(datadir("sim", "phiconfig_T50.jld2"))["phiconfig_T50"]

saved_values = SavedValues(Float32, Any)
cb = SavingCallback(
    (u, t, integrator) -> begin
        @show t
        u_c = @view u[:,:,:,1]
        ϕ=abs.(mean(u_c, dims = [1, 2])[1,1,:])
        return [mean(ϕ), var(ϕ)]
    end,
    saved_values;
    saveat = 0.0:0.2:200.0,
)




@time sol_quench = solve(
    langevin_2d_SDE_prob(;
        γ = 0.3f0,
        m2 = -1.0f0,
        λ = 1.0f0,
        J = 0.0f0,
        tspan = (0.0f0, 200.0f0),
        T = 5.0f0,
        # u0fun=x ->
        #     CUDA.fill(5.0f0, LangevinDynamics.N, LangevinDynamics.N, LangevinDynamics.M,2),
        u0fun = x -> CuArray(phiconfig_T50),
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
    dtmax = 0.1,
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

@tagsave(datadir("sim", "2D_relax_compare", "quench.jld2"), @strdict saved_values)


@time sol_randini = solve(
    langevin_2d_SDE_prob(;
        γ = 0.3f0,
        m2 = -1.0f0,
        λ = 1.0f0,
        J = 0.0f0,
        tspan = (0.0f0, 200.0f0),
        T = 5.0f0,
        # u0fun=x ->
        #     CUDA.fill(5.0f0, LangevinDynamics.N, LangevinDynamics.N, LangevinDynamics.M,2),
        u0fun = x -> 1.0f0 .+ CUDA.randn(64, 64, 4096, 2),
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
    dtmax = 0.1,
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

@tagsave(datadir("sim", "2D_relax_compare", "randini.jld2"), @strdict saved_values)




plot(saved_values.t, stack(saved_values.saveval)[1, :])
