using AverageShiftedHistograms, QuadGK
sol_0D_SDE = solve(
    langevin_0d_tex_SDE_prob(
        langevin_0d_tex_loop_GPU;
        γ = 2.2f0,
        m2 = -1.0f0,
        λ = 1.0f0,
        J = 0.0f0,
        tspan = (0.0f0, 20.0f0),
        T = 1.0f0,
        tex = δUδσTextureTex,
        # u0fun=x ->
        #     CUDA.fill(0.0f0, LangevinDynamics.N,2),
        u0fun = x -> 0.0f0 .* CUDA.randn(LangevinDynamics.M, LangevinDynamics.M, 2),
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
    saveat = 0.0:2.0:20.0,
    save_everystep = false,
    # save_start=true,
    # save_end=false,
    dense = false,
    # callback=cb,
    abstol = 1e-1,
    reltol = 1e-1,
)
heatmap(
    sol_1D_SDE[1][:, :, 1, 3],
    size = (512, 512),
    aspect_ratio = 1,
    # right_margin = 8Plots.mm,
)


temash = ash(
    vec(sol_0D_SDE[1][:, :, 1, 1]);
    # kernel = Kernels.gaussian,
    m = 3,
    rng = -7:0.1:7,
)
plot!(temash, labels = "Sims ini", title = "T=10.0,0+1 D", xlabel = "σ", ylabel = "P[σ]")
temash = ash(
    vec(sol_0D_SDE[1][:, :, 1, 4]);
    # kernel = Kernels.gaussian,
    m = 4,
    rng = -1:0.02:1,
)
plot!(temash, labels = "Sims end", title = "T=10.0,0+1 D")

sol_0D_SDE[1][:, :, 1, 2]

plot(mean(sol_0D_SDE[1][:, :, 2, end], dims = 2)[:, 1])
Z_0D = quadgk(σ -> exp(-Uσ_CPU_fun2(σ) / 0.5), -5.5, 5.5)[1]
plot(σ -> exp(-Uσ_CPU_fun2(σ) / 0.5) / Z_0D, -3, 3, label = "Exact", color = :black)



plot(x -> Uσ_CPU_fun2(abs(x)), -0.45, 0.45)
GC.gc(true)
CUDA.reclaim()

Uσ_CPU_fun2(x) = Uσ_CPU_fun(abs(x)) - 0.1 * x
plot(
    σ -> exp(-Uσ_CPU_fun2(abs(σ)) / 0.1) / Z_0D,
    -1.5,
    1.5,
    label = "Exact",
    color = :black,
)
Z_0D=quadgk(σ -> exp(-Uσ_CPU_fun2(σ)/0.5),-5.5,5.5)[1]
mU =
    quadgk(σ -> (σ) * exp(-Uσ_CPU_fun2(σ) / 0.5) / Z_0D, -5.5, 5.5, rtol = 1e-5, atol = 1e-5)[1]
quadgk(
    σ -> abs((σ - mU)^4) * exp(-Uσ_CPU_fun2(σ) / 0.5) / Z_0D,
    -5.5,
    5.5,
    rtol = 1e-8,
    atol = 1e-8,
)[1] /
quadgk(
    σ -> (σ - mU)^2 * exp(-Uσ_CPU_fun2(σ) / 0.5) / Z_0D,
    -5.5,
    5.5,
    rtol = 1e-8,
    atol = 1e-8,
)[1]^2

plot(Uσ_CPU_fun, -2.5, 2.5)
