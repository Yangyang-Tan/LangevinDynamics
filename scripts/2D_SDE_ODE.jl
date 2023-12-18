function langevin_kernel!(dσ, σ, γ, m2, λ, J)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    N = size(σ, 1)
    M = size(σ, 3)
    # M=blockDim().z *gridDim().z
    if (i > 0 && i <= N && j > 0 && j <= N && k > 0 && k <= M)
        ip1, im1, jp1, jm1 = limitbound(i + 1, N),
        limitbound(i - 1, N),
        limitbound(j + 1, N),
        limitbound(j - 1, N)
        @inbounds dσ[i, j, k, 1] = σ[i, j, k, 2]
        @inbounds dσ[i, j, k, 2] =
            (
                σ[im1, j, k, 1] + σ[ip1, j, k, 1] + σ[i, jm1, k, 1] + σ[i, jp1, k, 1] -
                4 * σ[i, j, k, 1]
            ) - δUδσ(σ[i, j, k, 1]; m2 = m2, λ = λ, J = J) - γ * σ[i, j, k, 2]
        # @inbounds dσ[i, j] =i+j
    end
    return nothing
end



saved_values = SavedValues(Float32, Any)
cb = SavingCallback(
    (u, t, integrator) -> begin
        @show t
        u_c = @view u[:, :, :, 1]
        ϕ = abs.(mean(u_c, dims = [1, 2])[1, 1, :])
        return [mean(ϕ), var(ϕ)]
    end,
    saved_values;
    saveat = 0.0:0.5:40.0,
)
@time sol_randini = solve(
    langevin_2d_SDE_prob(;
        γ = 8.0f0,
        m2 = -1.0f0,
        λ = 1.0f0,
        J = 0.0f0,
        tspan = (0.0f0, 40.0f0),
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
        SOSRI(),EM(),
    ][end-1],
    EnsembleSerial();
    dtmax = 0.5,
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
GC.gc(true)
CUDA.reclaim()
a1=CUDA.randn(128, 128, 4096, 2)

saved_values.t
plot(saved_values.t, stack(saved_values.saveval)[1, :])
