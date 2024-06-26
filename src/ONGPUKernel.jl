function update_3d_O4_simple_langevin!(dσ, σ, fun,dx,c)
    N = size(σ, 1)
    # M = size(σ, 4)
    # dx = 1
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    # z = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    # id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    #     stride = blockDim().x * gridDim().x
    cind = CartesianIndices(σ)
    # M=blockDim().z *gridDim().z
    for i = id:blockDim().x*gridDim().x:prod(size(σ))
        # x = cind[i][1]
        # y = cind[i][2]
        # z = cind[i][3]
        # k = cind[i][4]
        x, y, z, k = Tuple(cind[i])
        xp1, xm1 = limitbound(x + 1, N), limitbound(x - 1, N)
        yp1, ym1 = limitbound(y + 1, N), limitbound(y - 1, N)
        zp1, zm1 = limitbound(z + 1, N), limitbound(z - 1, N)
        phi=sqrt(σ[x, y, z, k,1]^2+σ[x, y, z, k,2]^2+σ[x, y, z, k,3]^2+σ[x, y, z, k,4]^2)
        @inbounds dσ[x, y, z, k,1] =     #dσ = -δF/δσ
            (
                σ[xp1, y, z, k,1] +
                σ[xm1, y, z, k,1] +
                σ[x, yp1, z, k,1] +
                σ[x, ym1, z, k,1] +
                σ[x, y, zp1, k,1] +
                σ[x, y, zm1, k,1] - 6 * σ[x, y, z, k,1]
            ) / dx^2 - σ[xp1, y, z, k,1]/phi *fun(phi)+c
        @inbounds dσ[x, y, z, k,2] =     #dσ = -δF/δσ
            (
                σ[xp1, y, z, k,2] +
                σ[xm1, y, z, k,2] +
                σ[x, yp1, z, k,2] +
                σ[x, ym1, z, k,2] +
                σ[x, y, zp1, k,2] +
                σ[x, y, zm1, k,2] - 6 * σ[x, y, z, k,2]
            ) / dx^2 - σ[xp1, y, z, k,2]/phi *fun(phi)
        @inbounds dσ[x, y, z, k,3] =     #dσ = -δF/δσ
            (
                σ[xp1, y, z, k,3] +
                σ[xm1, y, z, k,3] +
                σ[x, yp1, z, k,3] +
                σ[x, ym1, z, k,3] +
                σ[x, y, zp1, k,3] +
                σ[x, y, zm1, k,3] - 6 * σ[x, y, z, k,3]
            ) / dx^2 - σ[xp1, y, z, k,3]/phi *fun(phi)
        @inbounds dσ[x, y, z, k,4] =     #dσ = -δF/δσ
            (
                σ[xp1, y, z, k,4] +
                σ[xm1, y, z, k,4] +
                σ[x, yp1, z, k,4] +
                σ[x, ym1, z, k,4] +
                σ[x, y, zp1, k,4] +
                σ[x, y, zm1, k,4] - 6 * σ[x, y, z, k,4]
            ) / dx^2 - σ[xp1, y, z, k,4]/phi *fun(phi)
    end
    #  + σ[i, jm1, k, 1] + σ[i, jp1, k, 1] -
    #     4 * σ[i, j, k, 1]) + 0.22f0 - sign(σ[i,k, 1])*tex[(abs(σ[i,k, 1])/0.005f0)+1] - γ * σ[i, k, 2]
    # @inbounds dσ[i, j] =i+j
    return nothing
end
export update_3d_O4_simple_langevin!


function langevin_3d_O4_loop_simple_GPU(dσ, σ, fun, dx,c)
    # alpha = alpha / dx^2
    # N = size(σ, 1)
    # M = size(σ, 4)
    threads = 1024
    blocks = 2^8
    @cuda blocks = blocks threads = threads maxregs = 4 update_3d_O4_simple_langevin!(
        dσ,
        σ,
        fun,
        dx,
        c,
    )
end
export langevin_3d_O4_loop_simple_GPU



function modelA_3d_O4_SDE_Simple_prob(;
    u0 = error("u0 not provided"),
    γ = 1.0f0,
    tspan = myT.((0.0, 15.0)),
    T = 5.0f0,
    c=0.1f0,
    dt = 0.1f0,
    dx=1,
    solver=DRI1NM(),
    save_start = false,
    save_everystep = false,
    save_end = false,
    abstol = 5e-2,
    reltol = 5e-2,
    args...,
)
    # u0 = init_langevin_2d(xyd_brusselator)

    # Ufun = funout_cut2(para)
    # Ufun = funout(para)
    function Ufun(x)
        -x + x^3/6
    end

    # output_func(sol, i) = i
    # output_func(sol, i) = (begin
    #     ar=Array(sol)[:,1,1,:]
    #     # mean(ar,dims=1)
    #     mσ=sum(ar.*weight_mean2.^2,dims=1) *4*pi/(V)
    #     varσ=sum((ar.-mσ).^2 .*weight_mean2.^2,dims=1) *4*pi/(V)
    #     kσ=sum((ar.-mσ).^4 .*weight_mean2.^2,dims=1) *4*pi./(V*varσ.^2)
    #     stack([mσ, varσ, kσ.-3])
    # end, false)
    # p = myT.((γ, m2, λ, J))
    # ODEfun_tex(dσ, σ) = langevin_3d_loop_simple_GPU(dσ, σ, Ufun)
    function g2(du, u, p, t)
        du .= sqrt(2*T/γ)
    end
    u0_GPU = CuArray(u0)
    GC.gc(true)
    CUDA.reclaim()
    function ODEfun_tex(dσ, σ, p, t)
        CUDA.@sync langevin_3d_O4_loop_simple_GPU(dσ, σ, Ufun,dx,c)
        CUDA.@sync dσ .= (1 / γ) .* dσ
    end
    #println("noise=", sqrt(T))
    sdeprob = SDEProblem(ODEfun_tex,g2, u0_GPU, tspan)
    # sdeprob = SDEProblem(ODEfun_tex,g, u0_GPU, tspan,dx)
    saved_values = SavedValues(Float32, Any)
    cb = SavingCallback(
        (x, t, integrator) -> begin
            @show t
            x1 = (mean(x, dims = [1, 2, 3])[1, 1, 1, :,:])
            # return var(x1)/T
            x=sqrt.(mean(x1.^2,dims=[2])[:,1])
            return [mean(x) var(x)]
            # x1
        end,
        saved_values;
        saveat=tspan[1]:(0.05):tspan[2],
        save_everystep = false,
        save_start = true,
    )
    solve(
        sdeprob,
        # PCEuler(ggprime),
        solver,
        #DRI1(),
        #SRA3(),
        # SRIW1(),
        #RDI1WM(),
        #EM(),
        # dt = 0.002,
        save_start = save_start,
        save_everystep = save_everystep,
        save_end = save_end,
        abstol = abstol,
        reltol = reltol,
        callback = cb,
        # dt=0.0015f0,
        args...,
    )
     [saved_values.t saved_values.saveval]
end
export modelA_3d_O4_SDE_Simple_prob



function modelA_3d_O4_SDE_Simple_prob2(;
    u0 = error("u0 not provided"),
    γ = 1.0f0,
    tspan = myT.((0.0, 15.0)),
    T = 5.0f0,
    c=0.1f0,
    dt = 0.1f0,
    dx=1,
    solver=DRI1NM(),
    save_start = false,
    save_everystep = false,
    save_end = false,
    abstol = 5e-2,
    reltol = 5e-2,
    args...,
)
    # u0 = init_langevin_2d(xyd_brusselator)

    # Ufun = funout_cut2(para)
    # Ufun = funout(para)
    function Ufun(x)
        -x + x^3/6
    end

    # output_func(sol, i) = i
    # output_func(sol, i) = (begin
    #     ar=Array(sol)[:,1,1,:]
    #     # mean(ar,dims=1)
    #     mσ=sum(ar.*weight_mean2.^2,dims=1) *4*pi/(V)
    #     varσ=sum((ar.-mσ).^2 .*weight_mean2.^2,dims=1) *4*pi/(V)
    #     kσ=sum((ar.-mσ).^4 .*weight_mean2.^2,dims=1) *4*pi./(V*varσ.^2)
    #     stack([mσ, varσ, kσ.-3])
    # end, false)
    # p = myT.((γ, m2, λ, J))
    # ODEfun_tex(dσ, σ) = langevin_3d_loop_simple_GPU(dσ, σ, Ufun)
    function g2(du, u, p, t)
        du .= sqrt(2*T/γ)
    end
    u0_GPU = CuArray(u0)
    GC.gc(true)
    CUDA.reclaim()
    function ODEfun_tex(dσ, σ, p, t)
        CUDA.@sync langevin_3d_O4_loop_simple_GPU(dσ, σ, Ufun,dx,c)
        CUDA.@sync dσ .= (1 / γ) .* dσ
    end
    #println("noise=", sqrt(T))
    sdeprob = SDEProblem(ODEfun_tex,g2, u0_GPU, tspan)
    # sdeprob = SDEProblem(ODEfun_tex,g, u0_GPU, tspan,dx)
    saved_values = SavedValues(Float32, Any)
    cb = SavingCallback(
        (x, t, integrator) -> begin
            @show t
            x1 = (mean(x, dims = [1, 2, 3,4])[1, 1, 1, 1,:])
            # return var(x1)/T
            return Array(x1)
            # x1
        end,
        saved_values;
        saveat=tspan[1]:(0.05):tspan[2],
        save_everystep = false,
        save_start = true,
    )
    solve(
        sdeprob,
        # PCEuler(ggprime),
        solver,
        #DRI1(),
        #SRA3(),
        # SRIW1(),
        #RDI1WM(),
        #EM(),
        # dt = 0.002,
        save_start = save_start,
        save_everystep = save_everystep,
        save_end = save_end,
        abstol = abstol,
        reltol = reltol,
        callback = cb,
        # dt=0.0015f0,
        args...,
    )
     [saved_values.t saved_values.saveval]
end
export modelA_3d_O4_SDE_Simple_prob2