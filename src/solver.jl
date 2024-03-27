function langevin_2d_SDE_prob(
    ODEfun = langevin_2d_loop_GPU;
    u0fun = (i) -> CUDA.randn(myT, N, N, M, 2),
    γ = 1.0f0,
    m2 = -1.0f0,
    λ = 1.0f0,
    J = 0.0f0,
    tspan = myT.((0.0, 15.0)),
    T = 5.0f0,
)
    # u0 = init_langevin_2d(xyd_brusselator)
    function g(du, u, p, t)
        du[:, :, :, 1] .= 0.0f0
        du[:, :, :, 2] .= sqrt(2 * T * γ)
        # du[:, :, :, 2] .= sqrt((20f0 * (1 - tanh((t - 20)/20)) + 2*T) * γ)
    end
    u0_GPU = 1.0f0
    function prob_func(prob, i, repeat)
        return remake(prob; u0 = u0fun(i))
    end
    output_func(sol, i) = (Array(sol), false)
    # W = WienerProcess(0.0,0.0,0.0)
    p = myT.((γ, m2, λ, J))
    return EnsembleProblem(
        SDEProblem(ODEfun, g, u0_GPU, tspan, p);
        prob_func = prob_func,
        output_func = output_func,
    )
end
export langevin_2d_SDE_prob



# function langevin_2d_SDE_nonlocal_prob(;
#     u0fun = (i) -> CUDA.randn(myT, 512, 512, 1, 2),
#     γ = 1.0f0,
#     m2 = -1.0f0,
#     λ = 1.0f0,
#     J = 0.0f0,
#     tspan = myT.((0.0, 15.0)),
#     T = 5.0f0,
# )
#     u0_temp=u0fun(1)
#     Nx,M=size(u0_temp,1),size(u0_temp,3)
#     indexlist = [0:Nx÷2 -1; Nx÷2:-1:1]
#     Ztdata_2D_cache = fft(
#         cu([1 / (x^2 + y^2 + 1.0f0^2) for x in indexlist, y in indexlist, z = 1:M]),
#         1:2,
#     )
#     u_2D_cache = CuArray{ComplexF32}(undef, Nx, Nx, M)
#     γσ_cache = CuArray{Float32}(undef, Nx, Nx, M)
#     Γ=
#     function SDEfun_f!(du,u,p,t)
#         langevin_2d_loop_nonlocal_GPU(du, u, γσ_cache, p, Ztdata_2D_cache, u_2D_cache)
#     end
#     nonlocal_noise = CorrelatedWienerProcess!(,tspan[1],zeros(Nx,Nx,M,2),zeros(Nx,Nx,M,2))
#     function g(du, u, p, t)
#         du[:, :, :, 1] .= 0.0f0
#         # du[:,:,:,2] .= sqrt(2 * T*γ)
#         du[:, :, :, 2] .= sqrt((20.0f0 * (1 - tanh((t - 20) / 20)) + 2 * T) * γ)
#     end
#     u0_GPU = 1.0f0
#     function prob_func(prob, i, repeat)
#         return remake(prob; u0 = u0fun(i))
#     end
#     output_func(sol, i) = (Array(sol), false)
#     # W = WienerProcess(0.0,0.0,0.0)
#     p = myT.((m2, λ, J))
#     return EnsembleProblem(
#         SDEProblem(ODEfun, g, u0_GPU, tspan, p);
#         prob_func = prob_func,
#         output_func = output_func,
#     )
# end
# export langevin_2d_SDE_nonlocal_prob


function langevin_1d_SDE_prob(
    ODEfun = langevin_1d_loop_GPU;
    u0fun = (i) -> CUDA.randn(myT, N, M, 2),
    γ = 1.0f0,
    m2 = -1.0f0,
    λ = 1.0f0,
    J = 0.0f0,
    tspan = myT.((0.0, 15.0)),
    T = 5.0f0,
)
    # u0 = init_langevin_2d(xyd_brusselator)
    function g(du, u, p, t)
        # return du .= mσ*coth(mσ/(2*T))
        du[:, :, 1] .= 0.0f0
        du[:, :, 2] .= sqrt(2 * T * γ)
    end
    u0_GPU = 1.0f0
    function prob_func(prob, i, repeat)
        return remake(prob; u0 = u0fun(i))
    end
    output_func(sol, i) = (Array(sol), false)
    # output_func(sol, i) = (begin
    #     ar=Array(sol)[:,1,1,:]
    #     # mean(ar,dims=1)
    #     mσ=sum(ar.*weight_mean2.^2,dims=1) *4*pi/(V)
    #     varσ=sum((ar.-mσ).^2 .*weight_mean2.^2,dims=1) *4*pi/(V)
    #     kσ=sum((ar.-mσ).^4 .*weight_mean2.^2,dims=1) *4*pi./(V*varσ.^2)
    #     stack([mσ, varσ, kσ.-3])
    # end, false)
    p = myT.((γ, m2, λ, J))
    return EnsembleProblem(
        SDEProblem(ODEfun, g, u0_GPU, tspan, p);
        prob_func = prob_func,
        output_func = output_func,
    )
end
export langevin_1d_SDE_prob


function langevin_1d_tex_SDE_prob(
    ODEfun = langevin_1d_tex_loop_GPU;
    u0fun = (i) -> CUDA.randn(myT, N, M, 2),
    γ = 1.0f0,
    m2 = -1.0f0,
    λ = 1.0f0,
    J = 0.0f0,
    tspan = myT.((0.0, 15.0)),
    T = 5.0f0,
    tex = tex,
)
    # u0 = init_langevin_2d(xyd_brusselator)
    function g(du, u, p, t)
        # return du .= mσ*coth(mσ/(2*T))
        du[:, :, 1] .= 0.0f0
        du[:, :, 2] .= sqrt(2 * T * γ)
    end
    u0_GPU = 1.0f0
    function prob_func(prob, i, repeat)
        return remake(prob; u0 = u0fun(i))
    end
    output_func(sol, i) = (Array(sol), false)
    # output_func(sol, i) = (begin
    #     ar=Array(sol)[:,1,1,:]
    #     # mean(ar,dims=1)
    #     mσ=sum(ar.*weight_mean2.^2,dims=1) *4*pi/(V)
    #     varσ=sum((ar.-mσ).^2 .*weight_mean2.^2,dims=1) *4*pi/(V)
    #     kσ=sum((ar.-mσ).^4 .*weight_mean2.^2,dims=1) *4*pi./(V*varσ.^2)
    #     stack([mσ, varσ, kσ.-3])
    # end, false)
    p = myT.((γ, m2, λ, J))
    ODEfun_tex(dσ, σ, p, t) = langevin_1d_tex_loop_GPU(dσ, σ, tex, p, t)
    return EnsembleProblem(
        SDEProblem(ODEfun_tex, g, u0_GPU, tspan, p);
        prob_func = prob_func,
        output_func = output_func,
    )
end
export langevin_1d_tex_SDE_prob


function langevin_0d_SDE_prob(
    ODEfun = langevin_0d_loop_GPU;
    u0fun = (i) -> CUDA.randn(myT, N, M, 2),
    γ = 1.0f0,
    m2 = -1.0f0,
    λ = 1.0f0,
    J = 0.0f0,
    tspan = myT.((0.0, 15.0)),
    T = 5.0f0,
)
    # u0 = init_langevin_2d(xyd_brusselator)
    function g(du, u, p, t)
        # return du .= mσ*coth(mσ/(2*T))
        du[:, :, 1] .= 0.0
        du[:, :, 2] .= sqrt(2 * T * γ)
    end
    u0_GPU = 1.0f0
    function prob_func(prob, i, repeat)
        return remake(prob; u0 = u0fun(i))
    end
    output_func(sol, i) = (Array(sol), false)
    p = myT.((γ, m2, λ, J))
    # W = WienerProcess(0.0,0.0,0.0)
    return EnsembleProblem(
        SDEProblem(ODEfun, g, u0_GPU, tspan, p);
        prob_func = prob_func,
        output_func = output_func,
    )
end
export langevin_0d_SDE_prob



function langevin_0d_tex_SDE_prob(
    ODEfun = langevin_0d_tex_loop_GPU;
    u0fun = (i) -> CUDA.randn(myT, N, M, 2),
    γ = 1.0f0,
    m2 = -1.0f0,
    λ = 1.0f0,
    J = 0.0f0,
    tspan = myT.((0.0, 15.0)),
    T = 5.0f0,
    tex = tex,
)
    # u0 = init_langevin_2d(xyd_brusselator)
    function g(du, u, p, t)
        # return du .= mσ*coth(mσ/(2*T))
        du[:, :, 1] .= 0.0
        du[:, :, 2] .= sqrt(2 * T * γ)
    end
    u0_GPU = 1.0f0
    function prob_func(prob, i, repeat)
        return remake(prob; u0 = u0fun(i))
    end
    output_func(sol, i) = (Array(sol), false)
    p = myT.((γ, m2, λ, J))
    # W = WienerProcess(0.0,0.0,0.0)
    ODEfun_tex(dσ, σ, p, t) = langevin_0d_tex_loop_GPU(dσ, σ, tex, p, t)
    return EnsembleProblem(
        SDEProblem(ODEfun_tex, g, u0_GPU, tspan, p);
        prob_func = prob_func,
        output_func = output_func,
    )

end
export langevin_0d_tex_SDE_prob


function langevin_3d_tex_SDE_prob(
    ODEfun = langevin_3d_tex_loop_GPU;
    u0fun = (i) -> CUDA.randn(myT, N, M, 2),
    γ = 1.0f0,
    m2 = -1.0f0,
    λ = 1.0f0,
    J = 0.0f0,
    tspan = myT.((0.0, 15.0)),
    T = 5.0f0,
    tex = tex,
)
    # u0 = init_langevin_2d(xyd_brusselator)

    mσ = 600 / 197.33f0
    function g(du, u, p, t)
        # return du .= mσ*coth(mσ/(2*T))
        du[:, :, :, :, 1] .= 0.0f0
        du[:, :, :, :, 2] .= sqrt(mσ * coth(mσ / (2 * T)) * γ)
    end
    u0_GPU = 1.0f0
    function prob_func(prob, i, repeat)
        return remake(prob; u0 = u0fun(i))
    end
    output_func(sol, i) = (Array(sol), false)
    # output_func(sol, i) = i
    # output_func(sol, i) = (begin
    #     ar=Array(sol)[:,1,1,:]
    #     # mean(ar,dims=1)
    #     mσ=sum(ar.*weight_mean2.^2,dims=1) *4*pi/(V)
    #     varσ=sum((ar.-mσ).^2 .*weight_mean2.^2,dims=1) *4*pi/(V)
    #     kσ=sum((ar.-mσ).^4 .*weight_mean2.^2,dims=1) *4*pi./(V*varσ.^2)
    #     stack([mσ, varσ, kσ.-3])
    # end, false)
    p = myT.((γ, m2, λ, J))
    function ggprime(du, u, p, t)
        du .= 0.0f0 * u
    end
    ODEfun_tex(dσ, σ, p, t) = langevin_3d_tex_loop_GPU(dσ, σ, tex, p, t)
    sdefun = SDEFunction(ODEfun_tex, g; ggprime = ggprime)
    return EnsembleProblem(
        SDEProblem(sdefun, g, u0_GPU, tspan, p);
        prob_func = prob_func,
        output_func = output_func,
    )
end
export langevin_3d_tex_SDE_prob




function langevin_3d_SDE_prob(
    ODEfun = langevin_3d_loop_GPU;
    u0fun = (i) -> CUDA.randn(myT, N, M, 2),
    γ = 1.0f0,
    m2 = -1.0f0,
    λ = 1.0f0,
    J = 0.0f0,
    tspan = myT.((0.0, 15.0)),
    T = 5.0f0,
    para = para::TaylorParameters,
)
    # u0 = init_langevin_2d(xyd_brusselator)

    mσ = sqrt(para.λ1 + 2 * para.ρ0 * para.λ2)
    Ufun = funout(para)
    function g(du, u, p, t)
        # return du .= mσ*coth(mσ/(2*T))
        du[:, :, :, :, 1] .= 0.0f0
        du[:, :, :, :, 2] .= sqrt(mσ * coth(mσ / (2 * T)) * γ)
    end
    u0_GPU = 1.0f0
    function prob_func(prob, i, repeat)
        return remake(prob; u0 = u0fun(i))
    end
    output_func(sol, i) = (Array(sol), false)
    # output_func(sol, i) = i
    # output_func(sol, i) = (begin
    #     ar=Array(sol)[:,1,1,:]
    #     # mean(ar,dims=1)
    #     mσ=sum(ar.*weight_mean2.^2,dims=1) *4*pi/(V)
    #     varσ=sum((ar.-mσ).^2 .*weight_mean2.^2,dims=1) *4*pi/(V)
    #     kσ=sum((ar.-mσ).^4 .*weight_mean2.^2,dims=1) *4*pi./(V*varσ.^2)
    #     stack([mσ, varσ, kσ.-3])
    # end, false)
    p = myT.((γ, m2, λ, J))
    function ggprime(du, u, p, t)
        du .= 0.0f0 * u
    end
    ODEfun_tex(dσ, σ, p, t) = langevin_3d_loop_GPU(dσ, σ, Ufun, p, t)
    sdefun = SDEFunction(ODEfun_tex, g; ggprime = ggprime)
    return EnsembleProblem(
        SDEProblem(sdefun, g, u0_GPU, tspan, p);
        prob_func = prob_func,
        output_func = output_func,
    )
end
export langevin_3d_SDE_prob




function langevin_3d_SDE_Simple_prob(;
    u0 = error("u0 not provided"),
    v0 = error("v0 not provided"),
    γ = 1.0f0,
    tspan = myT.((0.0, 15.0)),
    T = 5.0f0,
    para = para::TaylorParameters,
    dt = 0.1f0,
    args...,
)
    # u0 = init_langevin_2d(xyd_brusselator)

    mσ = sqrt(abs(para.λ1) + 2 * para.ρ0 * para.λ2)
    # Ufun = funout_cut2(para)
    Ufun = funout(para)
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
    ODEfun_tex(dσ, σ) = langevin_3d_loop_simple_GPU(dσ, σ, Ufun)
    sdefun = SimpleSDEProblem(ODEfun_tex, v0, u0, tspan)
    println("noise=", sqrt(mσ * coth(mσ / (2 * T))))
    solve(
        sdefun,
        SimpleBAOABGPU(eta = γ, noise = sqrt(mσ * coth(mσ / (2 * T)) / 2));
        #SimpleBAOABGPU(eta = γ, noise = sqrt(T));
        # SimpleBAOABGPU(eta = γ, noise = 0.0f0);
        dt = dt,
        fun = Ufun,
        args...,
    )
end
export langevin_3d_SDE_Simple_prob

function modelA_3d_SDE_Simple_prob(;
    u0 = error("u0 not provided"),
    γ = 1.0f0,
    tspan = myT.((0.0, 15.0)),
    T = 5.0f0,
    para = para::TaylorParameters,
    dt = 0.1f0,
    noise="coth",
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

    mσ = sqrt(abs(para.λ1) + 2 * para.ρ0 * para.λ2)
    # Ufun = funout_cut2(para)
    Ufun = funout(para)
    # function Ufun(x)
    #     -x + x^3/6
    # end

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
    println("noise=", sqrt(mσ * coth(mσ / (2 * T))))

    function g1(du, u, p, t)
        du .= sqrt(mσ * coth(mσ / (2 * T))/γ)
    end
    function g2(du, u, p, t)
        du .= sqrt(2*T/γ)
    end
    u0_GPU = CuArray(u0)
    GC.gc(true)
    CUDA.reclaim()
    function ODEfun_tex(dσ, σ, p, t)
        CUDA.@sync langevin_3d_loop_simple_GPU(dσ, σ, Ufun,dx)
        CUDA.@sync dσ .= (1 / γ) .* dσ
    end
    #println("noise=", sqrt(T))
    if noise=="coth"
        sdeprob = SDEProblem(ODEfun_tex,g1, u0_GPU, tspan)
    elseif noise=="sqrt"
        sdeprob = SDEProblem(ODEfun_tex,g2, u0_GPU, tspan)
    end
    # sdeprob = SDEProblem(ODEfun_tex,g, u0_GPU, tspan,dx)
    saved_values = SavedValues(Float32, Any)
    cb = SavingCallback(
        (x, t, integrator) -> begin
            x1 = (mean(x, dims = [1, 2, 3])[1, 1, 1, :])
            # return var(x1)/T
            return mean(x)
            # x1
        end,
        saved_values;
        saveat=tspan[1]:0.05:tspan[2],
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
export modelA_3d_SDE_Simple_prob




function modelA_3d_SDE_Simple_prob_EM(;
    u0 = error("u0 not provided"),
    γ = 1.0f0,
    tspan = myT.((0.0, 15.0)),
    T = 5.0f0,
    para = para::TaylorParameters,
    dt = 0.1f0,
    dx=1,
    args...,
)
    Ufun = funout(para)
    GC.gc(true)
    CUDA.reclaim()
    function ODEfun_tex(dσ, σ)
        langevin_3d_loop_simple_GPU(dσ, σ, Ufun,dx)
    end
    #println("noise=", sqrt(T))
    sdefun = OverdampSDEProblem(ODEfun_tex, u0, tspan)
    # sdeprob = SDEProblem(ODEfun_tex,g, u0_GPU, tspan,dx)
    solve(
        sdefun,
        # PCEuler(ggprime),
        OverdampEM(;eta = γ, noise = sqrt(2*T));
        dt=dt,
        save_interval=0.05f0,
    )
end
export modelA_3d_SDE_Simple_prob_EM




function modelA_3d_SDE_Simple_tex_prob(;
    u0 = error("u0 not provided"),
    γ = 1.0f0,
    tspan = myT.((0.0, 15.0)),
    T = 5.0f0,
    dt = 0.1f0,
    x_grid,
    δUδσ_grid,
    noise=0.0f0,
    args...,
)
    println("noise=",noise)
    function g(du, u, p, t)
        #du[:, :, :, :] .= sqrt(ms*coth(ms/(2*T))/γ)
        du .=noise
    end
    u0_GPU = CuArray(u0)
    GC.gc(true)
    CUDA.reclaim()
    @show 197.33*T
    x_ini=x_grid[1]
    @show x_step=x_grid[2]-x_grid[1]
    δUδσ_grid_gpu=cu(δUδσ_grid)
    # Ufun = TexSpline1D(x_grid, δUδσ_grid_gpu)
    δUδσTextureMem = CuTextureArray(δUδσ_grid_gpu)
    δUδσTextureTex = CuTexture(δUδσTextureMem; interpolation = CUDA.CubicInterpolation())
    function ODEfun_tex(dσ, σ, p, t)
        CUDA.@sync langevin_3d_loop_simple_tex_GPU(dσ, σ, δUδσTextureTex,x_ini,x_step)
        CUDA.@sync dσ .= (1 / γ) .* dσ
    end
    sdeprob = SDEProblem(ODEfun_tex,g, u0_GPU, tspan)
    saved_values = SavedValues(Float32, Any)
    cb = SavingCallback(
        (x, t, integrator) -> begin
            x1 = Array(abs.(mean(x, dims = [1, 2, 3])[1, 1, 1, :]))
            return mean(x1)
        end,
        saved_values;
        # saveat=0.0:0.1:100.0,
        save_everystep = true,
        save_start = true,
    )
    solve(
        sdeprob,
        # PCEuler(ggprime),
        DRI1NM(),
        dt = dt,
        save_start = false,
        save_everystep = false,
        save_end = false,
        abstol = 1e-2,
        reltol = 1e-2,
        callback = cb,
    )
    [saved_values.t saved_values.saveval]
end
export modelA_3d_SDE_Simple_tex_prob






function langevin_3d_SDE_Simple_prob2(;
    u0 = error("u0 not provided"),
    v0 = error("v0 not provided"),
    γ = 1.0f0,
    tspan = myT.((0.0, 15.0)),
    T = 5.0f0,
    para = para::TaylorParameters,
    dt = 0.1f0,
    args...,
)
    # u0 = init_langevin_2d(xyd_brusselator)

    mσ = sqrt(abs(para.λ1) + 2 * para.ρ0 * para.λ2)
    # Ufun = funout_cut2_64(para)
    Ufun = funout(para)

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
    ODEfun_tex(dσ, σ) = langevin_3d_loop_simple_GPU(dσ, σ, Ufun)
    sdefun = SimpleSDEProblem(ODEfun_tex, v0, u0, tspan)
    println("noise=", sqrt(mσ * coth(mσ / (2 * T))))
    solve(
        sdefun,
        SimpleBAOABGPU_64(eta = γ, noise = 0.0 * sqrt(mσ * coth(mσ / (2 * T)) / 2));
        dt = dt,
        fun = Ufun,
        args...,
    )
end
export langevin_3d_SDE_Simple_prob2



function langevin_3d_SDE_Simple_prob3(;
    u0 = error("u0 not provided"),
    v0 = error("v0 not provided"),
    γ = 1.0f0,
    tspan = myT.((0.0, 15.0)),
    T = 5.0f0,
    para = para::TaylorParameters,
    dt = 0.1f0,
    args...,
)
    # u0 = init_langevin_2d(xyd_brusselator)

    mσ = sqrt(abs(para.λ1) + 2 * para.ρ0 * para.λ2)
    # Ufun = funout_cut2_64(para)
    Ufun = funout(para)
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
    ODEfun_tex(dσ, σ) = langevin_3d_loop_simple_GPU(dσ, σ, Ufun)
    sdefun = SimpleSDEProblem(ODEfun_tex, v0, u0, tspan)
    println("noise=", sqrt(mσ * coth(mσ / (2 * T))))
    solve(
        sdefun,
        SimpleBAOABGPU_64(eta = γ, noise = sqrt(mσ * coth(mσ / (2 * T)) / 2));
        dt = dt,
        fun = Ufun,
        args...,
    )
end
export langevin_3d_SDE_Simple_prob3

"""
```julia
u0_1 = fill(0.6f0, 32, 32, 32, 2^11)
v0_1 = fill(0.0f0, 32, 32, 32, 2^11)
langevin_3d_Ising_Simple_prob(;
    u0 = u0_1,
    v0 = v0_1,
    γ = 1.0f0,
    tspan = (0.0f0, 5.0f0),
    T = 5.0f0,
    dt = 0.1f0,
)
```
"""
function langevin_3d_Ising_Simple_prob(;
    u0 = error("u0 not provided"),
    v0 = error("v0 not provided"),
    γ = 1.0f0,
    tspan = myT.((0.0, 15.0)),
    T = 5.0f0,
    dt = 0.1f0,
    args...,
)

    #Ufun1(x) =0.1f0*x+x^3-0.1
    Tc = 5.0f0
    Ufun1(x) = x^3 + (T - Tc) * x
    Ufun(x) = Ufun1(x)
    ODEfun_tex(dσ, σ) = langevin_3d_loop_simple_GPU(dσ, σ, Ufun)
    sdefun = SimpleSDEProblem(ODEfun_tex, v0, u0, tspan)
    println("noise=", sqrt(T))
    function savefun(x::AbstractArray{T,4}) where {T}
        x1 = Array(abs.(mean(x, dims = [1, 2, 3])[1, 1, 1, :]))
        return mean(x1), var(x1)
    end
    solve(
        sdefun,
        SimpleBAOABGPUIsing(eta = γ, noise = sqrt(T));
        dt = dt,
        fun = Ufun,
        savefun = savefun,
        args...,
    )
end
export langevin_3d_Ising_Simple_prob



function modelA_3d_Ising_Simple_prob(;
    u0 = error("u0 not provided"),
    γ = 1.0f0,
    tspan = myT.((0.0, 15.0)),
    T = 5.0f0,
    dt = 0.1f0,
    args...,
)

    #Ufun1(x) =0.1f0*x+x^3-0.1
    Tc = 5.0f0
    Ufun1(x) = x^3 + (T - Tc) * x
    Ufun(x) = Ufun1(x)
    function g(du, u, p, t)
        du[:, :, :, :] .= sqrt(2 * γ * T)
    end
    u0_GPU = CuArray(u0)
    GC.gc(true)
    CUDA.reclaim()
    function ODEfun_tex(dσ, σ, p, t)
        CUDA.@sync langevin_3d_loop_simple_GPU(dσ, σ, Ufun)
        CUDA.@sync dσ .= (1 / γ) .* dσ
    end
    function ggprime(du, u, p, t)
        du .= 0.0f0 * u
    end
    sdefun = SDEFunction(ODEfun_tex, g; ggprime = ggprime)
    println("noise=", sqrt(T))
    sdeprob = SDEProblem(sdefun, u0_GPU, tspan)
    function savefun(x::AbstractArray{T,4}) where {T}
        x1 = Array(abs.(mean(x, dims = [1, 2, 3])[1, 1, 1, :]))
        return mean(x1), var(x1)
    end
    saved_values = SavedValues(Float32, Any)
    cb = SavingCallback(
        (x, t, integrator) -> begin
            # u_c=Array(u);
            # ϕ=[mean(u[:,:,i,1]) for i in 1:size(u)[3]];
            # return mean(ϕ)
            x1 = Array(abs.(mean(x, dims = [1, 2, 3])[1, 1, 1, :]))
            return mean(x1), var(x1)
        end,
        saved_values;
        # saveat=0.0:0.1:100.0,
        save_everystep = true,
        save_start = true,
    )
    solve(
        sdeprob,
        # PCEuler(ggprime),
        SRA3(),
        dt=dt,
        save_start = false,
        save_everystep = false,
        abstol = 2e-1,
        reltol = 2e-1,
        callback = cb,
    )
    saved_values
end
export modelA_3d_Ising_Simple_prob




function langevin_3d_ODE_prob(
    ODEfun = langevin_3d_loop_GPU;
    u0 = u0,
    du0 = du0,
    γ = 1.0f0,
    m2 = -1.0f0,
    λ = 1.0f0,
    J = 0.0f0,
    tspan = myT.((0.0, 15.0)),
    T = 5.0f0,
    para = para::TaylorParameters,
)
    # u0 = init_langevin_2d(xyd_brusselator)
    Ufun = funout(para)
    # output_func(sol, i) = i
    # output_func(sol, i) = (begin
    #     ar=Array(sol)[:,1,1,:]
    #     # mean(ar,dims=1)
    #     mσ=sum(ar.*weight_mean2.^2,dims=1) *4*pi/(V)
    #     varσ=sum((ar.-mσ).^2 .*weight_mean2.^2,dims=1) *4*pi/(V)
    #     kσ=sum((ar.-mσ).^4 .*weight_mean2.^2,dims=1) *4*pi./(V*varσ.^2)
    #     stack([mσ, varσ, kσ.-3])
    # end, false)
    p = myT.((γ, m2, λ, J))
    ODEfun_tex(ddσ, dσ, σ, p, t) = langevin_3d_leapfrog_loop_GPU(ddσ, dσ, σ, Ufun, p, t)
    # sdefun=SDEFunction(ODEfun_tex, g;ggprime=ggprime)
    SecondOrderODEProblem(ODEfun_tex, du0, u0, tspan, p)
end
export langevin_3d_ODE_prob
