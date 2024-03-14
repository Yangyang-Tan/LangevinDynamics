
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
    mσ = sqrt(abs(para.λ1) + 2 * para.ρ0 * para.λ2)
    # Ufun = funout_cut2(para)
    Ufun = funout(para)
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
    args...,
)
    # u0 = init_langevin_2d(xyd_brusselator)

    # mσ = sqrt(abs(para.λ1) + 2 * para.ρ0 * para.λ2)
    # Ufun = funout_cut2(para)
    # Ufun = funout(para)
    ODEfun_tex(dσ, σ) = langevin_3d_loop_simple_GPU(dσ, σ, Ufun)
    println("noise=", sqrt(mσ * coth(mσ / (2 * T))))
    function g(du, u, p, t)
        du[:, :, :, :] .= sqrt(2 * γ * T)
        #du .= sqrt(mσ *γ* coth(mσ / (2 * T)))
    end
    u0_GPU = CuArray(u0)
    GC.gc(true)
    CUDA.reclaim()
    function ODEfun_tex(dσ, σ, p, t)
        CUDA.@sync langevin_3d_loop_simple_GPU(dσ, σ, Ufun)
        CUDA.@sync dσ .= (1 / γ) .* dσ
    end
    #println("noise=", sqrt(T))
    sdeprob = SDEProblem(ODEfun_tex,g, u0_GPU, tspan)
    saved_values = SavedValues(Float32, Any)
    cb = SavingCallback(
        (x, t, integrator) -> begin
            # x1 = Array(abs.(mean(x, dims = [1, 2, 3])[1, 1, 1, :]))
            return mean(x)
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
export modelA_3d_SDE_Simple_prob


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
