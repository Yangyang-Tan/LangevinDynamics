
function langevin_2d_ODE_prob(
    ODEfun=langevin_2d_loop_GPU;
    u0fun=(i) -> CUDA.randn(myT, N, N, M),
    η=10.0,
    σ0=0.1,
    h=0.0,
    tspan=myT.((0.0, 15.0)),
)
    # u0 = init_langevin_2d(xyd_brusselator)
    u0_GPU = 1.0f0
    function prob_func(prob, i, repeat)
        return remake(prob; u0=u0fun(i))
    end
    output_func(sol, i) = (Array(sol), false)
    return EnsembleProblem(
        ODEProblem(ODEfun, u0_GPU, tspan, myT.((1 / η, σ0, h, step(xyd_brusselator))));
        prob_func=prob_func,
        output_func=output_func,
    )
end
export langevin_2d_ODE_prob

function langevin_2d_SDE_prob(
    ODEfun=langevin_2d_loop_GPU;
    u0fun=(i) -> CUDA.randn(myT, N, N, M,2),
    γ=1.0f0,
    m2=-1.0f0,
    λ=1.0f0,
    J=0.0f0,
    tspan=myT.((0.0, 15.0)),
    T=5.0f0,
)
    # u0 = init_langevin_2d(xyd_brusselator)
    function g(du, u, p, t)
        du[:,:,:,1] .= 0.0f0
        du[:,:,:,2] .= sqrt(2 * T*γ)
    end
    u0_GPU = 1.0f0
    function prob_func(prob, i, repeat)
        return remake(prob; u0=u0fun(i))
    end
    output_func(sol, i) = (Array(sol), false)
    # W = WienerProcess(0.0,0.0,0.0)
    p = myT.((γ, m2, λ, J))
    return EnsembleProblem(
        SDEProblem(ODEfun, g, u0_GPU, tspan, p);
        prob_func=prob_func,
        output_func=output_func,
    )
end
export langevin_2d_SDE_prob

function langevin_1d_SDE_prob(
    ODEfun=langevin_1d_loop_GPU;
    u0fun=(i) -> CUDA.randn(myT, N,M,2),
    γ=1.0f0,
    m2=-1.0f0,
    λ=1.0f0,
    J=0.0f0,
    tspan=myT.((0.0, 15.0)),
    T=5.0f0,
)
    # u0 = init_langevin_2d(xyd_brusselator)
    function g(du, u, p, t)
        # return du .= mσ*coth(mσ/(2*T))
        du[:,:,1] .= 0.0f0
        du[:,:,2] .= sqrt(2 * T*γ)
    end
    u0_GPU = 1.0f0
    function prob_func(prob, i, repeat)
        return remake(prob; u0=u0fun(i))
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
        prob_func=prob_func,
        output_func=output_func,
    )
end
export langevin_1d_SDE_prob


function langevin_1d_tex_SDE_prob(
    ODEfun=langevin_1d_tex_loop_GPU;
    u0fun=(i) -> CUDA.randn(myT, N,M,2),
    γ=1.0f0,
    m2=-1.0f0,
    λ=1.0f0,
    J=0.0f0,
    tspan=myT.((0.0, 15.0)),
    T=5.0f0,
    tex=tex,
)
    # u0 = init_langevin_2d(xyd_brusselator)
    function g(du, u, p, t)
        # return du .= mσ*coth(mσ/(2*T))
        du[:,:,1] .= 0.0f0
        du[:,:,2] .= sqrt(2 * T*γ)
    end
    u0_GPU = 1.0f0
    function prob_func(prob, i, repeat)
        return remake(prob; u0=u0fun(i))
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
    ODEfun_tex(dσ, σ,p,t)=langevin_1d_tex_loop_GPU(dσ, σ,tex, p, t)
    return EnsembleProblem(
        SDEProblem(ODEfun_tex, g, u0_GPU, tspan, p);
        prob_func=prob_func,
        output_func=output_func,
    )
end
export langevin_1d_tex_SDE_prob


function langevin_0d_SDE_prob(
    ODEfun=langevin_0d_loop_GPU;
    u0fun=(i) -> CUDA.randn(myT, N,M,2),
    γ=1.0f0,
    m2=-1.0f0,
    λ=1.0f0,
    J=0.0f0,
    tspan=myT.((0.0, 15.0)),
    T=5.0f0,
)
    # u0 = init_langevin_2d(xyd_brusselator)
    function g(du, u, p, t)
        # return du .= mσ*coth(mσ/(2*T))
        du[:,:,1] .= 0.0
        du[:,:,2] .= sqrt(2 * T*γ)
    end
    u0_GPU = 1.0f0
    function prob_func(prob, i, repeat)
        return remake(prob; u0=u0fun(i))
    end
    output_func(sol, i) = (Array(sol), false)
    p = myT.((γ, m2, λ, J))
    # W = WienerProcess(0.0,0.0,0.0)
    return EnsembleProblem(
        SDEProblem(ODEfun, g, u0_GPU, tspan, p);
        prob_func=prob_func,
        output_func=output_func,
    )
end
export langevin_0d_SDE_prob



function langevin_0d_tex_SDE_prob(
    ODEfun=langevin_0d_tex_loop_GPU;
    u0fun=(i) -> CUDA.randn(myT, N,M,2),
    γ=1.0f0,
    m2=-1.0f0,
    λ=1.0f0,
    J=0.0f0,
    tspan=myT.((0.0, 15.0)),
    T=5.0f0,
    tex=tex,
)
    # u0 = init_langevin_2d(xyd_brusselator)
    function g(du, u, p, t)
        # return du .= mσ*coth(mσ/(2*T))
        du[:,:,1] .= 0.0
        du[:,:,2] .= sqrt(2 * T*γ)
    end
    u0_GPU = 1.0f0
    function prob_func(prob, i, repeat)
        return remake(prob; u0=u0fun(i))
    end
    output_func(sol, i) = (Array(sol), false)
    p = myT.((γ, m2, λ, J))
    # W = WienerProcess(0.0,0.0,0.0)
    ODEfun_tex(dσ, σ,p,t)=langevin_0d_tex_loop_GPU(dσ, σ,tex, p, t)
    return EnsembleProblem(
        SDEProblem(ODEfun_tex, g, u0_GPU, tspan, p);
        prob_func=prob_func,
        output_func=output_func,
    )

end
export langevin_0d_tex_SDE_prob

# function langevin_iso_SDE_prob(
#     ODEfun=langevin_iso_loop_GPU;
#     u0fun=(i) -> CUDA.randn(myT, N,M,2),
#     γ=1.0f0,
#     m2=-1.0f0,
#     λ=1.0f0,
#     J=0.0f0,
#     tspan=myT.((0.0, 15.0)),
#     T=5.0f0,
#     d=3.0f0,
# )
#     # u0 = init_langevin_2d(xyd_brusselator)
#     weight_mean2=0.1:0.1:1024*0.1
#     V=sum(weight_mean2.^2) *4*pi
#     ξ=myT(sqrt(γ*sqrt(abs(m2))*coth(sqrt(abs(m2))/(2*T))/V))
#     @show ξ
#     function g(du, u, p, t)
#         du[:,:,1] .= 0.0f0
#         du[:,:,2] .= ξ
#     end
#     u0_GPU = 1.0f0
#     function prob_func(prob, i, repeat)
#         return remake(prob; u0=u0fun(i))
#     end
#     output_func(sol, i) = (begin
#         ar=Array(sol)[:,1,1,:]
#         # mean(ar,dims=1)
#         mσ=sum(ar.*weight_mean2.^2,dims=1) *4*pi/(V)
#         varσ=sum((ar.-mσ).^2 .*weight_mean2.^2,dims=1) *4*pi/(V)
#         kσ=sum((ar.-mσ).^4 .*weight_mean2.^2,dims=1) *4*pi./(V*varσ.^2)
#         stack([mσ, varσ, kσ.-3])
#     end, false)
#     # W = WienerProcess(0.0,0.0,0.0)
#     p = myT.((d,γ, m2, λ, J))
#     W = WienerProcess(0.0, 0.0, 0.0)
#     return EnsembleProblem(
#         SDEProblem(ODEfun, g, u0_GPU, tspan, p,noise = W);
#         prob_func=prob_func,
#         output_func=output_func,
#     )
# end
# export langevin_iso_SDE_prob
global ct=0

function langevin_3d_tex_SDE_prob(
    ODEfun=langevin_3d_tex_loop_GPU;
    u0fun=(i) -> CUDA.randn(myT, N,M,2),
    γ=1.0f0,
    m2=-1.0f0,
    λ=1.0f0,
    J=0.0f0,
    tspan=myT.((0.0, 15.0)),
    T=5.0f0,
    tex=tex,
)
    # u0 = init_langevin_2d(xyd_brusselator)

    mσ = 600/197.33f0
    function g(du, u, p, t)
        # return du .= mσ*coth(mσ/(2*T))
        du[:,:,:,:,1] .= 0.0f0
        du[:,:,:,:,2] .= sqrt(mσ*coth(mσ/(2*T))*γ)
    end
    u0_GPU = 1.0f0
    function prob_func(prob, i, repeat)
        return remake(prob; u0=u0fun(i))
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
    function ggprime(du,u, p, t)
        du .= 0f0 *u
    end
    ODEfun_tex(dσ, σ,p,t)= langevin_3d_tex_loop_GPU(dσ, σ,tex, p, t)
    sdefun=SDEFunction(ODEfun_tex, g;ggprime=ggprime)
    return EnsembleProblem(
        SDEProblem(sdefun,g, u0_GPU, tspan, p);
        prob_func=prob_func,
        output_func=output_func,
    )
end
export langevin_3d_tex_SDE_prob




function langevin_3d_SDE_prob(
    ODEfun=langevin_3d_loop_GPU;
    u0fun=(i) -> CUDA.randn(myT, N,M,2),
    γ=1.0f0,
    m2=-1.0f0,
    λ=1.0f0,
    J=0.0f0,
    tspan=myT.((0.0, 15.0)),
    T=5.0f0,
    para=para::TaylorParameters,
)
    # u0 = init_langevin_2d(xyd_brusselator)

    mσ = sqrt(para.λ1+2*para.ρ0 *para.λ2)
    Ufun=funout(para)
    function g(du, u, p, t)
        # return du .= mσ*coth(mσ/(2*T))
        du[:,:,:,:,1] .= 0.0f0
        du[:,:,:,:,2] .= sqrt(mσ*coth(mσ/(2*T))*γ)
    end
    u0_GPU = 1.0f0
    function prob_func(prob, i, repeat)
        return remake(prob; u0=u0fun(i))
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
    function ggprime(du,u, p, t)
        du .= 0f0 *u
    end
    ODEfun_tex(dσ, σ,p,t)= langevin_3d_loop_GPU(dσ, σ,Ufun, p, t)
    sdefun=SDEFunction(ODEfun_tex, g;ggprime=ggprime)
    return EnsembleProblem(
        SDEProblem(sdefun,g, u0_GPU, tspan, p);
        prob_func=prob_func,
        output_func=output_func,
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
    dt=0.1f0,
    args...,
)
    # u0 = init_langevin_2d(xyd_brusselator)

    mσ = sqrt(abs(para.λ1) + 2 * para.ρ0 * para.λ2)
    Ufun = funout_cut2(para)
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
        SimpleBAOABGPU(eta = γ, noise = sqrt(mσ * coth(mσ / (2 * T))/2));
        dt = dt,
        fun=Ufun,
        args...,
    )
end
export langevin_3d_SDE_Simple_prob


function langevin_3d_Ising_Simple_prob(;
    u0 = error("u0 not provided"),
    v0 = error("v0 not provided"),
    γ = 1.0f0,
    tspan = myT.((0.0, 15.0)),
    T = 5.0f0,
    para = para::TaylorParameters,
    dt = 0.1f0,
    args...,
)

    Ufun(x)=-x/2 +x^3/3
    ODEfun_tex(dσ, σ) = langevin_3d_loop_simple_GPU(dσ, σ, Ufun)
    sdefun = SimpleSDEProblem(ODEfun_tex, v0, u0, tspan)
    println("noise=", sqrt(T))
    solve(
        sdefun,
        SimpleBAOABGPUIsing(eta = γ, noise = sqrt(T));
        dt = dt,
        fun=Ufun,
        args...,
    )
end
export langevin_3d_Ising_Simple_prob


function langevin_3d_ODE_prob(
    ODEfun=langevin_3d_loop_GPU;
    u0=u0,
    du0=du0,
    γ=1.0f0,
    m2=-1.0f0,
    λ=1.0f0,
    J=0.0f0,
    tspan=myT.((0.0, 15.0)),
    T=5.0f0,
    para=para::TaylorParameters,
)
    # u0 = init_langevin_2d(xyd_brusselator)
    Ufun=funout(para)
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
    ODEfun_tex(ddσ,dσ, σ,p,t)= langevin_3d_leapfrog_loop_GPU(ddσ,dσ, σ,Ufun, p, t)
    # sdefun=SDEFunction(ODEfun_tex, g;ggprime=ggprime)
    SecondOrderODEProblem(ODEfun_tex, du0,u0, tspan, p)
end
export langevin_3d_ODE_prob
