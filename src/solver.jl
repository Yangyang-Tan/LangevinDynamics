
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
    T=4.3f0,
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
    u0fun=(i) -> CUDA.randn(myT, N, M),
    η=10.0,
    σ0=0.1,
    h=0.0,
    tspan=myT.((0.0, 15.0)),
    g0=0.5f0,
)
    # u0 = init_langevin_2d(xyd_brusselator)
    function g(du, u, p, t)
        # return du .= mσ*coth(mσ/(2*T))
        return du .= sqrt(2 * T / γ)
    end
    u0_GPU = 1.0f0
    function prob_func(prob, i, repeat)
        return remake(prob; u0=u0fun(i))
    end
    output_func(sol, i) = (Array(sol), false)
    p = myT.((η, σ0, h, step(xyd_brusselator)))
    return EnsembleProblem(
        SDEProblem(ODEfun, g, u0_GPU, tspan, p);
        prob_func=prob_func,
        output_func=output_func,
    )
end
export langevin_1d_SDE_prob
