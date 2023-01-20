# prob_ode_brusselator_2d_sparse_GPU = ODEProblem(brusselator_2d_loop_GPU, u0_GPU, tspan, p)

# prob_ode_brusselator_2d_sparse_SDE3 = EnsembleProblem(
#     SDEProblem(brusselator_2d_loop_GPU, g3, u0_GPU, tspan, p);
#     prob_func=prob_func,
#     output_func=output_func,
# )

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
        ODEProblem(ODEfun, u0_GPU, tspan, myT.((1 / η, step(xyd_brusselator), σ0, h)));
        prob_func=prob_func,
        output_func=output_func,
    )
end
export langevin_2d_ODE_prob

function langevin_2d_SDE_prob(
    ODEfun=langevin_2d_loop_GPU;
    u0fun=(i) -> CUDA.randn(myT, N, N, M),
    η=10.0,
    σ0=0.1,
    h=0.0,
    tspan=myT.((0.0, 15.0)),
)
    # u0 = init_langevin_2d(xyd_brusselator)
    function g(du, u, p, t)
        return du .= 0.05
    end
    u0_GPU = 1.0f0
    function prob_func(prob, i, repeat)
        return remake(prob; u0=u0fun(i))
    end
    output_func(sol, i) = (Array(sol), false)
    p=myT.((1 / η, step(xyd_brusselator), σ0, h))
    return EnsembleProblem(
        SDEProblem(ODEfun, g, u0_GPU, tspan, p);
        prob_func=prob_func,
        # output_func=output_func,
    )
end
export langevin_2d_SDE_prob
