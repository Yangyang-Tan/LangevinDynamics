# prob_ode_brusselator_2d_sparse_GPU = ODEProblem(brusselator_2d_loop_GPU, u0_GPU, tspan, p)

# prob_ode_brusselator_2d_sparse_SDE3 = EnsembleProblem(
#     SDEProblem(brusselator_2d_loop_GPU, g3, u0_GPU, tspan, p);
#     prob_func=prob_func,
#     output_func=output_func,
# )

function langevin_2d_ODE_prob(
    ODEfun=langevin_2d_loop_GPU;
    u0fun=(i)->CUDA.randn(myT, N, N, M),
    η=100.0,
    σ0=0.1,
    tspan=myT.((0.0, 15.0)),
)
    u0 = init_brusselator_2d(xyd_brusselator)
	u0_GPU = CuArray(u0)
	function prob_func(prob, i, repeat)
        return remake(prob; u0=u0fun(i))
    end
    output_func(sol, i) = (Array(sol), false)
	CUDA.reclaim()
	GC.gc(true)
    return EnsembleProblem(
        ODEProblem(ODEfun, u0_GPU, tspan, myT.((1 / η, step(xyd_brusselator), σ0)));
        prob_func=prob_func,
        output_func=output_func,
    )
	CUDA.reclaim
end
export langevin_2d_ODE_prob
