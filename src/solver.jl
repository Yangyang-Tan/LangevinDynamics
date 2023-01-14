u0 = init_brusselator_2d(xyd_brusselator)
u0_GPU = CuArray(u0)
tspan = myT.((0.0, 1500.0))
prob_ode_brusselator_2d_sparse_GPU = ODEProblem(brusselator_2d_loop_GPU, u0_GPU, tspan, p)

function prob_func(prob, i, repeat)
	return remake(prob; u0 = CUDA.randn(myT, N, N, M))
end
output_func(sol, i) = (Array(sol), false)

prob_ode_brusselator_2d_sparse_SDE3 = EnsembleProblem(
	SDEProblem(brusselator_2d_loop_GPU, g3, u0_GPU, tspan, p); prob_func = prob_func,
	output_func = output_func,
)
ensemble_prob = EnsembleProblem(
	ODEProblem(
		brusselator_2d_loop_GPU, u0_GPU, tspan, myT.((0.02, step(xyd_brusselator), 0.1)),
	);
	prob_func = prob_func,
	output_func = output_func,
)
