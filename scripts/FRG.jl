saved_values = SavedValues(Float32, Any)
cb = SavingCallback(
    (u, t, integrator) -> begin
        # u_c=Array(u);
        # ϕ1=[mean(u[:,:,i,1]) for i in 1:size(u)[3]];
        # ϕ2=[mean(u[:,:,i,2]) for i in 1:size(u)[3]];
        return Array(sum(u[:,:,1].*weight_mean.^2 *4*pi,dims=1))
    end,
    saved_values;
    saveat=0.0:0.5:400.0,
)
weight_mean=cu(stack(collect.(fill(1:1024,LangevinDynamics.M))))

sol_SDE = solve(
    langevin_iso_SDE_prob(;
        γ=0.2f0,
        m2=-1.0f0,
        λ=1.0f0,
        J=0.0f0,
        tspan=(0.0f0, 125.0f0),
        T=0.1f0,
        # u0fun=x ->
        #     CUDA.fill(0.0f0, LangevinDynamics.N,2),
        u0fun=x ->0.0f0 .+0.1f0*CUDA.randn(LangevinDynamics.N,LangevinDynamics.M,2),
    ),
    [SOSRA(),ImplicitEM(),SImplicitMidpoint(),ImplicitRKMil(),SKSROCK(),SRA3(),SOSRI()][end],
    EnsembleSerial();
    dtmax=0.02,
    trajectories=1,
    dt=0.01f0,
    saveat=0.0:0.5:125.0,
    save_everystep=false,
    # save_start=true,
    # save_end=false,
    dense=false,
    # callback=cb,
    abstol=1e-2,
    reltol=1e-2,
)


stack(sol_SDE)

sum(u[:,:,1].*weight_mean*4*pi,dims=1)
sol_SDE[1]
plot(sol_SDE[1][1,:])
plot(mean(stack(sol_SDE)[:,:,3,:],dims=3)[1,:,1])

plot(mean(stack(saved_values.saveval),dims=[2])[1,1,:])


stack(saved_values.saveval)[:,1,1]
stack(saved_values.saveval)[:,1,2]

plot(stack(saved_values.saveval)[:,1,101])

plot(mean(abs.(stack(saved_values.saveval)[:,1,:]),dims=[1])[1,:])


plot(stack(saved_values.saveval)[2,1,:])


mean(stack(saved_values.saveval)[:,1,:],dims=[1])
saved_values.saveval[1]
