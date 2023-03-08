using DrWatson, Plots, DifferentialEquations,CUDA
@time @quickactivate :LangevinDynamics
langevin_2d_ODE_prob()


saved_values = SavedValues(Float32, Any)
# cb = SavingCallback(
#     (u, t, integrator) -> reshape(
#         mapslices(x -> cumulant.(Ref(abs.(x)), [1, 2, 3, 4]), Array(u); dims=[1, 2]),
#         (4, LangevinDynamics.M),
#     ),
#     saved_values;
#     # saveat=0.0:2.0:1500.0,
# )

cb = SavingCallback(
    (u, t, integrator) -> begin
        # u_c=Array(u);
        # ϕ=[mean(u[:,:,i,1]) for i in 1:size(u)[3]];
        # return mean(ϕ)
        return t
    end,
    saved_values;
    saveat=0.0:0.2:800.0,
)
function ggprime(du,u, p, t)
  du .= 0f0 *u
end
@time sol_1D_SDE = solve(
    langevin_3d_tex_SDE_prob(;
        γ = 2.2f0,
        m2 = -1.0f0,
        λ = 5.0f0,
        J = 0.0f0,
        tspan = (0.0f0, 50.0f0),
        T = 1.0f0,
        tex=δUδσTextureTex,
        # u0fun=x ->
        #     CUDA.fill(0.1f0, 64,64,64,2^7, 2),
        u0fun = x -> 0.1f0 *CUDA.randn(32,32,32,2^7, 2),
    ),
    [
        SOSRA(),
        SMEB(),
        SRA1(),
        SRA(),
        SOSRA2(),
        SOSRI2(),
        # SImplicitMidpoint(),
        # ImplicitRKMil(),
        SOSRI(),
        PCEuler(ggprime),
    ][end],
    EnsembleSerial();
    # dtmax = 0.01f0,
    # split=true,
    trajectories = 1,
    dt=0.02f0,
    # saveat = 0.0:0.1:10.0,
    save_everystep = false,
    save_start=false,
    # save_end=false,
    dense = false,
    save_on=false,
    initialize_save=false,
    calck=false,
    # callback=cb,
    abstol = 2e-0,
    reltol = 2e-0,
)



GC.gc(true)
CUDA.reclaim()
vec(sol_1D_SDE[1][:,:,:,:,1,1])
asdata=ash(vec(sol_1D_SDE[1][:,:,:,:,1,1]),m=5,rng=-2:0.01:2)
asdata[1]
plot(asdata2; hist=false)
plot!(asdata; hist=false)
mean(sol_1D_SDE[1][:,:,:,:,1,1],dims=[1,2,3])|>mean
0.707921/(mean(sol_1D_SDE[1][:,:,:,:,1,1],dims=[1,2,3])|>mean)
1/0.5^3

#######################

#data Input
lam1data=readdlm("data/eQCD_Input/buffer/lam1.dat")[:,1];
lam2data=readdlm("data/eQCD_Input/buffer/lam2.dat")[:,1];
lam3data=readdlm("data/eQCD_Input/buffer/lam3.dat")[:,1];
lam4data=readdlm("data/eQCD_Input/buffer/lam4.dat")[:,1];
lam5data=readdlm("data/eQCD_Input/buffer/lam5.dat")[:,1];
rho0data=readdlm("data/eQCD_Input/buffer/rho0.dat")[:,1];
lam3data[1:146].=0
lam4data[1:146].=0
lam5data[1:146].=0
cdata=3.904288549570293 ./sqrt.(readdlm("data/eQCD_Input/buffer/Zphi.dat")[:,1]);
taylordata=TaylorParameters(Float32,[lam1data lam2data lam3data lam4data lam5data rho0data cdata]);



@time sol_1D_SDE = solve(
    langevin_3d_SDE_prob(;
        γ = 0.5f0,
        m2 = -1.0f0,
        λ = 5.0f0,
        J = 0.0f0,
        tspan = (0.0f0, 400.0f0),
        T = 101.0f0/197.33f0,
        para=taylordata[100],
        # u0fun=x ->
        #     CUDA.fill(0.1f0, 64,64,64,2^7, 2),
        u0fun = x -> 0.1f0 *CUDA.randn(32,32,32,2^8, 2),
    ),
    [
        SOSRA(),
        SMEB(),
        SRA1(),
        SRA(),
        SOSRA2(),
        SOSRI2(),
        # SImplicitMidpoint(),
        # ImplicitRKMil(),
        SOSRI(),
        PCEuler(ggprime),
    ][end],
    EnsembleSerial();
    # dtmax = 0.01f0,
    # split=true,
    trajectories = 1,
    dt=0.01f0,
    # saveat = 0.0:0.1:10.0,
    save_everystep = false,
    save_start=false,
    # save_end=false,
    dense = false,
    save_on=false,
    initialize_save=false,
    calck=false,
    # callback=cb,
    abstol = 2e-0,
    reltol = 2e-0,
)
asdata=ash(vec(sol_1D_SDE[1][:,:,:,:,1,1]),m=5,rng=-1:0.01:1)
plot(asdata2; hist=false)
plot!(asdata; hist=false)
