using Distributed
using DrWatson
using CUDA
using DrWatson, Plots,DifferentialEquations,CUDA,DelimitedFiles,Random
addprocs(length(devices()))
@everywhere using DrWatson, Plots, DifferentialEquations,CUDA,DelimitedFiles,Random
@everywhere @time @quickactivate :LangevinDynamics
@time @quickactivate :LangevinDynamics
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
       ϕ=mean(u, dims = [1, 2, 3])[:, 1]
        return mean(abs.(ϕ))
    end,
    saved_values;
    # saveat=0.0:0.2:100.0,
    save_everystep=true,
    save_start = true
)
function ggprime(du,u, p, t)
  du .= 0f0
end
@time sol_1D_SDE = solve(
    langevin_3d_tex_SDE_prob(;
        γ = 0.3f0,
        m2 = -1.0f0,
        λ = 1.0f0,
        J = 0.0f0,
        tspan = (0.0f0, 10.0f0),
        T = 1.0f0,
        tex=δUδσTextureTex,
        # u0fun=x ->
        #     CUDA.fill(0.1f0, 64,64,64,2^7, 2),
        u0fun = x -> 0.1f0 *CUDA.randn(32,32,32,2^7, 2).+1.0f0,
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
lam3data[1:147].=0
lam4data[1:147].=0
lam5data[1:147].=0
cdata=3.904288549570293 ./sqrt.(readdlm("data/eQCD_Input/buffer/Zphi.dat")[:,1]);
taylordata=TaylorParameters(Float32,[lam1data lam2data lam3data lam4data lam5data rho0data cdata]);
Base.Flatten
cumulant
saved_values = SavedValues(Float32, Any)

cb = SavingCallback(
    (u, t, integrator) -> begin
        # u_c=Array(u);
        # ϕ=[mean(u[:,:,i,1]) for i in 1:size(u)[3]];
        # return mean(ϕ)
        vu=@view u[:,:,:,:,1]
        # c4=mapslices(x->cumulant(vec(cpuvu),4),cpuvu,dims=[1,2,3])
        # dvu=dropdims(mean(vu,dims=[1,2,3]),dims=(1,2,3))
        return mean(abs.(dropdims(var(vu,dims=[1,2,3]),dims=(1,2,3))))
    end,
    saved_values;
    # saveat=0.0:0.2:100.0,
    save_everystep=true,
    save_start = true
)

cb = SavingCallback(
    (u, t, integrator) -> begin
        # u_c=Array(u);
        # ϕ=[mean(u[:,:,i,1]) for i in 1:size(u)[3]];
        # return mean(ϕ)
        vu=@view u[:,:,:,:,1]
        dvu=dropdims(mean(vu,dims=[1,2,3]),dims=(1,2,3))
        return cumulant(dvu,1)
    end,
    saved_values;
    # saveat=0.0:0.2:100.0,
    save_everystep=true,
    save_start = true
)
[CUDA.fill(1f0, 32,32,32,2^9,1)]


@time sol_1D_SDE = solve(
    langevin_3d_SDE_prob(;
        γ = 450.0f0,
        m2 = -1.0f0,
        λ = 5.0f0,
        J = 0.0f0,
        tspan = (0.0f0, 15.0f0),
        T = 170.0f0/197.33f0,
        para=taylordata[169],
        u0fun=x ->cat(CUDA.fill(0.6f0,32,32,32,2^5,1),CUDA.fill(0f0, 32,32,32,2^5,1),dims=5),
        # u0fun = x -> 0.1f0*CUDA.randn(32,32,32,2^9, 2),
    ),
    [
    SimplifiedEM(),
    EM(),
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
    ][7],
    EnsembleSerial();
    # dtmax = 0.01f0,
    # split=true,
    trajectories = 1,
    # dt=0.05f0,
    saveat = 0.0:0.2:15.0,
    save_everystep = false,
    save_start=true,
    # save_end=false,
    # dense = false,
    # save_on=false,
    # initialize_save=false,
    calck=false,
    callback=cb,
    abstol = 1e-0,
    reltol = 1e-0,
)

plot!(saved_values.t[1:end],saved_values.saveval[1:end])



u0_1 = fill(0.6f0, 32, 32, 32, 2^15)
v0_1 = fill(0f0, 32, 32, 32, 2^15)
CUDA.@time sol_3D_SDE = langevin_3d_SDE_Simple_prob(;
    γ = 2.0f0,
    T = 170.0f0 / 197.33f0,
    para = taylordata[1],
    u0=u0_1,
    v0=v0_1,
    tspan=(0f0,20f0),
    dt=0.1f0,
    # u0fun = x -> 0.1f0*CUDA.randn(32,32,32,2^9, 2),
)
GC.gc(true)
CUDA.reclaim()

4*3/1000
@btime CUDA.@sync lmul!(2f0, $u0_1)

@btime CUDA.@sync axpy!(1.0f0, $v0_1, $u0_1)

14.263*6

sol_1D_SDE[1]
asdata=ash(vec(sol_1D_SDE[end][:,:,:,:,1,2]),m=5,rng=-0.5:0.005:0.5)
asdata2 = ash(vec(Array(u0_1)), m = 5, rng = -0.5:0.005:0.5)
plot!(asdata2; hist=false)
plot(asdata; hist=false)
plot(asdata.rng,asdata.density)
collect(asdata.rng)
writedlm("sims/mub=0/dis/T=150_ini=0.6_uniform_gam=2.0_t=0.2.dat",[collect(asdata.rng) asdata.density])

(CUDA.randn(32,32,32,2^8, 2)).*1f0
saved_values = SavedValues(Float32, Any)
plot!(saved_values.t[1:end],saved_values.saveval[1:end])


mphi=mean(abs.(mean(sol_1D_SDE[1][:,:,:,:,1,1],dims=[1,2,3])))


funvar=Spline1D(saved_values.t,saved_values.saveval .-saved_values.saveval[end])

plot(x->funvar(x),4,6,yaxis=:log)

plot(tdata,ydata,yaxis=:log)
plot(saved_values.t[1000:4000],-(saved_values.saveval.-mphi)[1000:4000],yaxis=:log)

(fita,fitb)=exp_fit(4:0.1:6, funvar.(4:0.1:6))
1/fitb
1/fitb

1/fitb

plot!(x->fita*exp(fitb*x),30,80,yaxis=:log)

@tagsave(datadir("sims/mub=0/", savename((@strdict Nx NM), "dat")), (@strdict tempv))
writedlm("sims/mub=0/T=100_ini=0.6_uniform_gam=2.0_mean.dat",[saved_values.t[1:end] saved_values.saveval[1:end]])
