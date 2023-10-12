using DrWatson, Plots, DifferentialEquations,CUDA
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
# cumulant(tempv,2)
cb = SavingCallback(
    (u, t, integrator) -> begin
        # u_c=Array(u);
        ϕ=abs.([mean(u[:,:,i,1]) for i in 1:size(u)[3]]);
        return [mean(ϕ), var(ϕ)]
    end,
    saved_values;
    saveat=0.0:1.0:50.0,
)

phiconfig_T50=sol_SDE[1][:, :, :, :, 2]

@tagsave(datadir("sim", "phiconfig_T50.jld2"), @strdict phiconfig_T50)




@time sol_SDE = solve(
    langevin_2d_SDE_prob(;
        γ=0.3f0,
        m2=-1.0f0,
        λ=1.0f0,
        J=0.0f0,
        tspan=(0.0f0, 50.0f0),
        T=50.0f0,
        # u0fun=x ->
        #     CUDA.fill(5.0f0, LangevinDynamics.N, LangevinDynamics.N, LangevinDynamics.M,2),
        u0fun=x ->1.0f0.+CUDA.randn(64,64,2048,2),
    ),
    [SOSRA(),ImplicitEM(),SImplicitMidpoint(),ImplicitRKMil(),SKSROCK(),SRA3(),SOSRI()][1],
    EnsembleSerial();
    dtmax=0.1,
    trajectories=1,
    # dt=0.1f0,
    # saveat=0.0:0.2:8.0,
    save_everystep=false,
    save_start=true,
    save_end=true,
    dense=false,
    callback=cb,
    abstol=1e-1,
    reltol=1e-1,
)


plot(saved_values.t, stack(saved_values.saveval)[1, :])



function getM(T)
    saved_values = SavedValues(Float32, Any)
    cb = SavingCallback(
    (u, t, integrator) -> begin
        # u_c=Array(u);
        # ϕ1=[mean(u[:,:,i,1]) for i in 1:size(u)[3]];
        # ϕ2=[mean(u[:,:,i,2]) for i in 1:size(u)[3]];
        return Array(dropdims(mean(u,dims=[1,2]),dims=(1,2)))
    end,
    saved_values;
    saveat=1000.0:0.02:1400.0,
)
sol_SDE = solve(
    langevin_2d_SDE_prob(;
        γ=0.1f0,
        m2=-1.0f0,
        λ=1.0f0,
        J=0.0f0,
        tspan=(0.0f0, 1400.0f0),
        T=T,
        # u0fun=x ->
        #     CUDA.fill(5.0f0, LangevinDynamics.N, LangevinDynamics.N, LangevinDynamics.M,2),
        u0fun=x ->1.0f0.+CUDA.randn(LangevinDynamics.N, LangevinDynamics.N, LangevinDynamics.M,2),
    ),
    [SOSRA(),ImplicitEM(),SImplicitMidpoint(),ImplicitRKMil(),SKSROCK(),SRA3(),SOSRI()][1],
    EnsembleSerial();
    dtmax=0.02,
    trajectories=1,
    # dt=0.1f0,
    # saveat=0.0:0.2:8.0,
    save_everystep=false,
    save_start=true,
    save_end=false,
    dense=false,
    callback=cb,
    abstol=1e-1,
    reltol=1e-1,
)
println("T=", T)
return saved_values.saveval
end

Nx=LangevinDynamics.N
NM=LangevinDynamics.M
Trng=[0.5f0]
tempv=getM.(Trng)
tempv3=getM.(Trng)
tempv4=getM.(Trng)
tempv_T05=getM.(Trng)
@tagsave(datadir("sims", savename((@strdict Nx NM), "jld2")), (@strdict tempv))


saved_values.saveval[1]
temash = ash(vec(sol_GPU[:, :, :, 1, :]); kernel=Kernels.gaussian, m=5, rng=-0.5:0.001:0.5)
plot([mean(sol_GPU[:, :, :, i, :]) for i in 1:50])
plot([mean(sol_SDE[:, :, :, i, :]) for i in 1:50])
heatmap(sol_SDE[:, :,1, 1, 2,1],size=(512,512),aspect_ratio=1,right_margin = 8Plots.mm)
plot(log.(abs.(map(x->mapslices(mean,x,dims=2)[1],saved_values.saveval)[10:end])))
plot!(saved_values.saveval[100:end])
vcat(saved_values.saveval...)
saved_values.saveval
sol_SDE[:,:,:,:,:,:]
mean(saved_values.saveval.^2)-mean(saved_values.saveval)^2
mean(saved_values.saveval.^2)-mean(saved_values.saveval)^2
mean(saved_values.saveval.^2)-mean(saved_values.saveval)^2
mean(saved_values.saveval[2500:end].^2)-mean(saved_values.saveval[2500:end])^2
mean(saved_values.saveval[2500:end].^2)-mean(saved_values.saveval[2500:end])^2
ini_SDE.=cu(sol_SDE[:, :, :, :, 2,1])
ini_SDE|>Array
plot([tempv[5][i][7] for i in 1:length(tempv)])
plot(Trng,[mean(mean.(tempv[i])[1000:end]) for i in 1:61])
plot(Trng,[mean((mean.(tempv[i])[8000:end]).^2)-mean(mean.(tempv[i])[8000:end])^2 for i in 1:19])
tempv[1][:,1]
mean((mean.(tempv[1])[8000:end]).^2)-mean(mean.(tempv[1])[8000:end])^2

plot(Trng,[mean(hcat(tempv[i]...)[1,5000:end].^2)-mean(hcat(tempv[i]...)[1,5000:end])^2 for i in 1:61]./Trng)

chidata=mean([[mean(hcat(tempv[i]...)[j,500:end].^2)-mean(hcat(tempv[i]...)[j,500:end])^2 for i in 1:61] for j in 1:64])./Trng
chidata2=mean([[mean(hcat(tempv2[i]...)[j,5000:end].^2)-mean(hcat(tempv2[i]...)[j,5000:end])^2 for i in 1:19] for j in 1:64])./Trng2
Trng2=collect(1.5f0:0.05f0:2.4f0)
plot!(Trng,128^2*chidata,seriestype=:scatter,label="128×128",xlabel="T",ylabel="χ")
plot(Trng2,64^2*chidata2,seriestype=:scatter,label="64×64",xlabel="T",ylabel="χ")
tempv2

chidata2

plot(log.((2.0.-Trng[1:30])./2.0),log.(chidata[1:30]))

plot!(x->chia+chib*x,-4.5,-1.0)

chia,chib=linear_fit(log.((2.0.-Trng[1:30])./2.0),log.(chidata[1:30]))



plot(Trng2,[mean(mean.(tempv2[i])[5000:end]) for i in 1:19],seriestype=:scatter,label="64×64",xlabel="T",ylabel="M")

plot!(Trng,[mean(mean.(tempv[i])[100:end]) for i in 1:61],seriestype=:scatter,label="128×128")
tempv2=wload(datadir("sims", "NM=64_Nx=64.jld2"))["tempv"]


plot([mean(tempv[1][t][2].*tempv[1][1][1]-tempv[1][1][2].*tempv[1][t][1]) for t in 1:10001])

1
