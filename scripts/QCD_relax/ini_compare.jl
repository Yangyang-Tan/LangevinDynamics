using TOML
config = TOML.parsefile("sims/ini_compare/config.toml")
solver_symbol = Symbol(config["solver"]["name"])
solver_constructor = eval(Expr(:call, solver_symbol))


u0_1 = CUDA.fill(0.6f0, 128, 128, 128, 2^6);


device!(1)
using CUDA

u0_1 = CUDA.fill(Float16, 0.1f0, 12, 32, 32, 32, 4096);



u0_1 = CUDA.randn(32, 32, 32, 2^16) .+ 0.6f0;

u0_1 = CUDA.fill(Float32(-0.6f0), 32, 32, 32, 2^6);
u0_1 = 0;
@everywhere include("/home/tyy/LangevinDynamics/scripts/QCD_relax/load.jl")
@everywhere qcdmodel = eqcd_potential_dataloader(63)[46]

##############
#compare the initial condition
##############


qcdmodel.U

TaylorParameters([0.0f0 0.5f0 0.0f0 0.0f0 0.0f0 1.0f0 0.0f0])[1]
u0_1 = CUDA.fill(-0.1f0, 32, 32, 32, 2^10);
@time sol_3D_SDE_2 = modelA_3d_SDE_Simple_prob(;
    solver = DRI1NM(),
    γ = 4.0f0,
    T = 1.0f0 * qcdmodel.T,
    para = qcdmodel.U,
    u0 = u0_1,
    tspan = (0.0f0, 50.0f0),
    # save_end=true,
    # dt = 0.02f0,
    dx = 0.1f0,
    noise = "sqrt",
    abstol = 5e-2,
    reltol = 5e-2,
    # dt=0.01f0,
)
plot(LangevinDynamics.Uσfunout(qcdmodel.U), -0.6, 0.6)

plot!(sol_3D_SDE_2[:, 1], sol_3D_SDE_2[:, 2])
sol_3D_SDE_2[end]




#############################
# const_ini
#############################
u0_1 = CUDA.fill(-0.1f0, 32, 32, 32, 2^10);
qcdmodel = eqcd_potential_dataloader(63)[108]
@time sol_3D_SDE_2 = modelA_3d_SDE_Simple_prob(;
    solver = DRI1NM(),
    γ = 1.0f0,
    T = 1.0f0 * qcdmodel.T,
    para = qcdmodel.U,
    u0 = u0_1,
    tspan = (0.0f0, 20.0f0),
    # save_end=true,
    # dt = 0.02f0,
    dx = 0.1f0,
    noise = "sqrt",
    abstol = 5e-2,
    reltol = 5e-2,
    # dt=0.01f0,
)

plot(sol_3D_SDE_2[:, 1], sol_3D_SDE_2[:, 2])


testdata2 = sol_3D_SDE_EM;
mv1 = [mean(vec(testdata2[:, i])) for i = 1:401]

@everywhere qcdmodel = eqcd_potential_dataloader(63)[108]
asyncmap((zip(2:3, [0 1]))) do (p, d)
    remotecall_wait(p) do
        # @info "Worker $p uses $d"
        device!(d)
        # u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^11)
        # v0_1 = CUDA.fill(0.0f0, 32, 32, 32, 2^11)
        task_muB = [0, 1.0f0, -1.0f0][p]
        for muB in task_muB
            for Tem = 0.2f0:0.2f0:0.8f0
                @info "T=$Tem muB=$(muB)"
                GC.gc(true)
                CUDA.reclaim()
                u0_1 = CUDA.fill(task_muB * Tem, 32, 32, 32, 2^13)
                GC.gc(true)
                CUDA.reclaim()
                sol_3D_SDE_EM = modelA_3d_SDE_Simple_prob_EM(;
                    γ = 1.0f0,
                    T = qcdmodel.T,
                    para = qcdmodel.U,
                    u0 = u0_1,
                    tspan = (0.0f0, 30.0f0),
                    # save_end=true,
                    # dt = 0.02f0,
                    dx = 0.1f0,
                    dt = 0.001f0,
                )
                testdata2 = sol_3D_SDE_EM
                mv1 = [mean(vec(testdata2[:, i])) for i = 1:601]
                writedlm(
                    "sims/ini_compare/mub=630_T=108/constini/gamma=1_ini=$(muB*Tem)_time.dat",
                    collect(0.0:0.05:30),
                )
                writedlm(
                    "sims/ini_compare/mub=630_T=108/constini/gamma=1_ini=$(muB*Tem).dat",
                    mv1,
                )
            end
        end
    end
end



u0_1 = CUDA.fill(0.0f0, 32, 32, 32, 2^13);
GC.gc(true)
CUDA.reclaim()
@time sol_3D_SDE_EM = modelA_3d_SDE_Simple_prob_EM(;
    γ = 1.0f0,
    T = qcdmodel.T,
    para = qcdmodel.U,
    u0 = u0_1,
    tspan = (0.0f0, 30.0f0),
    # save_end=true,
    # dt = 0.02f0,
    dx = 0.1f0,
    dt = 0.001f0,
)

testdata2 = sol_3D_SDE_EM;
mv1 = [mean(vec(testdata2[:, i])) for i = 1:601]
writedlm(
    "sims/ini_compare/mub=630_T=108/constini/gamma=1_ini=0_time.dat",
    collect(0.0:0.05:30),
)
writedlm("sims/ini_compare/mub=630_T=108/constini/gamma=1_ini=0.dat", mv1)













u0_1
qcdmodel = eqcd_potential_dataloader(63)[20]
u0_1 = CUDA.randn(32, 32, 32, 2^8) .- 0.2f0;
u0_1 = CUDA.fill(2.0f0, 32, 32, 32, 2^8);

GC.gc(true)
CUDA.reclaim()
u0_1 = 0

@time sol_3D_SDE_EM = modelA_3d_SDE_Simple_prob_EM(;
    γ = 1.0f0,
    T = 3.5f0,
    para = TaylorParameters([0.0f0 0.5f0 0.0f0 0.0f0 0.0f0 1.0f0 0.0f0])[1],
    u0 = u0_1,
    tspan = (0.0f0, 20.0f0),
    # save_end=true,
    # dt = 0.02f0,
    dx = 1.0f0,
    dt = 0.001f0,
)




testdata2 = sol_3D_SDE_EM;
testdata2 = Array(stack(sol_3D_SDE_2[:, 2]))

sum(vec(testdata2[:, end-2]))
mv1 = [mean(vec(testdata2[:, i])) for i = 1:301]
mv2 = [mean(vec(testdata2[:, i]) .^ 2) for i = 1:301]
mv3 = [mean(vec(testdata2[:, i]) .^ 3) for i = 1:301]
mv4 = [mean(vec(testdata2[:, i]) .^ 4) for i = 1:301]
(@.(mv4 - 4 * mv3 * mv1 - 3 * mv2^2 + 12mv2 * mv1^2 - 6 * mv1^4))[2:end]
plot(0.05:0.05:5, (@.(mv4 - 4 * mv3 * mv1 - 3 * mv2^2 + 12mv2 * mv1^2 - 6 * mv1^4))[2:end])
plot!(0.05:0.05:15, (@.(mv2 - mv1^2))[2:end])
plot(0.05:0.05:10, mv1[2:end])




asyncmap((zip(2:3, [0 1]))) do (p, d)
    remotecall_wait(p) do
        # @info "Worker $p uses $d"
        device!(d)
        # u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^11)
        # v0_1 = CUDA.fill(0.0f0, 32, 32, 32, 2^11)
        task_muB = [0, 1:1, 2:2][p]
        for muB in task_muB
            for Tem = 1:16
                @info "T=$Tem muB=$(muB)"
                GC.gc(true)
                CUDA.reclaim()
                u0_1 = CUDA.fill(0.0f0, 32, 32, 32, 2^16)
                GC.gc(true)
                CUDA.reclaim()
                sol_3D_SDE_EM = modelA_3d_SDE_Simple_prob_EM(;
                    γ = 1.0f0,
                    T = qcdmodel.T,
                    para = qcdmodel.U,
                    u0 = u0_1,
                    tspan = (0.0f0, 5.0f0),
                    # save_end=true,
                    # dt = 0.02f0,
                    dx = 0.1f0,
                    dt = 0.001f0,
                )
                writedlm(
                    "sims/ini_compare/mub=630_T=108/gamma=1_ini=0_$(muB)_$Tem.dat",
                    sol_3D_SDE_EM,
                )
            end
        end
    end
end


writedlm("sims/ini_compare/mub=630_T=108/gamma=4.dat", sol_3D_SDE_EM)



testdata = [
    readdlm(
        "/home/tyy/LangevinDynamics/sims/ini_compare/mub=630_T=108/gamma=1_ini=0_$(i)_$j.dat",
    ) for i = 1:2, j = 1:8
];

testdata2 = stack(testdata);
testdata2

mv1 = [mean(vec(testdata2[:, i, :, :])) for i = 1:101]
mv2 = [mean(vec(testdata2[:, i, :, :]) .^ 2) for i = 1:101]
mv3 = [mean(vec(testdata2[:, i, :, :]) .^ 3) for i = 1:101]
mv4 = [mean(vec(testdata2[:, i, :, :]) .^ 4) for i = 1:101]


plot(0.05:0.05:5, (@.(mv4 - 4 * mv3 * mv1 - 3 * mv2^2 + 12mv2 * mv1^2 - 6 * mv1^4))[2:end])





kurtosis(sol_3D_SDE_EM[:, i])
sol_3D_SDE_EM[:, 30]
plot(0.02:0.02:5, [skewness(sol_3D_SDE_EM[:, i]) for i = 2:251])
plot(
    sol_3D_SDE_2[2:end, 1],
    [skewness(Array(stack(sol_3D_SDE_2[:, 2]))[:, i]) for i = 2:101],
)

[skewness(Array(stack(sol_3D_SDE_2[:, 2]))[:, i]) for i = 1:101]
round(Int32, 89.7)





sol_3D_SDE_1[10:end, 1]
plot!(
    sol_3D_SDE_1[100:end, 1],
    log.(abs.(sol_3D_SDE_1[100:end, 2] .- sol_3D_SDE_1[end, 2])),
    ylims = (-8, -1),
)
plot(sol_3D_SDE_1[10:end, 1], stack(sol_3D_SDE_1[10:end, 2])[1, 4, :])

plot(
    sol_3D_SDE_1[:, 1],
    [kurtosis(Array([sol_3D_SDE_1[i, 2]; sol_3D_SDE_2[i, 2]])) for i = 1:51],
)

randn!




plot(Lang)

plot(
    LangevinDynamics.Uσfunout(
        TaylorParameters([0.0f0 20.0f0 0.0f0 0.0f0 0.0f0 0.01f0 0.0f0])[1],
    ),
    -0.6,
    0.6,
)
heatmap(Array(mean(sol_3D_SDE_1[end], dims = [3])[:, :, 1, 1]), aspect_ratio = 1)




Array(sol_3D_SDE_1[end])[:, :, :, 1]

using HDF5
save("sims/ini_compare/T=Tc+5.jld2", "sigma", Array(sol_3D_SDE_1[end])[:, :, :, 1])
h5write("sims/ini_compare/T=Tc.h5", "sigma", Array(sol_3D_SDE_1[end])[:, :, :, 1])

plot(0.0f0:0.02:20.0, sol_3D_SDE_1[1])
v1_his_1 = Array(mean(sol_3D_SDE_1[2], dims = [1, 2, 3])[1, 1, 1, :])
o_his_1 = ash(v1_his_1; rng = 0.04:0.0003:0.08, m = 5)
plot(o_his_1)

1
using AbstractPlotting

volume(rand(32, 32, 32), algorithm = :mip)

1

sol_3D_SDE_ini = langevin_3d_SDE_Simple_prob(;
    γ = 8.0f0,
    T = qcdmodelini.T,
    para = qcdmodelini.U,
    u0 = u0_1,
    v0 = v0_1,
    tspan = (0.0f0, 30.0f0),
    dt = 0.01f0,
    # savefun = meansave
    # u0fun = x -> 0.1f0*CUDA.randn(32,32,32,2^9, 2),
)


device!(1)
u0_2 = CUDA.randn(32, 32, 32, 2^14)
v0_2 = CUDA.randn(32, 32, 32, 2^14);
sol_3D_SDE_2 = langevin_3d_SDE_Simple_prob(;
    γ = 8.0f0,
    T = qcdmodel.T,
    para = qcdmodel.U,
    u0 = u0_2,
    v0 = v0_2,
    tspan = (0.0f0, 20.0f0),
    dt = 0.005f0,
    # savefun = meansave
    # u0fun = x -> 0.1f0*CUDA.randn(32,32,32,2^9, 2),
)

plot(0.0f0:0.005:20.0, sol_3D_SDE_2[1])
v1_his_2 = Array(mean(sol_3D_SDE_2[2], dims = [1, 2, 3])[1, 1, 1, :])
o_his_2 = ash(v1_his_2; rng = 0.04:0.0003:0.08, m = 5)
plot!(o_his_2)

device!(0)
u0_3 = CUDA.fill(0.0f0, 32, 32, 32, 2^14)
v0_3 = CUDA.fill(0.0f0, 32, 32, 32, 2^14);
sol_3D_SDE_3 = langevin_3d_SDE_Simple_prob(;
    γ = 8.0f0,
    T = qcdmodel.T,
    para = qcdmodel.U,
    u0 = u0_3,
    v0 = v0_3,
    tspan = (0.0f0, 20.0f0),
    dt = 0.01f0,
    # savefun = meansave
    # u0fun = x -> 0.1f0*CUDA.randn(32,32,32,2^9, 2),
)


plot(0.0f0:0.01:20.0, sol_3D_SDE_3[1])
v1_his_3 = Array(mean(sol_3D_SDE_3[2], dims = [1, 2, 3])[1, 1, 1, :])
o_his_3 = ash(v1_his_2; rng = 0.04:0.0003:0.08, m = 5)
plot!(o_his_3)




u0_4 = sol_3D_SDE_ini[2];
v0_4 = sol_3D_SDE_ini[3];
sol_3D_SDE_4 = langevin_3d_SDE_Simple_prob(;
    γ = 8.0f0,
    T = qcdmodel.T,
    para = qcdmodel.U,
    u0 = u0_4,
    v0 = v0_4,
    tspan = (0.0f0, 40.0f0),
    dt = 0.005f0,
    # savefun = meansave
    # u0fun = x -> 0.1f0*CUDA.randn(32,32,32,2^9, 2),
)


sol_3D_SDE_2[1]
plot(0.0f0:0.1:20.0, sol_3D_SDE_1[1])
plot!(0.0f0:0.005:40.0, sol_3D_SDE_2[1])
plot!(0.0f0:0.005:40.0, sol_3D_SDE_3[1])
plot!(0.0f0:0.05:40.0, sol_3D_SDE_ini[1])

plot!(0.0f0:0.005:40.0, sol_3D_SDE_4[1])

writedlm("data/const.dat", [0.0f0:0.005:40.0 sol_3D_SDE_1[1]])
writedlm("data/rand.dat", [0.0f0:0.005:40.0 sol_3D_SDE_2[1]])
writedlm("data/zero.dat", [0.0f0:0.005:40.0 sol_3D_SDE_3[1]])
writedlm("data/quench.dat", [0.0f0:0.005:40.0 sol_3D_SDE_4[1]])


sol_3D_SDE_ini[2]
