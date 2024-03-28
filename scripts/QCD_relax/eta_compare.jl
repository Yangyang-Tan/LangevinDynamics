using DelimitedFiles

@everywhere include("/home/tyy/LangevinDynamics/scripts/QCD_relax/load.jl")



@everywhere qcdmodel = eqcd_potential_dataloader(63)[108]
asyncmap((zip(2:3, [0 1]))) do (p, d)
    remotecall_wait(p) do
        # @info "Worker $p uses $d"
        device!(d)
        # u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^11)
        # v0_1 = CUDA.fill(0.0f0, 32, 32, 32, 2^11)
        task_muB = [0,[0.05f0,0.1f0,0.5f0,1.0f0], [2f0,5f0,10f0,20f0]][p]
        for muB in task_muB
            for Tem = 4:6
                @info "T=$Tem muB=$(muB)"
                GC.gc(true)
                CUDA.reclaim()
                u0_1 = CUDA.randn(Float32,32, 32, 32, 2^12)
                GC.gc(true)
                CUDA.reclaim()
                sol_3D_SDE_EM = modelA_3d_SDE_Simple_prob(;
                    solver = DRI1NM(),
                    γ = muB,
                    T = qcdmodel.T,
                    para = qcdmodel.U,
                    u0 = u0_1,
                    tspan = (0.0f0, muB*10),
                    # save_end=true,
                    # dt = 0.02f0,
                    dx = 0.1f0,
                    noise = "sqrt",
                    abstol = 6e-2,
                    reltol = 6e-2,
                    # dt=0.01f0,
                )
                testdata2 = sol_3D_SDE_EM
                # mv1 = [mean(vec(testdata2[:, i])) for i = 1:601]
                # writedlm(
                #     "sims/eta_compare/mub=630_T=108/gamma=$(muB)_time.dat",
                #     testdata2[:,1],
                # )
                writedlm(
                    "sims/eta_compare/mub=630_T=108_ini=randn/gamma=$(muB)_Batch=$Tem.dat",
                    testdata2,
                )
            end
        end
    end
end


@everywhere qcdmodel = eqcd_potential_dataloader(63)[200]
asyncmap((zip(2:3, [0 1]))) do (p, d)
    remotecall_wait(p) do
        # @info "Worker $p uses $d"
        device!(d)
        # u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^11)
        # v0_1 = CUDA.fill(0.0f0, 32, 32, 32, 2^11)
        task_muB = [0,[0.05f0,0.1f0,0.5f0,1.0f0], [2f0,5f0,10f0,20f0]][p]
        for muB in task_muB
            for Tem = 1:1
                @info "T=$Tem muB=$(muB)"
                GC.gc(true)
                CUDA.reclaim()
                u0_1 = CUDA.randn(Float32,32, 32, 32, 2^12)
                GC.gc(true)
                CUDA.reclaim()
                sol_3D_SDE_EM = modelA_3d_SDE_Simple_prob(;
                    solver = DRI1NM(),
                    γ = muB,
                    T = qcdmodel.T,
                    para = qcdmodel.U,
                    u0 = u0_1,
                    tspan = (0.0f0, muB*3),
                    # save_end=true,
                    # dt = 0.02f0,
                    dx = 0.1f0,
                    noise = "sqrt",
                    abstol = 5e-2,
                    reltol = 5e-2,
                    # dt=0.01f0,
                )
                testdata2 = sol_3D_SDE_EM
                # mv1 = [mean(vec(testdata2[:, i])) for i = 1:601]
                # writedlm(
                #     "sims/eta_compare/mub=630_T=108/gamma=$(muB)_time.dat",
                #     testdata2[:,1],
                # )
                writedlm(
                    "sims/eta_compare/mub=630_T=200/gamma=$muB.dat",
                    testdata2,
                )
            end
        end
    end
end









@everywhere qcdmodel = eqcd_potential_dataloader(63)[108]
asyncmap((zip(2:3, [0 1]))) do (p, d)
    remotecall_wait(p) do
        # @info "Worker $p uses $d"
        device!(d)
        # u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^11)
        # v0_1 = CUDA.fill(0.0f0, 32, 32, 32, 2^11)
        task_muB = [0,[0.05f0,0.1f0,0.5f0,1.0f0], [2f0,5f0,10f0,20f0]][p]
        for muB in task_muB
            for Tem = 4:6
                @info "T=$Tem muB=$(muB)"
                GC.gc(true)
                CUDA.reclaim()
                u0_1 = CUDA.fill(0.6f0,32, 32, 32, 2^12)
                GC.gc(true)
                CUDA.reclaim()
                sol_3D_SDE_EM = modelA_3d_SDE_Simple_prob(;
                    solver = DRI1NM(),
                    γ = muB,
                    T = qcdmodel.T,
                    para = qcdmodel.U,
                    u0 = u0_1,
                    tspan = (0.0f0, muB*10),
                    # save_end=true,
                    # dt = 0.02f0,
                    dx = 0.1f0,
                    noise = "sqrt",
                    abstol = 5e-2,
                    reltol = 5e-2,
                    # dt=0.01f0,
                )
                testdata2 = sol_3D_SDE_EM
                # mv1 = [mean(vec(testdata2[:, i])) for i = 1:601]
                # writedlm(
                #     "sims/eta_compare/mub=630_T=108/gamma=$(muB)_time.dat",
                #     testdata2[:,1],
                # )
                writedlm(
                    "sims/eta_compare/mub=630_T=108_constini/gamma=$(muB)_Batch=$Tem.dat",
                    testdata2,
                )
            end
        end
    end
end





@everywhere qcdmodel = eqcd_potential_dataloader(63)[200]
asyncmap((zip(2:3, [0 1]))) do (p, d)
    remotecall_wait(p) do
        # @info "Worker $p uses $d"
        device!(d)
        # u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^11)
        # v0_1 = CUDA.fill(0.0f0, 32, 32, 32, 2^11)
        task_muB = [0,[0.05f0,0.1f0,0.5f0,1.0f0], [2f0,5f0,10f0,20f0]][p]
        for muB in task_muB
            for Tem = 1:1
                @info "T=$Tem muB=$(muB)"
                GC.gc(true)
                CUDA.reclaim()
                u0_1 = CUDA.fill(0.6f0,32, 32, 32, 2^12)
                GC.gc(true)
                CUDA.reclaim()
                sol_3D_SDE_EM = modelA_3d_SDE_Simple_prob(;
                    solver = DRI1NM(),
                    γ = muB,
                    T = qcdmodel.T,
                    para = qcdmodel.U,
                    u0 = u0_1,
                    tspan = (0.0f0, muB*3),
                    # save_end=true,
                    # dt = 0.02f0,
                    dx = 0.1f0,
                    noise = "sqrt",
                    abstol = 5e-2,
                    reltol = 5e-2,
                    # dt=0.01f0,
                )
                testdata2 = sol_3D_SDE_EM
                # mv1 = [mean(vec(testdata2[:, i])) for i = 1:601]
                # writedlm(
                #     "sims/eta_compare/mub=630_T=108/gamma=$(muB)_time.dat",
                #     testdata2[:,1],
                # )
                writedlm(
                    "sims/eta_compare/mub=630_T=200_constini/gamma=$muB.dat",
                    testdata2,
                )
            end
        end
    end
end





@everywhere qcdmodel = eqcd_potential_dataloader(63)[50]
asyncmap((zip(2:3, [0 1]))) do (p, d)
    remotecall_wait(p) do
        # @info "Worker $p uses $d"
        device!(d)
        # u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^11)
        # v0_1 = CUDA.fill(0.0f0, 32, 32, 32, 2^11)
        task_muB = [0,[0.05f0,0.1f0,0.5f0,1.0f0], [2f0,5f0,10f0,20f0]][p]
        for muB in task_muB
            for Tem = 1:1
                @info "T=$Tem muB=$(muB)"
                GC.gc(true)
                CUDA.reclaim()
                u0_1 = CUDA.fill(0.6f0,32, 32, 32, 2^12)
                GC.gc(true)
                CUDA.reclaim()
                sol_3D_SDE_EM = modelA_3d_SDE_Simple_prob(;
                    solver = DRI1NM(),
                    γ = muB,
                    T = qcdmodel.T,
                    para = qcdmodel.U,
                    u0 = u0_1,
                    tspan = (0.0f0, muB*3),
                    # save_end=true,
                    # dt = 0.02f0,
                    dx = 0.1f0,
                    noise = "sqrt",
                    abstol = 5e-2,
                    reltol = 5e-2,
                    # dt=0.01f0,
                )
                testdata2 = sol_3D_SDE_EM
                # mv1 = [mean(vec(testdata2[:, i])) for i = 1:601]
                # writedlm(
                #     "sims/eta_compare/mub=630_T=108/gamma=$(muB)_time.dat",
                #     testdata2[:,1],
                # )
                writedlm(
                    "sims/eta_compare/mub=630_T=50_constini/gamma=$muB.dat",
                    testdata2,
                )
            end
        end
    end
end




@everywhere qcdmodel = eqcd_potential_dataloader(63)[50]
asyncmap((zip(2:3, [0 1]))) do (p, d)
    remotecall_wait(p) do
        # @info "Worker $p uses $d"
        device!(d)
        # u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^11)
        # v0_1 = CUDA.fill(0.0f0, 32, 32, 32, 2^11)
        task_muB = [0,[0.05f0,0.1f0,0.5f0,1.0f0], [2f0,5f0,10f0,20f0]][p]
        for muB in task_muB
            for Tem = 1:1
                @info "T=$Tem muB=$(muB)"
                GC.gc(true)
                CUDA.reclaim()
                u0_1 = CUDA.randn(Float32,32, 32, 32, 2^12)
                GC.gc(true)
                CUDA.reclaim()
                sol_3D_SDE_EM = modelA_3d_SDE_Simple_prob(;
                    solver = DRI1NM(),
                    γ = muB,
                    T = qcdmodel.T,
                    para = qcdmodel.U,
                    u0 = u0_1,
                    tspan = (0.0f0, muB*3),
                    # save_end=true,
                    # dt = 0.02f0,
                    dx = 0.1f0,
                    noise = "sqrt",
                    abstol = 6e-2,
                    reltol = 6e-2,
                    # dt=0.01f0,
                )
                testdata2 = sol_3D_SDE_EM
                # mv1 = [mean(vec(testdata2[:, i])) for i = 1:601]
                # writedlm(
                #     "sims/eta_compare/mub=630_T=108/gamma=$(muB)_time.dat",
                #     testdata2[:,1],
                # )
                writedlm(
                    "sims/eta_compare/mub=630_T=50_ini=randn/gamma=$muB.dat",
                    testdata2,
                )
            end
        end
    end
end






qcdmodelini = eqcd_potential_dataloader(63, T = Float64)[112];
u0_1 = CUDA.fill(0.6, 64, 64, 64, 2^10);
v0_1 = CUDA.fill(0.0, 64, 64, 64, 2^10);
qcdmodel = eqcd_potential_dataloader(63, T = Float64)[112]
@time sol_3D_SDE_1 = langevin_3d_SDE_Simple_prob3(;
    γ = 8.0,
    T = qcdmodel.T,
    para = qcdmodel.U,
    u0 = u0_1,
    v0 = v0_1,
    tspan = (0.0, 100.0),
    dt = 0.005,
    # savefun = meansave
    # u0fun = x -> 0.1f0*CUDA.randn(32,32,32,2^9, 2),
)noise1_T
writedlm("data/=112_mu=630_nocut.dat", [0.0:0.005:100.0 sol_3D_SDE_1[1]])

u0_1 = CUDA.fill(0.6, 64, 64, 64, 2^1);
v0_1 = CUDA.fill(0.0, 64, 64, 64, 2^1);
qcdmodel = eqcd_potential_dataloader(63, T = Float64)[112]
@time sol_3D_SDE_1 = langevin_3d_SDE_Simple_prob2(;
    γ = 8.0,
    T = qcdmodel.T,
    para = qcdmodel.U,
    u0 = u0_1,
    v0 = v0_1,
    tspan = (0.0, 100.0),
    dt = 0.005,
    # savefun = meansave
    # u0fun = x -> 0.1f0*CUDA.randn(32,32,32,2^9, 2),
)
writedlm("data/noise0_T=112_mu=630_nocut.dat", [0.0:0.005:100.0 sol_3D_SDE_1[1]])



##T=50
u0_3 = CUDA.fill(0.6, 32, 32, 32, 2^12);
v0_3 = CUDA.fill(0.0, 32, 32, 32, 2^12);
qcdmodel = eqcd_potential_dataloader(1, T = Float64)[50]
@time sol_3D_SDE_3 = langevin_3d_SDE_Simple_prob3(;
    γ = 8.0,
    T = qcdmodel.T,
    para = qcdmodel.U,
    u0 = u0_3,
    v0 = v0_3,
    tspan = (0.0, 100.0),
    dt = 0.005,
    # savefun = meansave
    # u0fun = x -> 0.1f0*CUDA.randn(32,32,32,2^9, 2),
)
writedlm("data/noise1_T=50_mu=10.dat", [0.0:0.005:100.0 sol_3D_SDE_3[1]])

u0_3 = CUDA.fill(0.6, 64, 64, 64, 2^1);
v0_3 = CUDA.fill(0.0, 64, 64, 64, 2^1);
@time sol_3D_SDE_3 = langevin_3d_SDE_Simple_prob2(;
    γ = 8.0,
    T = qcdmodel.T,
    para = qcdmodel.U,
    u0 = u0_3,
    v0 = v0_3,
    tspan = (0.0, 100.0),
    dt = 0.005,
    # savefun = meansave
    # u0fun = x -> 0.1f0*CUDA.randn(32,32,32,2^9, 2),
)
writedlm("data/noise0_T=50_mu=10.dat", [0.0:0.005:100.0 sol_3D_SDE_3[1]])
sol_3D_SDE_3=0
u0_3=0
v0_3=0
##T=250
u0_4 = CUDA.fill(0.6, 32, 32, 32, 2^12)
v0_4 = CUDA.fill(0.0, 32, 32, 32, 2^12);
qcdmodel = eqcd_potential_dataloader(1, T = Float64)[250]
@time sol_3D_SDE_4 = langevin_3d_SDE_Simple_prob3(;
    γ = 8.0,
    T = qcdmodel.T,
    para = qcdmodel.U,
    u0 = u0_4,
    v0 = v0_4,
    tspan = (0.0, 100.0),
    dt = 0.005,
    # savefun = meansave
    # u0fun = x -> 0.1f0*CUDA.randn(32,32,32,2^9, 2),
)
writedlm("data/noise1_T=250_mu=10.dat", [0.0:0.005:100.0 sol_3D_SDE_4[1]])


u0_4 = CUDA.fill(0.6, 64, 64, 64, 2^1)
v0_4 = CUDA.fill(0.0, 64, 64, 64, 2^1);
qcdmodel = eqcd_potential_dataloader(1, T = Float64)[250]
@time sol_3D_SDE_4 = langevin_3d_SDE_Simple_prob2(;
    γ = 8.0,
    T = qcdmodel.T,
    para = qcdmodel.U,
    u0 = u0_4,
    v0 = v0_4,
    tspan = (0.0, 100.0),
    dt = 0.005,
    # savefun = meansave
    # u0fun = x -> 0.1f0*CUDA.randn(32,32,32,2^9, 2),
)
writedlm("data/noise0_T=250_mu=10.dat", [0.0:0.005:100.0 sol_3D_SDE_4[1]])



plot(0.0f0:0.005f0:80.0, sol_3D_SDE_1[1])
plot(0.0f0:0.01f0:40.0, sol_3D_SDE_2[1])
sol_3D_SDE_2[1][1:250]

writedlm("data/noise1.dat", [0.0f0:0.005:40.0 sol_3D_SDE_1[1]])
writedlm("data/noise0.dat", [0.0f0:0.005:40.0 sol_3D_SDE_2[1]])

funout_cut2()
