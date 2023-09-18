@everywhere function eqcd_potential_dataloader(i;dim=3)
    lam1data =
        readdlm("data/eQCD_Input/eqcd_potential_data/Tem$i/buffer/lam1.dat")[:, 1] .*
        1.6481059699913014^2
    lam2data =
        readdlm("data/eQCD_Input/eqcd_potential_data/Tem$i/buffer/lam2.dat")[:, 1] .*
        1.6481059699913014^4
    lam3data =
        readdlm("data/eQCD_Input/eqcd_potential_data/Tem$i/buffer/lam3.dat")[:, 1] .*
        1.6481059699913014^6
    lam4data =
        readdlm("data/eQCD_Input/eqcd_potential_data/Tem$i/buffer/lam4.dat")[:, 1] .*
        1.6481059699913014^8
    lam5data =
        readdlm("data/eQCD_Input/eqcd_potential_data/Tem$i/buffer/lam5.dat")[:, 1] .*
        1.6481059699913014^10
    rho0data =
        readdlm("data/eQCD_Input/eqcd_potential_data/Tem$i/buffer/rho0.dat")[:, 1] ./
        1.6481059699913014^2
    cdata =
        readdlm("data/eQCD_Input/eqcd_potential_data/Tem$i/buffer/c.dat")[:, 1] .*
        1.6481059699913014^2
    Tdata= readdlm("data/eQCD_Input/eqcd_potential_data/Tem$i/buffer/TMeV.dat")[:, 1]./197.33
    taylordata = TaylorParameters(
        Float32,
        [lam1data lam2data lam3data lam4data lam5data rho0data cdata][:,[1:dim...,6,7]],
    )[:,1]
    return QCDModelParameters(taylordata, Tdata)
end
@everywhere function eqcd_relaxtime_datasaver(
    i::Int,
    j::Int,
    u0::AbstractArray{Float32,4},
    v0::AbstractArray{Float32,4},
)
    qcdmodel=eqcd_potential_dataloader(i);
    sol_3D_SDE = langevin_3d_SDE_Simple_prob(;
        γ = 8.0f0,
        T = qcdmodel[j].T,
        para = qcdmodel[j].U,
        u0 = u0,
        v0 = v0,
        tspan = (0.0f0, 20.0f0),
        dt = 0.1f0,
        # savefun = meansave
        # u0fun = x -> 0.1f0*CUDA.randn(32,32,32,2^9, 2),
    )
    writedlm(
        "sims/eqcd_relax_phase/relax_time_O5/T=$j muB=$(i*10).dat",
        [0.0f0:0.1f0:20.0f0 sol_3D_SDE[1]],
    )
end

v0_1
writedlm("data/eQCD_Input/eqcd_potential_data/Tem40/buffer/lam1.dat", 1 ./readdlm("data/eQCD_Input/eqcd_potential_data/Tem40/buffer/mSigma_phy.dat")[:, 1])
eqcd_potential_dataloader(2)[1].U
u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^8)
v0_1 = CUDA.fill(0.0f0, 32, 32, 32, 2^8)



eqcd_relaxtime_datasaver(5, 1, u0_1, v0_1)
CUDA.device!(0)
readdlm("data/eQCD_Input/eqcd_potential_data/Tem63/buffer/rho0.dat")[:, 1]
plot(1 ./readdlm("data/eQCD_Input/eqcd_potential_data/Tem40/buffer/mSigma_phy.dat")[:, 1])
@time sol_3D_SDE = langevin_3d_SDE_Simple_prob(;
    γ = 0.001f0,
    T = eqcd_potential_dataloader(63)[250].T,
    para = eqcd_potential_dataloader(63)[250].U,
    u0 = u0_1,
    v0 = v0_1,
    tspan = (0.0f0, 50f0),
    dt = 0.02f0,
    # savefun = meansave
    # u0fun = x -> 0.1f0*CUDA.randn(32,32,32,2^9, 2),
)
sol_3D_SDE[1]
plot!(0f0:0.02:50,sol_3D_SDE[1])





plot(LangevinDynamics.Uσfunout(eqcd_potential_dataloader(59,dim=2)[20].U), -0.58:0.01:0.58)

plot!(LangevinDynamics.Uσfunout(eqcd_potential_dataloader(59)[20].U), -0.58:0.01:0.58)


plot!(LangevinDynamics.Uσfunout(eqcd_potential_dataloader(59, dim = 4)[20].U), -0.58:0.01:0.58)
plot!(LangevinDynamics.Uσfunout(eqcd_potential_dataloader(59, dim = 5)[20].U), -0.58:0.01:0.58)






eqcd_potential_dataloader(52)[134].U
plot(1 ./readdlm("data/eQCD_Input/eqcd_potential_data/Tem40/buffer/lam1.dat")[1:250, 1])
eqcd_potential_dataloader(1)[1].T
readdlm("data/eQCD_Input/eqcd_potential_data/Tem10/buffer/lam5.dat")[:,1]


plot(LangevinDynamics.funout(eqcd_potential_dataloader(63)[150].U), -1:0.01:1)
plot!(LangevinDynamics.funout_cut(eqcd_potential_dataloader(63)[150].U), -1:0.01:1)




plot(LangevinDynamics.Uσfunout(eqcd_potential_dataloader(1)[250].U), -1:0.01:1)


@time @sync begin
    @async begin
        device!(0)
        u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^11)
        v0_1 = CUDA.fill(0.0f0, 32, 32, 32, 2^11)
        eqcd_relaxtime_datasaver(1, 3)

        # do work on GPU 0 here
    end
    @async begin
        device!(1)
        u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^11)
        v0_1 = CUDA.fill(0.0f0, 32, 32, 32, 2^11)
        eqcd_relaxtime_datasaver(1, 2)
        # do work on GPU 1 here
    end
end
zip(workers(1:2), devices(0:1))
(zip(workers(), devices()))
workers()

for j in 1:2:62
    asyncmap((zip(2:3, 0:1))) do (p, d)
    remotecall_wait(p) do
        # @info "Worker $p uses $d"
        device!(d)
            u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^11)
            v0_1 = CUDA.fill(0.0f0, 32, 32, 32, 2^11)
            for i = 1:2:250
            @info "T=$i muB=$(j*10)"
            eqcd_relaxtime_datasaver(j+d, i,u0_1, v0_1)
        end
    end
    end
end


asyncmap((zip(2:4, [0 1 3]))) do (p, d)
    remotecall_wait(p) do
        # @info "Worker $p uses $d"
        device!(d)
        u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^11)
        v0_1 = CUDA.fill(0.0f0, 32, 32, 32, 2^11)
        task_muB = [0, 1:1:31, 32:1:61, 62:1:63][p]
    for j in task_muB
        for i = 2:2:250
            @info "T=$i muB=$(j*10)"
            eqcd_relaxtime_datasaver(j, i, u0_1, v0_1)
        end
    end
    end
end


eqcd_relaxtime_datasaver(1, 1, u0_1, v0_1)


readdlm("data/eQCD_Input/eqcd_potential_data/Tem1/buffer/mSigma_phy.dat")[1, 1] /
readdlm("data/eQCD_Input/eqcd_potential_data/Tem1/buffer/mSigma.dat")[1, 1]
