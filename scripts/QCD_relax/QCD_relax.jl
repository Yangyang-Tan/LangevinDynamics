@everywhere function eqcd_potential_dataloader(i;dim=3)
    lam1data =
        readdlm("data/eQCD_Input/eqcd_potential_data/lamdata/Tem$i/lam1_cut3.dat")[:, 1] .*
        1.6481059699913014^2
    lam2data =
        readdlm("data/eQCD_Input/eqcd_potential_data/lamdata/Tem$i/lam2_cut3.dat")[:, 1] .*
        1.6481059699913014^4
    lam3data =
        readdlm("data/eQCD_Input/eqcd_potential_data/lamdata/Tem$i/lam3_cut3.dat")[:, 1] .*
        1.6481059699913014^6
    lam4data =
        readdlm("data/eQCD_Input/eqcd_potential_data/Tem$i/buffer/lam4_nc.dat")[:, 1] .*
        1.6481059699913014^8
    lam5data =
        readdlm("data/eQCD_Input/eqcd_potential_data/Tem$i/buffer/lam5_nc.dat")[:, 1] .*
        1.6481059699913014^10
    rho0data =
        readdlm("data/eQCD_Input/eqcd_potential_data/Tem$i/buffer/rho0.dat")[:, 1] ./
        1.6481059699913014^2
    cdata =
        readdlm("data/eQCD_Input/eqcd_potential_data/Tem$i/buffer/c.dat")[:, 1] .*
        1.6481059699913014
    Tdata= readdlm("data/eQCD_Input/eqcd_potential_data/Tem$i/buffer/TMeV.dat")[:, 1]./197.33
    taylordata = TaylorParameters(
        Float32,
        [lam1data lam2data lam3data lam4data lam5data rho0data cdata][:,[1:dim...,6,7]],
    )[:,1]
    return QCDModelParameters(taylordata, Tdata)
end

@everywhere function eqcd_Zt_dataloader(muB)
    readdlm("data/Ztdata/mub$muB.dat")[:, 1]
end


@everywhere function eqcd_relaxtime_datasaver(
    muB::Int,#muB
    Tem::Int,#Temperature
    u0::AbstractArray{Float32,4},
)
    qcdmodel=eqcd_potential_dataloader(muB);
    Ztdata=Float32.(eqcd_Zt_dataloader(muB));
    sol_3D_SDE = modelA_3d_SDE_Simple_prob(;
        γ = Ztdata[Tem],
        # γ=0.2f0,
        T = qcdmodel[Tem].T,
        para = qcdmodel[Tem].U,
        u0 = u0,
        # v0 = v0,
        tspan = (0.0f0, 5f0*Ztdata[Tem]),
        dx=0.1f0,
        # dt = 0.0001f0,
        noise="coth",
        
        # savefun = meansave
        # u0fun = x -> 0.1f0*CUDA.randn(32,32,32,2^9, 2),
    )
    writedlm(
        "outputdata/relax_time_O2_noise=coth_dx=0.1/T=$Tem muB=$(muB*10).dat",
        sol_3D_SDE,
    )
end
mkdir("outputdata/relax_time_O2_noise=coth_4Zt_dx=0.1")
eqcd_potential_dataloader(63, dim = 3)[113].U
@time sol_3D_SDE = modelA_3d_SDE_Simple_prob(;
    γ = 1f0,
    T = eqcd_potential_dataloader(63,dim=3)[10].T,
    para = eqcd_potential_dataloader(63, dim = 3)[10].U,
    u0 = u_high,
    tspan = (0.0f0, 2f0),
    noise="sqrt",
    dx=0.1,
    # dt =0.00000000001f0,
    # savefun = meansave
    # u0fun = x -> 0.1f0*CUDA.randn(32,32,32,2^9, 2),
)
u_high


plot(sol_3D_SDE[:,1], sol_3D_SDE[:,2])


u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^6)



du0_1=similar(u0_1)
eqcd_relaxtime_datasaver(63, 100, u0_1)
plot(readdlm("outputdata/relax_time_O2_noise=coth_dx=0.1/T=100 muB=630.dat")[:,1],readdlm("outputdata/relax_time_O2_noise=coth_dx=0.1/T=100 muB=630.dat")[:,2])

plot!(eqcd_Zt_dataloader(50))


for muB in 2:2:10 
    for i = 4:2:250
        @info "T=$i muB=$(muB*10)"
        eqcd_relaxtime_datasaver(muB, i, u0_1)
    end
end
2:2:24|>length
26:2:48|>length
50:2:54|>length
56:2:58|>length

2:4:40|>length
42:2:58|>length
60:2:62|>length
61:2:63|>length

asyncmap((zip(2:5, [0 1 2 3]))) do (p, d)
    remotecall_wait(p) do
        # @info "Worker $p uses $d"
        device!(d)
        # u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^11)
        # v0_1 = CUDA.fill(0.0f0, 32, 32, 32, 2^11)
        u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^8)
        task_muB = [0, 2:2:26, 28:2:50,52:2:56,58:2:62][p]
    for muB in task_muB
        for Tem = 4:2:250
            @info "T=$Tem muB=$(muB*10)"
            eqcd_relaxtime_datasaver(muB, Tem, u0_1)
        end
    end
    end
end

asyncmap((zip(2:3, [0 1]))) do (p, d)
    remotecall_wait(p) do
        # @info "Worker $p uses $d"
        device!(d)
        # u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^11)
        # v0_1 = CUDA.fill(0.0f0, 32, 32, 32, 2^11)
        u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^8)
        task_muB = [0, 61, 63][p]
    for muB in task_muB
        for Tem = 4:2:250
            @info "T=$Tem muB=$(muB*10)"
            eqcd_relaxtime_datasaver(muB, Tem, u0_1)
        end
    end
    end
end




asyncmap((zip(2:3, [0 1]))) do (p, d)
    remotecall_wait(p) do
        # @info "Worker $p uses $d"
        device!(d)
        # u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^11)
        # v0_1 = CUDA.fill(0.0f0, 32, 32, 32, 2^11)
        u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^6)
        task_muB = [0, 2+2:4:20,20+2:4:40][p]
    for muB in task_muB
        for Tem = 50:2:200
            @info "T=$Tem muB=$(muB*10)"
            eqcd_relaxtime_datasaver(muB, Tem, u0_1)
        end
    end
    end
end


asyncmap((zip(2:3, [0 1]))) do (p, d)
    remotecall_wait(p) do
        # @info "Worker $p uses $d"
        device!(d)
        # u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^11)
        # v0_1 = CUDA.fill(0.0f0, 32, 32, 32, 2^11)
        u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^6)
        task_muB = [0, 56,58][p]
    for muB in task_muB
        for Tem = 4:2:250
            @info "T=$Tem muB=$(muB*10)"
            eqcd_relaxtime_datasaver(muB, Tem, u0_1)
        end
    end
    end
end







8:2:20|>length

asyncmap((zip(2:3, [0 1]))) do (p, d)
    remotecall_wait(p) do
        # @info "Worker $p uses $d"
        device!(d)
        # u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^11)
        # v0_1 = CUDA.fill(0.0f0, 32, 32, 32, 2^11)
        u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^8)
        task_muB = [0, 8:2:20, 22:2:34, 36:2:38,40:2:42][p]
    for muB in task_muB
        for Tem = 4:2:250
            @info "T=$Tem muB=$(muB*10)"
            eqcd_relaxtime_datasaver(muB, Tem, u0_1)
        end 
    end
    end
end









asyncmap((zip(2:3, [0 1]))) do (p, d)
    remotecall_wait(p) do
        # @info "Worker $p uses $d"
        device!(d)
        # u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^11)
        # v0_1 = CUDA.fill(0.0f0, 32, 32, 32, 2^11)
        u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^8)
        task_muB = [0, 4:2:4, 6:2:6][p]
    for muB in task_muB
        for Tem = 4:2:250
            @info "T=$Tem muB=$(muB*10)"
            eqcd_relaxtime_datasaver(muB, Tem, u0_1)
        end
    end
    end
end

eqcd_relaxtime_datasaver(58, 120, u0_1)

plot(LangevinDynamics.funout(eqcd_potential_dataloader(10)[120].U), -0.6:0.01:0.6)
plot(LangevinDynamics.Uσfunout(eqcd_potential_dataloader(10)[50].U), -0.6:0.01:.6)

