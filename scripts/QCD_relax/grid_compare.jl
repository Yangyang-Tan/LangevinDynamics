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
        tspan = (0.0f0, 20f0*Ztdata[Tem]),
        dt = 0.0001f0,
        # savefun = meansave
        # u0fun = x -> 0.1f0*CUDA.randn(32,32,32,2^9, 2),
    )
    writedlm(
        "outputdata/relax_time_O2_noise=coth_grid/T=$Tem muB=$(muB*10).dat",
        sol_3D_SDE,
    )
end
u0_1 = CUDA.fill(0.6f0, 4, 4, 4, 2^14)
u0_1 = CUDA.fill(0.6f0, 64, 64, 64, 2^8)
u0_1 = CUDA.fill(0.6f0, 64, 64, 64, 2^8)
u0_1 = CUDA.fill(0.6f0, 64, 64, 64, 2^2)
u0_1 = CUDA.randn(4, 4, 4, 2^14)

u0_1 = 0.2f0*CUDA.randn(64, 64, 64, 2^2).+0.6f0
u0_1 = 0.2f0*CUDA.randn(32, 32, 32, 2^7).+0.6f0
u0_1 = 0.2f0*CUDA.randn(16, 16, 16, 2^7).+0.6f0
u0_1 = 0.2f0*CUDA.randn(8, 8, 8, 2^9).+0.6f0

eqcd_relaxtime_datasaver(10, 40, u0_1)
@time sol_3D_SDE = modelA_3d_SDE_Simple_prob(;
    γ = 1.f0,
    T = eqcd_potential_dataloader(63,dim=3)[110].T,
    para = eqcd_potential_dataloader(63, dim = 3)[110].U,
    u0 = u0_1,
    tspan = (0.0f0, 20f0),
    dx=1f0,
    # noise = 0.01,
    # dt =0.00000000001f0,
    # savefun = meansave
    # u0fun = x -> 0.1f0*CUDA.randn(32,32,32,2^9, 2),
)


plot!(sol_3D_SDE[:,1],sol_3D_SDE[:,2])