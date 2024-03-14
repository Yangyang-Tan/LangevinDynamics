using DelimitedFiles
function eqcd_potential_dataloader(i; dim = 3,T=Float32)
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
    Tdata =
        readdlm("data/eQCD_Input/eqcd_potential_data/Tem$i/buffer/TMeV.dat")[:, 1] ./ 197.33
    taylordata = TaylorParameters(
        T,
        [lam1data lam2data lam3data lam4data lam5data rho0data cdata][:, [1:dim..., 6, 7]],
    )[
        :,
        1,
    ]
    return QCDModelParameters(taylordata, Tdata)
end
device!(0)
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
)
writedlm("data/noise1_T=112_mu=630_nocut.dat", [0.0:0.005:100.0 sol_3D_SDE_1[1]])

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
