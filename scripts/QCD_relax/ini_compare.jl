using DelimitedFiles
function eqcd_potential_dataloader(i;dim=3)
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

eqcd_potential_dataloader(1)[150]
qcdmodelini = eqcd_potential_dataloader(1)[250];

u0_1 = CUDA.fill(0.6f0, 32, 32, 32, 2^14);
v0_1 = CUDA.fill(0.0f0, 32, 32, 32, 2^14);
qcdmodel = eqcd_potential_dataloader(1)[150]
sol_3D_SDE_1 = langevin_3d_SDE_Simple_prob(;
        γ = 8.0f0,
        T = qcdmodel.T,
        para = qcdmodel.U,
        u0 = u0_1,
        v0 = v0_1,
        tspan = (0.0f0, 20.0f0),
        dt = 0.02f0,
        # savefun = meansave
        # u0fun = x -> 0.1f0*CUDA.randn(32,32,32,2^9, 2),
    )

plot(0.0f0:0.02:20.0, sol_3D_SDE_1[1])
v1_his_1=Array(mean(sol_3D_SDE_1[2],dims=[1,2,3])[1,1,1,:])
o_his_1=ash(v1_his_1; rng = 0.04:0.0003:0.08, m = 5)
plot(o_his_1)




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
u0_3 = CUDA.fill(0f0,32, 32, 32, 2^14)
v0_3 = CUDA.fill(0f0,32, 32, 32, 2^14);
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
plot(0.0f0:0.1:20.0,sol_3D_SDE_1[1])
plot!(0.0f0:0.005:40.0,sol_3D_SDE_2[1])
plot!(0.0f0:0.005:40.0, sol_3D_SDE_3[1])
plot!(0.0f0:0.05:40.0, sol_3D_SDE_ini[1])

plot!(0.0f0:0.005:40.0, sol_3D_SDE_4[1])

writedlm("data/const.dat", [0.0f0:0.005:40.0 sol_3D_SDE_1[1]])
writedlm("data/rand.dat", [0.0f0:0.005:40.0 sol_3D_SDE_2[1]])
writedlm("data/zero.dat", [0.0f0:0.005:40.0 sol_3D_SDE_3[1]])
writedlm("data/quench.dat", [0.0f0:0.005:40.0 sol_3D_SDE_4[1]])


sol_3D_SDE_ini[2]
