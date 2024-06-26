@everywhere include("/home/tyy/LangevinDynamics/scripts/QCD_relax/load.jl")
qcdmodel = eqcd_potential_dataloader(63)[1]
u0_1 = CUDA.randn(256, 256, 256, 2^0);
qcdmodel.T
sol_3D_SDE_2=modelA_3d_SDE_Simple_prob2(;
    solver = RK4(),
    γ = 8.0f0,
    T = 0.0001f0,
    para = qcdmodel.U,
    u0 = u0_1,
    tspan = (0.0f0, 80f0),
    save_end=true,
    dt = 0.0001f0,
    dx = 0.1f0,
    noise = "none",
    abstol = 1e-5,
    reltol = 1e-5,
    # dt=0.01f0,
)
[u0_1;u0_1]
sol_3D_SDE_2[end]
using HDF5
h5write("sims/ini_compare/T=0.h5", "sigmat40", Array(sol_3D_SDE_2[end])[:, :, :, 1])

1

heatmap(Array(sol_3D_SDE_2[end][:,:,10,1]),aspect_ratio=1)
heatmap(Array(sol_3D_SDE_2[end][:,:,10,1,1]),aspect_ratio=1)