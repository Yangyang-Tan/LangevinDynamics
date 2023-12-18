taylordata = TaylorParameters(Float32, [-1.0 1 0])
Tdata = Float32.(readdlm("data/eQCD_Input/chiral/T.dat")[:, 1]);

function meansave(u,mvec)
    dvu = abs.(dropdims(mean(u, dims = [1, 2, 3]), dims = (1, 2, 3)))
    push!(mvec, mean(dvu.^2)-mean(dvu)^2)
end

function meansave(u)
    dvu = mean(u, dims = [1, 2, 3])
    return mean(abs.(dvu))
end



CUDA.device!(0)
u0_1 = CUDA.fill(1f0, 64, 64, 64, 2^7)
u0_1 = CUDA.randn(64, 64, 64, 2^7).+1.0f0
v0_1 = CUDA.fill(0.0f0, 64, 64, 64, 2^7)
u0_1_g=0
@time for i in 1:2000
    CUDA.@sync randn!(u0_1_g)
end

c1_chiral = Float32[]
CUDA.seed!(1)
@time sol_3D_SDE = langevin_3d_Ising_Simple_prob(;
    γ = 0.3f0,
    # T = 9.37074f0-0.05f0,
    T=30.18f0,
    para = taylordata[1],
    u0 = u0_1,
    v0 = v0_1,
    tspan = (0.0f0, 100f0),
    dt = 0.05f0,
    savefun = meansave
    # u0fun = x -> 0.1f0*CUDA.randn(32,32,32,2^9, 2),
)
sol_3D_SDE[1]
sol_3D_SDE[1]
plot!(0.0f0:0.05f0:100.0f0, sol_3D_SDE[1])
plot!(0f0:0.01f0:500.0f0, sol_3D_SDE[1])

sol_3D_SDE[1]

c1_chiral
plot!(0:0.1:200, c1_chiral)
# writedlm(
#     "sims/mub=0_chiral/FiniteSize_Ising/T=Tc ini=0.6_uniform_gam=0.3_L=40.dat",
#     [0:0.1:1200 c1_chiral],
# )
