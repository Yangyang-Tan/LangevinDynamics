using Distributed
addprocs(2)
@everywhere using CUDA




lam1data=readdlm("data/eQCD_Input/chiral/lam1.dat")[:,1];
lam2data=readdlm("data/eQCD_Input/chiral/lam2.dat")[:,1];
taylordata = TaylorParameters(Float32, [lam1data lam2data])
Tdata = Float32.(readdlm("data/eQCD_Input/chiral/T.dat")[:, 1]);

function meansave(u,mvec)
    dvu = dropdims(mean(u, dims = [1, 2, 3]), dims = (1, 2, 3))
    push!(mvec, cumulant(dvu, 1))
end

u0_1 = fill(0.6f0, 32, 32, 32, 2^12)
v0_1 = fill(0.0f0, 32, 32, 32, 2^12)
c1_chiral = Float32[]

@time sol_3D_SDE = langevin_3d_SDE_Simple_prob(;
    γ = 0.1f0,
    T = Tdata[1],
    para = taylordata[1],
    u0 = u0_1,
    v0 = v0_1,
    tspan = (0.0f0, 2400.0f0),
    dt = 0.1f0,
    savefun = x -> meansave(x, c1_chiral)
    # u0fun = x -> 0.1f0*CUDA.randn(32,32,32,2^9, 2),
)

plot!(0:0.1:2400, c1_chiral)



writedlm(
    "sims/mub=0_chiral/FiniteSize/Tn=1 ini=0.6_uniform_gam=0_L=32.dat",
    [0:0.1:2400 c1_chiral],
)



for i in 20:1:30
    c1_chiral = Float32[]
    langevin_3d_SDE_Simple_prob(;
        γ = 0.1f0,
        T = Tdata[i],
        para = taylordata[i],
        u0 = u0_1,
        v0 = v0_1,
        tspan = (0.0f0, 600.0f0),
        dt = 0.1f0,
        savefun = x -> meansave(x, c1_chiral),
        # u0fun = x -> 0.1f0*CUDA.randn(32,32,32,2^9, 2),
    )
    writedlm(
        "sims/mub=0_chiral/mean/Tn=$i ini=0.6_uniform_gam=0.1_mean2.dat",
        [0:0.1f0:600 c1_chiral],
    )
end
