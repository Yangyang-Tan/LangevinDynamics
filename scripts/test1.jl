plot(x->-x^2/2 + x^4 / 24,-4,4)
using AverageShiftedHistograms


u0_1 = fill(4f0, 32, 32, 32, 2^11)
v0_1 = fill(0.0f0, 32, 32, 32, 2^11)
sol_tem=langevin_3d_Ising_Simple_prob(;
    u0 = u0_1,
    v0 = v0_1,
    γ = 1.0f0,
    tspan = (0.0f0, 300.0f0),
    T = 9.4f0,
    dt = 0.1f0,
)
sol_tem[2]
plot!(sol_tem[2]/9.5)
v_tem=Array(sol_tem[3])
o_tem=ash(sol_tem[3]; rng = -1:0.001:1, m = 5)
