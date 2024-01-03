plot(x->-x^1/2 + 2*x^3,-1,1)
using AverageShiftedHistograms

u0_1 = fill(0.6f0, 32, 32, 32, 2^1)|>cu
v0_1 = fill(0.0f0, 32, 32, 32, 2^11)
using LinearAlgebra

u0_1
sol_tem=langevin_3d_Ising_Simple_prob(;
    u0 = u0_1,
    v0 = v0_1,
    γ = 0.1f0,
    tspan = (0.0f0, 200.0f0),
    T = 4.f0,
    dt = 0.1f0,
)
sol_tem[2]
plot!(sol_tem[1][3:end])
v_tem=Array(mean(sol_tem[3], dims = [1, 2, 3])[1, 1, 1, :]).|>abs
o_tem = ash(v_tem; rng = -2:0.01:2, m = 5)
plot(o_tem)
cumulant(cu(v_tem),4)


###### modelA


u0_1 = fill(1.0f0, 32, 32, 32, 2^10)
sol_tem = modelA_3d_Ising_Simple_prob(;
    u0 = u0_1,
    γ = 0.1f0,
    tspan = (0.0f0, 10.0f0),
    T = 4.9f0,
    dt = 0.1f0,
)
sol_tem.t
plot(sol_tem.t[1:end], stack(sol_tem.saveval)[1, 1:end])


length((0.0f0, 20.0f0))
sol_tem[2]
plot!(sol_tem[1][3:end])
v_tem = Array(mean(sol_tem[3], dims = [1, 2, 3])[1, 1, 1, :]) .|> abs
o_tem = ash(v_tem; rng = -2:0.01:2, m = 5)
plot(o_tem)
cumulant(cu(v_tem), 4)
