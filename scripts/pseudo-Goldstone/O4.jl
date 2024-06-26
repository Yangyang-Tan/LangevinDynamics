u0_1 = CUDA.fill(3.0f0, 32, 32, 32, 2^6, 4);
using JLD2


solO4 = modelA_3d_O4_SDE_Simple_prob(;
    u0 = u0_1,
    γ = 1.0f0,
    tspan = (0.0f0, 240.0f0),
    T = 5.7f0,
    c = 0.0f0,
    dt = 0.1f0,
    dx = 1.0f0,
    solver = DRI1NM(),
    abstol = 5e-2,
    reltol = 5e-2,
)
240 * 0.05


solO42 = modelA_3d_O4_SDE_Simple_prob2(;
    u0 = u0_1,
    γ = 1.0f0,
    tspan = (0.0f0, 12.0f0 / 0.05f0),
    T = 2.7f0,
    c = 0.05f0,
    dt = 0.1f0,
    dx = 1.0f0,
    solver = DRI1NM(),
    abstol = 5e-2,
    reltol = 5e-2,
)

stack(solO42[:, 2])


plot(solO42[:, 1], stack(solO42[:, 2])[1, :])

plot(solO42[:, 1], stack(solO42[:, 2])[2, :])


vardata2 = Dict()
meandatac = Dict()
meandatac2 = Dict()

cdata = exp.(log(0.05):0.5:log(5))


for Tem = 5.5f0:0.1f0:6.5f0
    vardata[rationalize(Tem)] = modelA_3d_O4_SDE_Simple_prob(;
        u0 = u0_1,
        γ = 1.0f0,
        tspan = (0.0f0, 240.0f0),
        T = Tem,
        c = 0.0f0,
        dt = 0.1f0,
        dx = 1.0f0,
        solver = DRI1NM(),
        abstol = 5e-2,
        reltol = 5e-2,
    )
end



for Tem in cdata
    meandatac2[rationalize(Tem)] = modelA_3d_O4_SDE_Simple_prob2(;
        u0 = u0_1,
        γ = 1.0f0,
        tspan = (0.0f0, 20.0f0 / Tem),
        T = 0.4f0,
        c = Tem,
        dt = 0.1f0,
        dx = 1.0f0,
        solver = DRI1NM(),
        abstol = 5e-2,
        reltol = 5e-2,
    )
end


1
for Tem = 5.5f0:0.1f0:6.0f0
    display(
        plot!(
            vardata[rationalize(Tem)][:, 1],
            stack(vardata[rationalize(Tem)][:, 2])[1, 2, :],
        ),
    )
end
plot()

save("sims/pseudo-Goldstone/O4_32.jld2", "vardata", vardata)

save("sims/pseudo-Goldstone/O4_32_mean.jld2", "meandatac", meandatac)

save("sims/pseudo-Goldstone/O4_32_mean.jld2", "T=1.0", meandatac2)



plot(vardata[rationalize(Tem)][:, 1], stack(vardata[rationalize(Tem)][:, 2])[1, 2, :])
meandatac
stack(meandatac2[rationalize(cdata[1])][:, 2])[1, :]
plot(
    meandatac2[rationalize(cdata[1])][:, 1],
    stack(meandatac2[rationalize(cdata[1])][:, 2])[1, :],
)