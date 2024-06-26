u0 = CUDA.randn(Float32, 512, 512)
u0 = CUDA.fill(1.0, 256, 256)
u0=Float32.(readdlm("sims/ModelB/t=0.dat"))
@time solB3=modelB_2d_ODE_prob(
    u0 = u0,
    solver =Tsit5(),
    tspan = (0.0f0, 10000f0),
    abstol = 1e-4,
    reltol = 1e-4,
    save_start=true,
    save_end=true,
    γ=1.0f0,
    dx=1f0,
)

solB3=0
solA3=0

@time solA3=modelA_2d_ODE_prob(
    u0 = u0,
    solver =Tsit5(),
    tspan = (0.0f0, 10000f0),
    abstol = 1e-6,
    reltol = 1e-6,
    save_start=true,
    save_end=true,
    γ=1.0f0,
    dx=1f0,
)
solC3=0
@time solC3=modelC_2d_ODE_prob(
    u0 = u0,
    solver =Tsit5(),
    tspan = (0.0f0, 10000f0),
    abstol = 1e-4,
    reltol = 1e-4,
    save_start=true,
    save_end=true,
    γ=1.0f0,
    dx=1f0,
)



using DelimitedFiles

writedlm("sims/ModelB/t=0.dat",Array(solB3[1]))
writedlm("sims/ModelB/t=1.dat",Array(solB3[2]))
writedlm("sims/ModelB/t=10.dat",Array(solB3[11]))
writedlm("sims/ModelB/t=100.dat",Array(solB3[101]))
writedlm("sims/ModelB/t=1000.dat",Array(solB3[1001]))
writedlm("sims/ModelB/t=10000.dat",Array(solB3[10001]))


writedlm("sims/ModelA/t=0.dat",Array(solA3[1]))
writedlm("sims/ModelA/t=1.dat",Array(solA3[2]))
writedlm("sims/ModelA/t=10.dat",Array(solA3[11]))
writedlm("sims/ModelA/t=100.dat",Array(solA3[101]))
writedlm("sims/ModelA/t=1000.dat",Array(solA3[1001]))
writedlm("sims/ModelA/t=10000.dat",Array(solA3[10001]))



writedlm("sims/ModelC1/t=0.dat",Array(solC3[1][:,:,1]))
writedlm("sims/ModelC1/t=1.dat",Array(solC3[2][:,:,1]))
writedlm("sims/ModelC1/t=10.dat",Array(solC3[11][:,:,1]))
writedlm("sims/ModelC1/t=100.dat",Array(solC3[101][:,:,1]))
writedlm("sims/ModelC1/t=1000.dat",Array(solC3[1001][:,:,1]))
writedlm("sims/ModelC1/t=10000.dat",Array(solC3[10001][:,:,1]))

writedlm("sims/ModelC2/t=0.dat",Array(solC3[1][:,:,2]))
writedlm("sims/ModelC2/t=1.dat",Array(solC3[2][:,:,2]))
writedlm("sims/ModelC2/t=10.dat",Array(solC3[11][:,:,2]))
writedlm("sims/ModelC2/t=100.dat",Array(solC3[101][:,:,2]))
writedlm("sims/ModelC2/t=1000.dat",Array(solC3[1001][:,:,2]))
writedlm("sims/ModelC2/t=10000.dat",Array(solC3[10001][:,:,2]))



plot([mean(solB3[i]) for i in 1:10001])
plot!([mean(solA3[i]) for i in 1:10001])
plot!([mean(solC3[i][:,:,1]) for i in 1:10001])
plot!([mean(solC3[i][:,:,2]) for i in 1:10001])


writedlm("sims/ModelB/mean.dat",[solB3.t [mean(solB3[i]) for i in 1:10001]])
writedlm("sims/ModelA/mean.dat",[solA3.t [mean(solA3[i]) for i in 1:10001]])
writedlm("sims/ModelC1/mean.dat",[solC3.t [mean(solC3[i][:,:,1]) for i in 1:10001]])
writedlm("sims/ModelC2/mean.dat",[solC3.t [mean(solC3[i][:,:,2]) for i in 1:10001]])




plot([mean(solB3[i]) for i in 1:10001])



plot([mean(solC3[i][:,:,2]) for i in 1:10001])
heatmap(Array(solA3[600]),aspect_ratio=1)
solC3[1]
heatmap(Array(solC3[10001][:,:,1]),aspect_ratio=1)


solB
plot(x->Ufun(x),-4,4)
1
du0=similar(u0)
function Ufun(x)
    -x + x^3/6
end
modelB_2d_loop_GPU(du0, u0, Ufun,1f0)