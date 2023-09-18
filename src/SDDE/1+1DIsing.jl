using Distributed
using DrWatson, Plots,CurveFit,DifferentialEquations,CUDA,DelimitedFiles,Random
using DifferentialEquations
using Plots
function hayes_modelf(du, u, h, p, t)
    τ, a, b, c, α, β, γ = p
    du .= a .* u .+ b .* h(p, t - τ) .+ c
end
function hayes_modelg(du, u, h, p, t)
    τ, a, b, c, α, β, γ = p
    du .= α .* u .+ γ
end
h(p, t) = (ones(1) .+ t);
tspan = (0.0, 10.0)

pmul = [1.0, -4.0, -2.0, 10.0, -1.3, -1.2, 1.1]
padd = [1.0, -4.0, -2.0, 10.0, -0.0, -0.0, 0.1]

prob = SDDEProblem(hayes_modelf, hayes_modelg, [1.0], h, tspan, pmul;
                   constant_lags = (pmul[1],));
sol = solve(prob, RKMil())
