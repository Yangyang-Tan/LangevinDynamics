using DifferentialEquations
using Plots
f(du, u, p, t) = (du .= u)
g(du, u, p, t) = (du .= 0)
u0 = rand(4, 2)

W = WienerProcess(0.0, 0.0, 0.0)
prob = SDEProblem(f, g, u0, (0.0, 20.0), noise = W)
sol = solve(prob, SRIW1())
plot(sol)
1