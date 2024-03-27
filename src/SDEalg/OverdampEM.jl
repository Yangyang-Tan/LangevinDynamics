struct OverdampEM{T} <: StochasticDiffEqAlgorithm
    eta::T
    noise::T
end

struct OverdampSDEProblem{T<:AbstractFloat,N}
    f1::Function
    u0::AbstractArray{T,N}
    tspan::Tuple{T,T}
    p::Union{SciMLBase.NullParameters,NTuple}
end

OverdampSDEProblem(f1, u0, tspan) =
    OverdampSDEProblem(f1, u0, tspan, SciMLBase.NullParameters())
export OverdampSDEProblem


OverdampEM(; eta = 1.0, noise = error("noise not provided")) = OverdampEM(eta, noise)
export OverdampEM




@muladd function DiffEqBase.solve(
    prob::OverdampSDEProblem,
    alg::OverdampEM,
    args...;
    dt = error("dt required for OverdampEM"),
    save_interval = error("save_interval required for OverdampEM"),
)
    GC.gc(true)
    CUDA.reclaim()
    f1 = prob.f1
    # f2 = prob.f.f2
    # g = prob.g
    tspan = prob.tspan
    u1 = prob.u0
    du1 = similar(u1)
    # CUDA.@sync randn!(dW_1)
    # CUDA.@sync randn!(dW_2)
    fdt = dt / alg.eta
    noisedt = sqrt(dt) * alg.noise / sqrt(alg.eta)
    @inbounds begin
        n = round(Int32,(tspan[2] - tspan[1]) / dt) + 1
        # t = [tspan[1] + i * dt for i = 0:(n-1)]
        # sqdt = sqrt(dt)
    end
    m_1 = CUDA.zeros(Float32, size(u1, 4), round(Int32,(tspan[2] - tspan[1])/save_interval) + 1)
    save_j = 1
    for i = 1:n
        if abs((tspan[1] + (i - 1) * dt) % save_interval) < 1e-5 ||
            abs((tspan[1] + (i - 1) * dt) % save_interval - save_interval) < 1e-5
             println("Saving at t = $(tspan[1] + (i-1) * dt)")  # Replace this with your actual save operation
             m_1[:, save_j] = @view mean(u1, dims = [1, 2, 3])[1, 1, 1, :]
             save_j += 1
         end
        #f*dt
        CUDA.@sync f1(du1, u1)
        CUDA.@sync @. u1 += fdt * du1
        # dW 
        CUDA.@sync CUDA.randn!(du1)
        CUDA.@sync @. u1 += noisedt * du1
    end
    return Array(m_1)
    # sol = DiffEqBase.build_solution(prob, alg, t, u, calculate_error = false)
end




# function update_3d_simple_EM_langevin!(dσ, σ, fun,dx)
#     N = size(σ, 1)
#     # M = size(σ, 4)
#     # dx = 1
#     id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     # z = (blockIdx().z - 1) * blockDim().z + threadIdx().z
#     # id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     #     stride = blockDim().x * gridDim().x
#     cind = CartesianIndices(σ)
#     # M=blockDim().z *gridDim().z
#     for i = id:blockDim().x*gridDim().x:prod(size(σ))
#         # x = cind[i][1]
#         # y = cind[i][2]
#         # z = cind[i][3]
#         # k = cind[i][4]
#         x, y, z, k = Tuple(cind[i])
#         xp1, xm1 = limitbound(x + 1, N), limitbound(x - 1, N)
#         yp1, ym1 = limitbound(y + 1, N), limitbound(y - 1, N)
#         zp1, zm1 = limitbound(z + 1, N), limitbound(z - 1, N)
#         @inbounds dσ[x, y, z, k] =     #dσ = -δF/δσ
#             (
#                 σ[xp1, y, z, k] +
#                 σ[xm1, y, z, k] +
#                 σ[x, yp1, z, k] +
#                 σ[x, ym1, z, k] +
#                 σ[x, y, zp1, k] +
#                 σ[x, y, zm1, k] - 6 * σ[x, y, z, k]
#             ) / dx^2 - fun(σ[x, y, z, k]) #fun is δU/δσ
#     end
#     #  + σ[i, jm1, k, 1] + σ[i, jp1, k, 1] -
#     #     4 * σ[i, j, k, 1]) + 0.22f0 - sign(σ[i,k, 1])*tex[(abs(σ[i,k, 1])/0.005f0)+1] - γ * σ[i, k, 2]
#     # @inbounds dσ[i, j] =i+j
#     return nothing
# end
# export update_3d_simple_EM_langevin!


# function langevin_3d_loop_simple_EM_GPU(dσ, σ, fun,dx)
#     # alpha = alpha / dx^2
#     # N = size(σ, 1)
#     # M = size(σ, 4)
#     threads = 1024
#     blocks = 2^8
#     @cuda blocks = blocks threads = threads maxregs = 4 update_3d_simple_EM_langevin!(
#         dσ,
#         σ,
#         fun,
#         dx,
#     )
# end
# export langevin_3d_loop_EM_simple_GPU