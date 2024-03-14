limitbound(a, n) =
    if a == n + 1
        1
    elseif a == 0
        n
    else
        a
    end
# export limitbound

function update_0d_langevin!(dσ, σ, γ, m2, λ, J)
    k1 = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k2 = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    # k = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    # N = size(σ, 1)
    M1 = size(σ, 1)
    M2 = size(σ, 2)
    # M=blockDim().z *gridDim().z
    if (k1 > 0 && k1 <= M1 && k2 > 0 && k2 <= M2)
        # ip1, im1= limitbound(i + 1, N),limitbound(i - 1, N)
        @inbounds dσ[k1, k2, 1] = σ[k1, k2, 2]
        @inbounds dσ[k1, k2, 2] =
            -δUδσ(σ[k1, k2, 1]; m2 = m2, λ = λ, J = J) - γ * σ[k1, k2, 2]
        # @inbounds dσ[i, j] =i+j
    end
    return nothing
end
export update_0d_langevin!



function update_0d_tex_langevin!(dσ, σ, tex, γ, m2, λ, J)
    k1 = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k2 = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    # k = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    # N = size(σ, 1)
    M1 = size(σ, 1)
    M2 = size(σ, 2)
    # M=blockDim().z *gridDim().z
    if (k1 > 0 && k1 <= M1 && k2 > 0 && k2 <= M2)
        # ip1, im1= limitbound(i + 1, N),limitbound(i - 1, N)
        @inbounds dσ[k1, k2, 1] = σ[k1, k2, 2]
        @inbounds dσ[k1, k2, 2] =
            -sign(σ[k1, k2, 1]) * tex[(abs(σ[k1, k2, 1])/0.005f0)+1] - γ * σ[k1, k2, 2]
        # @inbounds dσ[i, j] =i+j
    end
    return nothing
end
export update_0d_tex_langevin!

# abs(σ[k1,k2, 1])/0.005f0 +1

function update_1d_langevin!(dσ, σ, γ, m2, λ, J)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    N = size(σ, 1)
    M = size(σ, 2)
    # M=blockDim().z *gridDim().z
    if (i > 0 && i <= N && k > 0 && k <= M)
        ip1, im1 = limitbound(i + 1, N), limitbound(i - 1, N)
        @inbounds dσ[i, k, 1] = σ[i, k, 2]
        @inbounds dσ[i, k, 2] =
            (σ[im1, k, 1] + σ[ip1, k, 1] - 2 * σ[i, k, 1]) -
            δUδσ(σ[i, k, 1]; m2 = m2, λ = λ, J = J) - γ * σ[i, k, 2]
        # @inbounds dσ[i, j] =i+j
    end
    return nothing
end
export update_1d_langevin!


function update_1d_tex_langevin!(dσ, σ, tex, γ, m2, λ, J)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    N = size(σ, 1)
    M = size(σ, 2)
    # M=blockDim().z *gridDim().z
    if (i > 0 && i <= N && k > 0 && k <= M)
        ip1, im1 = limitbound(i + 1, N), limitbound(i - 1, N)
        @inbounds dσ[i, k, 1] = σ[i, k, 2]
        @inbounds dσ[i, k, 2] =
            (σ[im1, k, 1] + σ[ip1, k, 1] - 2 * σ[i, k, 1]) + 0.22f0 -
            sign(σ[i, k, 1]) * tex[(abs(σ[i, k, 1])/0.005f0)+1] - γ * σ[i, k, 2]
        # @inbounds dσ[i, j] =i+j
    end
    return nothing
end
export update_1d_tex_langevin!



function update_3d_langevin!(dσ, σ, fun, γ, m2, λ, J)
    N = size(σ, 1)
    M = size(σ, 4)
    dx = 1.0f0
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    # z = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    # id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    #     stride = blockDim().x * gridDim().x
    cind = CartesianIndices((N, N, N))
    # M=blockDim().z *gridDim().z
    if (id > 0 && id <= N^3 && k > 0 && k <= M)
        x = cind[id][1]
        y = cind[id][2]
        z = cind[id][3]
        xp1, xm1 = limitbound(x + 1, N), limitbound(x - 1, N)
        yp1, ym1 = limitbound(y + 1, N), limitbound(y - 1, N)
        zp1, zm1 = limitbound(z + 1, N), limitbound(z - 1, N)
        @inbounds dσ[x, y, z, k, 1] = σ[x, y, z, k, 2]
        @inbounds dσ[x, y, z, k, 2] =
            (
                σ[xp1, y, z, k, 1] +
                σ[xm1, y, z, k, 1] +
                σ[x, yp1, z, k, 1] +
                σ[x, ym1, z, k, 1] +
                σ[x, y, zp1, k, 1] +
                σ[x, y, zm1, k, 1] - 6 * σ[x, y, z, k, 1]
            ) / dx^2 - fun(σ[x, y, z, k, 1]) - γ * σ[x, y, z, k, 2]
    end
    #  + σ[i, jm1, k, 1] + σ[i, jp1, k, 1] -
    #     4 * σ[i, j, k, 1]) + 0.22f0 - sign(σ[i,k, 1])*tex[(abs(σ[i,k, 1])/0.005f0)+1] - γ * σ[i, k, 2]
    # @inbounds dσ[i, j] =i+j
    return nothing
end
export update_3d_langevin!




function update_3d_simple_langevin!(dσ, σ, fun)
    N = size(σ, 1)
    # M = size(σ, 4)
    dx = 1
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    # z = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    # id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    #     stride = blockDim().x * gridDim().x
    cind = CartesianIndices(σ)
    # M=blockDim().z *gridDim().z
    for i = id:blockDim().x*gridDim().x:prod(size(σ))
        # x = cind[i][1]
        # y = cind[i][2]
        # z = cind[i][3]
        # k = cind[i][4]
        x, y, z, k = Tuple(cind[i])
        xp1, xm1 = limitbound(x + 1, N), limitbound(x - 1, N)
        yp1, ym1 = limitbound(y + 1, N), limitbound(y - 1, N)
        zp1, zm1 = limitbound(z + 1, N), limitbound(z - 1, N)
        @inbounds dσ[x, y, z, k] =     #dσ = -δF/δσ
            (
                σ[xp1, y, z, k] +
                σ[xm1, y, z, k] +
                σ[x, yp1, z, k] +
                σ[x, ym1, z, k] +
                σ[x, y, zp1, k] +
                σ[x, y, zm1, k] - 6 * σ[x, y, z, k]
            ) / dx^2 - fun(σ[x, y, z, k]) #fun is δU/δσ
    end
    #  + σ[i, jm1, k, 1] + σ[i, jp1, k, 1] -
    #     4 * σ[i, j, k, 1]) + 0.22f0 - sign(σ[i,k, 1])*tex[(abs(σ[i,k, 1])/0.005f0)+1] - γ * σ[i, k, 2]
    # @inbounds dσ[i, j] =i+j
    return nothing
end
export update_3d_simple_langevin!

function update_3d_simple_tex_langevin!(dσ, σ, tex, x_ini, x_step)
    N = size(σ, 1)
    dx = 1
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    cind = CartesianIndices(σ)
    for i = id:blockDim().x*gridDim().x:prod(size(σ))
        x, y, z, k = Tuple(cind[i])
        xp1, xm1 = limitbound(x + 1, N), limitbound(x - 1, N)
        yp1, ym1 = limitbound(y + 1, N), limitbound(y - 1, N)
        zp1, zm1 = limitbound(z + 1, N), limitbound(z - 1, N)
        @inbounds dσ[x, y, z, k] =     #dσ = -δF/δσ
            (
                σ[xp1, y, z, k] +
                σ[xm1, y, z, k] +
                σ[x, yp1, z, k] +
                σ[x, ym1, z, k] +
                σ[x, y, zp1, k] +
                σ[x, y, zm1, k] - 6 * σ[x, y, z, k]
            ) / dx^2 - tex[(((σ[x, y, z, k] - x_ini) / x_step) +1f0)]
    end
    return nothing
end
export update_3d_simple_tex_langevin!

function langevin_3d_loop_simple_tex_GPU(dσ, σ, fun, x_ini, x_step)
    threads = 1024
    blocks = 2^8
    @cuda blocks = blocks threads = threads update_3d_simple_tex_langevin!(
        dσ,
        σ,
        fun,
        x_ini,
        x_step,
    )
end
export langevin_3d_loop_simple_tex_GPU








function update_3d_tex_langevin!(dσ, σ, tex, γ, m2, λ, J)
    N = size(σ, 1)
    M = size(σ, 4)
    dx = 0.2f0
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    # z = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    # id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    #     stride = blockDim().x * gridDim().x
    cind = CartesianIndices((N, N, N))
    # M=blockDim().z *gridDim().z
    if (id > 0 && id <= N^3 && k > 0 && k <= M)
        x = cind[id][1]
        y = cind[id][2]
        z = cind[id][3]
        xp1, xm1 = limitbound(x + 1, N), limitbound(x - 1, N)
        yp1, ym1 = limitbound(y + 1, N), limitbound(y - 1, N)
        zp1, zm1 = limitbound(z + 1, N), limitbound(z - 1, N)
        @inbounds dσ[x, y, z, k, 1] = σ[x, y, z, k, 2]
        @inbounds dσ[x, y, z, k, 2] =
            (
                (
                    σ[xp1, y, z, k, 1] +
                    σ[xm1, y, z, k, 1] +
                    σ[x, yp1, z, k, 1] +
                    σ[x, ym1, z, k, 1] +
                    σ[x, y, zp1, k, 1] +
                    σ[x, y, zm1, k, 1] - 6 * σ[x, y, z, k, 1]
                ) / dx^2 + 0.22f0 -
                sign(σ[x, y, z, k, 1]) * tex[(abs(σ[x, y, z, k, 1])/0.05f0)+1]
            ) - γ * σ[x, y, z, k, 2]
    end
    #  + σ[i, jm1, k, 1] + σ[i, jp1, k, 1] -
    #     4 * σ[i, j, k, 1]) + 0.22f0 - sign(σ[i,k, 1])*tex[(abs(σ[i,k, 1])/0.005f0)+1] - γ * σ[i, k, 2]
    # @inbounds dσ[i, j] =i+j
    return nothing
end
export update_3d_tex_langevin!

function update_2d_langevin!(dσ, σ, γ, m2, λ, J)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    N = size(σ, 1)
    M = size(σ, 3)
    # M=blockDim().z *gridDim().z
    if (i > 0 && i <= N && j > 0 && j <= N && k > 0 && k <= M)
        ip1, im1, jp1, jm1 = limitbound(i + 1, N),
        limitbound(i - 1, N),
        limitbound(j + 1, N),
        limitbound(j - 1, N)
        @inbounds dσ[i, j, k, 1] = σ[i, j, k, 2]
        @inbounds dσ[i, j, k, 2] =
            (
                σ[im1, j, k, 1] + σ[ip1, j, k, 1] + σ[i, jm1, k, 1] + σ[i, jp1, k, 1] -
                4 * σ[i, j, k, 1]
            ) - δUδσ(σ[i, j, k, 1]; m2 = m2, λ = λ, J = J) - γ * σ[i, j, k, 2]
        # @inbounds dσ[i, j] =i+j
    end
    return nothing
end
export update_2d_langevin!



function update_2d_pure_langevin_nonlocal!(dσ, σ, γσ, m2, λ, J)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    N = size(σ, 1)
    M = size(σ, 3)
    # M=blockDim().z *gridDim().z
    if (i > 0 && i <= N && j > 0 && j <= N && k > 0 && k <= M)
        ip1, im1, jp1, jm1 = limitbound(i + 1, N),
        limitbound(i - 1, N),
        limitbound(j + 1, N),
        limitbound(j - 1, N)
        @inbounds dσ[i, j, k, 1] = σ[i, j, k, 2]
        @inbounds dσ[i, j, k, 2] =
            (
                σ[im1, j, k, 1] + σ[ip1, j, k, 1] + σ[i, jm1, k, 1] + σ[i, jp1, k, 1] -
                4 * σ[i, j, k, 1]
            ) - δUδσ(σ[i, j, k, 1]; m2 = m2, λ = λ, J = J) - γσ[i, j, k]
        # @inbounds dσ[i, j] =i+j
    end
    return nothing
end
export update_2d_pure_langevin_nonlocal!


# function update_iso_langevin!(dσ, σ, d, γ, m2, λ, J)
#     i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     k = (blockIdx().y - 1) * blockDim().y + threadIdx().y
#     N = size(σ, 1)
#     M = size(σ, 2)
#     dx=0.1
#     # M=blockDim().z *gridDim().z
#     if (i > 1 && i <= N - 1 && k > 0 && k <= M)
#         # ip1, im1, jp1, jm1 = limitbound(i + 1, N),
#         # limitbound(i - 1, N), limitbound(j + 1, N),
#         # limitbound(j - 1, N)
#         @inbounds dσ[i, k, 1] = σ[i, k, 2]
# @inbounds dσ[i, k, 2] =
#     (σ[i-1, k, 1] + σ[i+1, k, 1] - 2 * σ[i, k, 1])/(dx^2) +
#     (σ[i+1, k, 1] - σ[i-1, k, 1]) * (d - 1) / (2 *dx^2* (i - 1)) -
#     δUδσ(σ[i, k, 1]; m2 = m2, λ = λ, J = J) - γ * σ[i, k, 2]
#     elseif (i == 1 && k > 0 && k <= M)
#         @inbounds dσ[i, k, 1] = σ[i, k, 2]
#         @inbounds dσ[i, k, 2] =
#             (2 * σ[i+1, k, 1] - 2 * σ[i, k, 1])/(dx^2) - δUδσ(σ[i, k, 1]; m2 = m2, λ = λ, J = J) -
#             γ * σ[i, k, 2]
#     elseif (i == N && k > 0 && k <= M)
#         @inbounds dσ[i, k, 1] = σ[i, k, 2]
#         @inbounds dσ[i, k, 2] =
#             (2 * σ[i-1, k, 1] - 2 * σ[i, k, 1])/(dx^2) - δUδσ(σ[i, k, 1]; m2 = m2, λ = λ, J = J) -
#             γ * σ[i, k, 2]
#     end
#     return nothing
# end
# export update_iso_langevin!

# @inline function kernel_1D_langevin!(dσ, σ, II, I, t,N,p)
#     alpha, σ0, h, dx = p
#     i, k = Tuple(I)
#     ip1 = limitbound(i + 1, N)
#     im1 = limitbound(i - 1, N)
#     dσ[II[i, k]] =
#         alpha * ((σ[II[im1, k]] + σ[II[ip1, k]] - 2 * σ[II[i, k]]) / (dx^2) -
#         δUδσ(σ[II[i, k]]; σ0=σ0, h=h))
#     return nothing
# end

# @inline function kernel_2D_langevin!(dσ, σ, II, I, t,N,p)
#     alpha, σ0, h, dx = p
#     i, j, k = Tuple(I)
#     ip1, im1, jp1, jm1 = limitbound(i + 1, N), limitbound(i - 1, N), limitbound(j + 1, N), limitbound(j - 1, N)
#     dσ[II[i, j, k]] =
#         alpha * ((
#             σ[II[im1, j, k]] + σ[II[ip1, j, k]] + σ[II[i, jm1, k]] + σ[II[i, jp1, k]] -
#             4 * σ[II[i, j, k]]
#         ) / (dx^2) - δUδσ(σ[II[i, j, k]]; σ0=σ0, h=h))
#     return nothing
# end

# function langevin_1d!(dσ, σ, p, t)
#     @inbounds begin
#       II = LinearIndices((N, M))
#       kernel_1D_langevin!.(Ref(dσ), Ref(σ), Ref(II), CartesianIndices((N, M)), t,N, Ref(p))
#     #   kernel_1D!.(Ref(du), Ref(u), α, Ref(II), CartesianIndices((N, M)), t)
#       return nothing
#     end
# end
# export langevin_1d!
# function langevin_2d!(dσ, σ, p, t)
#     @inbounds begin
#       II = LinearIndices((N, N,M))
#       kernel_2D_langevin!.(Ref(dσ), Ref(σ), Ref(II), CartesianIndices((N, N,M)), t,N, Ref(p))
#     #   kernel_1D!.(Ref(du), Ref(u), α, Ref(II), CartesianIndices((N, M)), t)
#       return nothing
#     end
# end
# export langevin_2d!

# function update_1d_langevin!(dσ, σ, alpha, dx, σ0, h)
#     i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
#     N = size(σ, 1)
#     M = size(σ, 2)
#     # M=blockDim().z *gridDim().z
#     if (i > 0 && i <= N && k > 0 && k <= M)
#         ip1, im1 = limitbound(i + 1, N), limitbound(i - 1, N)
#         @inbounds dσ[i, k] =alpha * ((σ[im1, k] + σ[ip1, k] - 2 * σ[i, k]) / (dx^2)- δUδσ(σ[i, k]; σ0=σ0, h=h))
#         # @inbounds dσ[i, j] =i+j
#     end
#     return nothing
# end
# export update_1d_langevin!

# function update_2d_flat_langevin!(dσ, σ, alpha, dx, σ0)
#     id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     stride = blockDim().x * gridDim().x
#     cind = CartesianIndices((N, N, M))
#     for l in id:stride:(N * N * M)
#         i = cind[l][1]
#         j = cind[l][2]
#         k = cind[l][3]
#         ip1, im1, jp1, jm1 = limitbound(i + 1, N),
#         limitbound(i - 1, N), limitbound(j + 1, N),
#         limitbound(j - 1, N)
#         @inbounds dσ[i, j, k] =
#             alpha *
#             (σ[im1, j, k] + σ[ip1, j, k] + σ[i, jm1, k] + σ[i, jp1, k] - 4 * σ[i, j, k]) /
#             (dx^2) - δUδσ(σ[i, j, k]; σ0=σ0)
#     end
#     return nothing
# end
# export update_2d_flat_langevin!

# function update_2d_flat2_langevin!(dσ, σ, alpha, dx, σ0)
#     idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
#     idz = (blockIdx().z - 1) * blockDim().z + threadIdx().z
#     strx = blockDim().x * gridDim().x
#     stry = blockDim().y * gridDim().y
#     strz = blockDim().z * gridDim().z
#     for k in idz:strz:M
#         for j in idy:stry:N
#             for i in idx:strx:N
#                 ip1, im1, jp1, jm1 = limitbound(i + 1, N),
#                 limitbound(i - 1, N), limitbound(j + 1, N),
#                 limitbound(j - 1, N)
#                 @inbounds dσ[i, j, k] =
#                     alpha * (
#                         σ[im1, j, k] + σ[ip1, j, k] + σ[i, jm1, k] + σ[i, jp1, k] -
#                         4 * σ[i, j, k]
#                     ) / (dx^2) - δUδσ(σ[i, j, k]; σ0=σ0)
#             end
#         end
#     end
#     return nothing
# end
# export update_2d_flat2_langevin!

# function update_2d_shem_langevin!(dσ, σ, alpha, dx, σ0)
#     i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
#     k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
#     ti = threadIdx().x
#     tj = threadIdx().y
#     tk = threadIdx().z
#     σ_l = CuDynamicSharedArray(myT, (blockDim().x, blockDim().y, blockDim().z))
#     if (i > 0 && i <= N && j > 0 && j <= N && k > 0 && k <= M)
#         # @inbounds σ_l[ti,tj,tk] = σ[i,j,k]
#         ip1, im1, jp1, jm1 = limitbound(ti + 1, 10),
#         limitbound(ti - 1, 10), limitbound(tj + 1, 10),
#         limitbound(tj - 1, 10)
#         @inbounds σ_l[ti, tj, tk] =
#             alpha * (
#                 σ_l[im1, tj, tk] + σ_l[ip1, tj, tk] + σ_l[ti, jm1, tk] + σ_l[ti, jp1, tk] -
#                 4 * σ_l[ti, tj, tk]
#             ) / (dx^2) - δUδσ(σ_l[ti, tj, tk]; σ0=σ0)
#         # @inbounds dσ[i, j] =i+j
#     end
#     return nothing
# end
# export update_2d_shem_langevin!

# function update_2d_flat_shem_langevin!(dσ, σ, alpha, dx, σ0)
#     i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
#     idk = (blockIdx().z - 1) * blockDim().z + threadIdx().z
#     ti = threadIdx().x
#     tj = threadIdx().y
#     σ_l = CuDynamicSharedArray(myT, (blockDim().x, blockDim().y))
#     bdim = blockDim().x
#     strz = blockDim().z * gridDim().z
#     for k in idk:strz:M
#         @inbounds σ_l[ti, tj] = σ[i, j, k]
#         ip1, im1, jp1, jm1 = limitbound(ti + 1, bdim),
#         limitbound(ti - 1, bdim), limitbound(tj + 1, bdim),
#         limitbound(tj - 1, bdim)
#         @inbounds σ[i, j, k] =
#             alpha *
#             (σ_l[im1, tj] + σ_l[ip1, tj] + σ_l[ti, jm1] + σ_l[ti, jp1] - 4 * σ_l[ti, tj]) /
#             (dx^2) - δUδσ(σ_l[ti, tj]; σ0=σ0)
#     end
#     return nothing
# end
# export update_2d_flat_shem_langevin!

# function langevin_1d_loop_GPU(dσ, σ, p, t)
#     alpha, σ0, h, dx = p
#     # alpha = alpha / dx^2
#     threads = (512, 1)
#     blocks = cld.((N, M), threads)
#     @cuda blocks = blocks threads = threads update_1d_langevin!(dσ, σ, alpha, dx, σ0, h)
# end
# export langevin_1d_loop_GPU


function langevin_0d_tex_loop_GPU(dσ, σ, tex, p, t)
    γ, m2, λ, J = p
    # alpha = alpha / dx^2
    threads = (32, 32)
    blocks = cld.((M, M), threads)
    @cuda blocks = blocks threads = threads update_0d_tex_langevin!(dσ, σ, tex, γ, m2, λ, J)
end
export langevin_0d_tex_loop_GPU


function langevin_0d_loop_GPU(dσ, σ, p, t)
    γ, m2, λ, J = p
    # alpha = alpha / dx^2
    threads = (32, 32)
    blocks = cld.((M, M), threads)
    @cuda blocks = blocks threads = threads update_0d_langevin!(dσ, σ, γ, m2, λ, J)
end
export langevin_0d_loop_GPU



function langevin_1d_tex_loop_GPU(dσ, σ, tex, p, t)
    γ, m2, λ, J = p
    # alpha = alpha / dx^2
    threads = (64, 16)
    blocks = cld.((N, M), threads)
    @cuda blocks = blocks threads = threads update_1d_tex_langevin!(dσ, σ, tex, γ, m2, λ, J)
end
export langevin_1d_tex_loop_GPU

function langevin_3d_tex_loop_GPU(dσ, σ, tex, p, t)
    γ, m2, λ, J = p
    # alpha = alpha / dx^2
    N = size(σ, 1)
    M = size(σ, 4)
    threads = (32, 32)
    blocks = cld.((N^3, M), threads)
    @cuda blocks = blocks threads = threads update_3d_tex_langevin!(dσ, σ, tex, γ, m2, λ, J)
end
export langevin_3d_tex_loop_GPU



function langevin_3d_loop_GPU(dσ, σ, fun, p, t)
    γ, m2, λ, J = p
    # alpha = alpha / dx^2
    N = size(σ, 1)
    M = size(σ, 4)
    threads = (512, 2)
    blocks = cld.((N^3, M), threads)
    @cuda blocks = blocks threads = threads update_3d_langevin!(dσ, σ, fun, γ, m2, λ, J)
end
export langevin_3d_loop_GPU

function langevin_3d_loop_simple_GPU(dσ, σ, fun)
    # alpha = alpha / dx^2
    # N = size(σ, 1)
    # M = size(σ, 4)
    threads = 1024
    blocks = 2^8
    @cuda blocks = blocks threads = threads maxregs = 4 update_3d_simple_langevin!(
        dσ,
        σ,
        fun,
    )
end
export langevin_3d_loop_simple_GPU






# function langevin_3d_tex_loop_GPU(dσ, σ, tex, p, t)
#     γ, m2, λ, J = p
#     # alpha = alpha / dx^2
#     # dev=KernelAbstractions.get_device(σ)
#     # n = dev isa GPU ? 512 : 8
#     kernel! = update_3d_tex_langevin!(CUDADevice(), 512)
#     ev = kernel!(dσ, σ, tex, γ, m2, λ, J, ndrange = size(σ)[1:end-1])
#     wait(ev)
# end
# export langevin_3d_tex_loop_GPU


function langevin_1d_loop_GPU(dσ, σ, p, t)
    γ, m2, λ, J = p
    # alpha = alpha / dx^2
    N = size(σ, 1)
    M = size(σ, 2)
    threads = (64, 16)
    blocks = cld.((N, M), threads)
    @cuda blocks = blocks threads = threads update_1d_langevin!(dσ, σ, γ, m2, λ, J)
end
export langevin_1d_loop_GPU

function langevin_2d_loop_GPU(dσ, σ, p, t)
    γ, m2, λ, J = p
    # alpha = alpha / dx^2
    threads = (16, 16, 4)
    N = size(σ, 1)
    M = size(σ, 3)
    blocks = cld.((N, N, M), threads)
    @cuda blocks = blocks threads = threads update_2d_langevin!(dσ, σ, γ, m2, λ, J)
end
export langevin_2d_loop_GPU


function langevin_2d_loop_nonlocal_GPU(dσ, σ, γσ_cache, p, Zt_fft, u_2D_cache)
    γσ_cache .= @view σ[:, :, :, 2]
    fastconv(γσ_cache, Zt_fft, u_2D_cache; dims = 1:2)
    m2, λ, J = p
    # alpha = alpha / dx^2
    threads = (16, 16, 4)
    N = size(σ, 1)
    M = size(σ, 3)
    blocks = cld.((N, N, M), threads)
    @cuda blocks = blocks threads = threads update_2d_pure_langevin_nonlocal!(
        dσ,
        σ,
        γσ_cache,
        m2,
        λ,
        J,
    )
end
export langevin_2d_loop_nonlocal_GPU

function fastconv(u, Z, u_2D_catch; dims = 1:3)
    CUDA.@sync copy!(u_2D_catch, u)
    CUDA.@sync fft!(u_2D_catch, dims)
    CUDA.@sync u_2D_catch .= u_2D_catch .* Z
    CUDA.@sync ifft!(u_2D_catch, dims)
    CUDA.@sync u .= real.(u_2D_catch)
    return nothing
    # CUDA.@sync copy!(u, a_2D_catch)
end
export fastconv

# function langevin_iso_loop_GPU(dσ, σ, p, t)
#     d, γ, m2, λ, J = p
#     # alpha = alpha / dx^2
#     threads = (512, 1)
#     blocks = cld.((N, M), threads)
#     @cuda blocks = blocks threads = threads update_iso_langevin!(dσ, σ, d, γ, m2, λ, J)
# end
# export langevin_iso_loop_GPU

# function langevin_2d_loop_flat_GPU(dσ, σ, p, t)
#     alpha, dx, σ0 = p
#     # alpha = alpha / dx^2
#     threads = 1024
#     blocks = 1048
#     @cuda blocks = blocks threads = threads update_2d_flat_langevin!(dσ, σ, alpha, dx, σ0)
# end
# export langevin_2d_loop_flat_GPU

# function langevin_2d_loop_flat2_GPU(dσ, σ, p, t)
#     alpha, dx, σ0 = p
#     # alpha = alpha / dx^2
#     threads = (16, 16, 2)
#     blocks = (1, 1, 8)
#     @cuda blocks = blocks threads = threads update_2d_flat2_langevin!(dσ, σ, alpha, dx, σ0)
# end
# export langevin_2d_loop_flat2_GPU

# function langevin_2d_loop_shem_GPU(dσ, σ, p, t)
#     alpha, dx, σ0 = p
#     # alpha = alpha / dx^2
#     threads = (32, 32, 1)
#     blocks = cld.((N, N, M), threads)
#     @cuda blocks = blocks threads = threads shmem = prod(threads) * sizeof(Float32) update_2d_shem_langevin!(
#         dσ, σ, alpha, dx, σ0
#     )
# end
# export langevin_2d_loop_shem_GPU

# function langevin_2d_loop_flat_shem_GPU(dσ, σ, p, t)
#     alpha, dx, σ0 = p
#     # alpha = alpha / dx^2
#     threads = (32, 32, 1)
#     blocks = cld.((N, N, 128), (32, 32, 1))
#     @cuda blocks = blocks threads = threads shmem = prod(threads) * sizeof(Float32) update_2d_flat_shem_langevin!(
#         dσ, σ, alpha, dx, σ0
#     )
# end
# export langevin_2d_loop_flat_shem_GPU
