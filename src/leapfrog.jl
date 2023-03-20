function update_3d_leapfrog_langevin!(ddσ,dσ, σ, fun, γ, m2, λ, J)
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
        @inbounds dσ[x, y, z, k] = σ[x, y, z, k]
        @inbounds ddσ[x, y, z, k] =
            (
                σ[xp1, y, z, k] +
                σ[xm1, y, z, k] +
                σ[x, yp1, z, k] +
                σ[x, ym1, z, k] +
                σ[x, y, zp1, k] +
                σ[x, y, zm1, k] - 6 * σ[x, y, z, k]
            )/dx^2 - fun(σ[x, y, z, k])
    end
    #  + σ[i, jm1, k, 1] + σ[i, jp1, k, 1] -
    #     4 * σ[i, j, k, 1]) + 0.22f0 - sign(σ[i,k, 1])*tex[(abs(σ[i,k, 1])/0.005f0)+1] - γ * σ[i, k, 2]
    # @inbounds dσ[i, j] =i+j
    return nothing
end
export update_3d_leapfrog_langevin!


function U_kernel!(U, σ, fun)
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
        @inbounds U[x, y, z, k] =
            -(
                σ[xp1, y, z, k] +
                σ[xm1, y, z, k] +
                σ[x, yp1, z, k] +
                σ[x, ym1, z, k] +
                σ[x, y, zp1, k] +
                σ[x, y, zm1, k]
            )*σ[x, y, z, k]/(2*dx^2) + fun(σ[x, y, z, k])
    end
    #  + σ[i, jm1, k, 1] + σ[i, jp1, k, 1] -
    #     4 * σ[i, j, k, 1]) + 0.22f0 - sign(σ[i,k, 1])*tex[(abs(σ[i,k, 1])/0.005f0)+1] - γ * σ[i, k, 2]
    # @inbounds dσ[i, j] =i+j
    return nothing
end
export U_kernel!


function langevin_3d_leapfrog_loop_GPU(ddσ,dσ, σ, fun, p, t)
    γ, m2, λ, J = p
    # alpha = alpha / dx^2
    N = size(σ, 1)
    M = size(σ, 4)
    threads = (1024, 1)
    blocks = cld.((N^3, M), threads)
    @cuda blocks = blocks threads = threads update_3d_leapfrog_langevin!(ddσ,
        dσ,
        σ,
        fun,
        γ,
        m2,
        λ,
        J,
    )
end
export langevin_3d_leapfrog_loop_GPU

function getU!(U,σ,fun)
    # alpha = alpha / dx^2
    N = size(σ, 1)
    M = size(σ, 4)
    threads = (1024, 1)
    blocks = cld.((N^3, M), threads)
    @cuda blocks = blocks threads = threads U_kernel!(U,
        σ,
        fun,
    )
end
export getU!

function getU(σ,fun)
    # alpha = alpha / dx^2
    U=similar(σ)
    N = size(σ, 1)
    M = size(σ, 4)
    threads = (1024, 1)
    blocks = cld.((N^3, M), threads)
    @cuda blocks = blocks threads = threads U_kernel!(U,
        σ,
        fun,
    )
    return U
end
export getU
