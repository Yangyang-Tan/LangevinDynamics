limit(a, n) =
    if a == n + 1
        1
    elseif a == 0
        n
    else
        a
    end

function update_2d_langevin!(dσ, σ, alpha, dx, σ0,h)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    N=size(σ,1)
    M=size(σ,3)
    # M=blockDim().z *gridDim().z
    if (i > 0 && i <= N && j > 0 && j <= N && k > 0 && k <= M)
        ip1, im1, jp1, jm1 = limit(i + 1, N),
        limit(i - 1, N), limit(j + 1, N),
        limit(j - 1, N)
        @inbounds dσ[i, j, k] =
            alpha *
            (σ[im1, j, k] + σ[ip1, j, k] + σ[i, jm1, k] + σ[i, jp1, k] - 4 * σ[i, j, k]) /
            (dx^2) - δUδσ(σ[i, j, k]; σ0=σ0,h=h)
        # @inbounds dσ[i, j] =i+j
    end
    return nothing
end
export update_2d_langevin!

function update_2d_flat_langevin!(dσ, σ, alpha, dx, σ0)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    cind = CartesianIndices((N, N, M))
    for l in id:stride:(N * N * M)
        i = cind[l][1]
        j = cind[l][2]
        k = cind[l][3]
        ip1, im1, jp1, jm1 = limit(i + 1, N),
        limit(i - 1, N), limit(j + 1, N),
        limit(j - 1, N)
        @inbounds dσ[i, j, k] =
            alpha *
            (σ[im1, j, k] + σ[ip1, j, k] + σ[i, jm1, k] + σ[i, jp1, k] - 4 * σ[i, j, k]) /
            (dx^2) - δUδσ(σ[i, j, k]; σ0=σ0)
    end
    return nothing
end
export update_2d_flat_langevin!

function update_2d_flat2_langevin!(dσ, σ, alpha, dx, σ0)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    idz = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    strx = blockDim().x * gridDim().x
    stry = blockDim().y * gridDim().y
    strz = blockDim().z * gridDim().z
    for k in idz:strz:M
        for j in idy:stry:N
            for i in idx:strx:N
                ip1, im1, jp1, jm1 = limit(i + 1, N),
                limit(i - 1, N), limit(j + 1, N),
                limit(j - 1, N)
                @inbounds dσ[i, j, k] =
                    alpha * (
                        σ[im1, j, k] + σ[ip1, j, k] + σ[i, jm1, k] + σ[i, jp1, k] -
                        4 * σ[i, j, k]
                    ) / (dx^2) - δUδσ(σ[i, j, k]; σ0=σ0)
            end
        end
    end
    return nothing
end
export update_2d_flat2_langevin!

function update_2d_shem_langevin!(dσ, σ, alpha, dx, σ0)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    ti = threadIdx().x
    tj = threadIdx().y
    tk = threadIdx().z
    σ_l = CuDynamicSharedArray(myT, (blockDim().x, blockDim().y, blockDim().z))
    if (i > 0 && i <= N && j > 0 && j <= N && k > 0 && k <= M)
        # @inbounds σ_l[ti,tj,tk] = σ[i,j,k]
        ip1, im1, jp1, jm1 = limit(ti + 1, 10),
        limit(ti - 1, 10), limit(tj + 1, 10),
        limit(tj - 1, 10)
        @inbounds σ_l[ti, tj, tk] =
            alpha * (
                σ_l[im1, tj, tk] + σ_l[ip1, tj, tk] + σ_l[ti, jm1, tk] + σ_l[ti, jp1, tk] -
                4 * σ_l[ti, tj, tk]
            ) / (dx^2) - δUδσ(σ_l[ti, tj, tk]; σ0=σ0)
        # @inbounds dσ[i, j] =i+j
    end
    return nothing
end
export update_2d_shem_langevin!

function update_2d_flat_shem_langevin!(dσ, σ, alpha, dx, σ0)
	i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
	idk = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    ti = threadIdx().x
    tj = threadIdx().y
    σ_l = CuDynamicSharedArray(myT, (blockDim().x,blockDim().y))
	bdim=blockDim().x
	strz = blockDim().z * gridDim().z
    for k in idk:strz:M
        @inbounds σ_l[ti,tj] = σ[i, j, k]
        ip1, im1, jp1, jm1 = limit(ti + 1, bdim),
        limit(ti - 1, bdim), limit(tj + 1, bdim),
        limit(tj - 1, bdim)
        @inbounds σ[i, j, k] =
            alpha * (
                σ_l[im1, tj] + σ_l[ip1, tj] + σ_l[ti, jm1] + σ_l[ti, jp1] -
                4 * σ_l[ti, tj]
            ) / (dx^2) - δUδσ(σ_l[ti, tj]; σ0=σ0)
    end
    return nothing
end
export update_2d_flat_shem_langevin!

function langevin_2d_loop_GPU(dσ, σ, p, t)
    alpha, dx, σ0,h = p
    # alpha = alpha / dx^2
    threads = (8, 8, 8)
    blocks = cld.((N, N, M), threads)
    @cuda blocks = blocks threads = threads update_2d_langevin!(dσ, σ, alpha, dx, σ0,h)
end
export langevin_2d_loop_GPU

function langevin_2d_loop_flat_GPU(dσ, σ, p, t)
    alpha, dx, σ0 = p
    # alpha = alpha / dx^2
    threads = 1024
    blocks = 1048
    @cuda blocks = blocks threads = threads update_2d_flat_langevin!(dσ, σ, alpha, dx, σ0)
end
export langevin_2d_loop_flat_GPU

function langevin_2d_loop_flat2_GPU(dσ, σ, p, t)
    alpha, dx, σ0 = p
    # alpha = alpha / dx^2
    threads = (16, 16, 2)
    blocks = (1, 1, 8)
    @cuda blocks = blocks threads = threads update_2d_flat2_langevin!(dσ, σ, alpha, dx, σ0)
end
export langevin_2d_loop_flat2_GPU

function langevin_2d_loop_shem_GPU(dσ, σ, p, t)
    alpha, dx, σ0 = p
    # alpha = alpha / dx^2
    threads = (32, 32, 1)
    blocks = cld.((N, N, M), threads)
    @cuda blocks = blocks threads = threads shmem = prod(threads) * sizeof(Float32) update_2d_shem_langevin!(
        dσ, σ, alpha, dx, σ0
    )
end
export langevin_2d_loop_shem_GPU

function langevin_2d_loop_flat_shem_GPU(dσ, σ, p, t)
    alpha, dx, σ0 = p
    # alpha = alpha / dx^2
    threads = (32,32,1)
    blocks = cld.((N, N,128), (32,32,1))
    @cuda blocks = blocks threads = threads shmem = prod(threads) * sizeof(Float32) update_2d_flat_shem_langevin!(
        dσ, σ, alpha, dx, σ0
    )
end
export langevin_2d_loop_flat_shem_GPU
