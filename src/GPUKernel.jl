limit(a, N) =
	if a == N + 1
		1
	elseif a == 0
		N
	else
		a
	end


function update_2d_langevin!(dσ, σ, alpha, dx,σ0)
	i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
	if (i > 0 && i <= N && j > 0 && j <= N&&k > 0 && k <= M)
		ip1, im1, jp1, jm1 = limit(i + 1, N), limit(i - 1, N), limit(j + 1, N), limit(j - 1, N)
		@inbounds dσ[i, j,k] =
			alpha * (σ[im1, j,k] + σ[ip1, j,k] + σ[i, jm1,k] + σ[i, jp1,k] - 4 * σ[i, j,k]) / (dx^2) -
			δUδσ(σ[i, j,k],σ0=σ0)
		# @inbounds dσ[i, j] =i+j
	end
	return nothing
end
export update_2d_langevin!


function update_2d_shem_langevin!(dσ, σ, alpha, dx,σ0)
	i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
	ti = threadIdx().x
    tj = threadIdx().y
	tk = threadIdx().z
    σ_l = CuDynamicSharedArray(Float32, (blockDim().x, blockDim().y,blockDim().z))
	@inbounds σ_l[ti,tj,tk] = σ[i,j,k]
	sync_threads()
	if (i > 0 && i <= N && j > 0 && j <= N&&k > 0 && k <= M)
		ip1, im1, jp1, jm1 = limit(ti + 1, N), limit(ti - 1, N), limit(tj + 1, N), limit(tj - 1, N)
		@inbounds dσ[i, j,k] =
			alpha * (σ_l[im1, tj,tk] + σ_l[ip1, tj,tk] + σ_l[ti, jm1,tk] + σ_l[ti, jp1,tk] - 4 * σ_l[ti, tj,tk]) / (dx^2) -
			δUδσ(σ_l[ti, tj,tk],σ0=σ0)
		# @inbounds dσ[i, j] =i+j
	end
	return nothing
end
export update_2d_shem_langevin!

function langevin_2d_loop_GPU(dσ, σ, p, t)
	alpha, dx,σ0 = p
	# alpha = alpha / dx^2
	threads = (8, 8,8)
	blocks = cld.((N,N,M), threads)
	@cuda blocks = blocks threads = threads update_2d_langevin!(dσ, σ, alpha, dx,σ0)
end
export langevin_2d_loop_GPU

function langevin_2d_loop_shem_GPU(dσ, σ, p, t)
	alpha, dx,σ0 = p
	# alpha = alpha / dx^2
	threads = (8, 8,8)
	blocks = cld.((N,N,M), threads)
	@cuda blocks=blocks threads=threads shmem=prod(threads)*sizeof(Float32) update_2d_shem_langevin!(dσ, σ, alpha, dx,σ0)
end
export langevin_2d_loop_shem_GPU
