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

function langevin_2d_loop_GPU(du, u, p, t)
	alpha, dx,σ0 = p
	# alpha = alpha / dx^2
	threads = (8, 8,8)
	blocks = cld.((N,N,M), threads)
	@cuda blocks = blocks threads = threads update_2d_langevin!(du, u, alpha, dx,σ0)
end

function init_langevin_2d(xyd)
	N = length(xyd)
	u = zeros(myT, N, N,M)
	for I in CartesianIndices((N, N,M))
		x = xyd[I[1]]
		y = xyd[I[2]]
		u[I] = randn()
	end
	return u
end
