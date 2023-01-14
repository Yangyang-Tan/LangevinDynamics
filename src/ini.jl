device!(0)
const N = 100
const M = 100
const xyd_brusselator = range(0; stop = 10, length = N)
myT=Float32
function δUδσ(σ::T; h::T = T(0.0), σ0::T = T(0.1)) where {T}
	return σ * (σ^2 - σ0^2) - h
end
export δUδσ

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
export init_langevin_2d
