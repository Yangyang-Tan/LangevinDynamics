device!(0)
N = 32*2^1
M = 2^0
xyd_brusselator = range(0; stop = 10, length = N)
const myT=Float32
# function δUδσ(σ::T; h::T = 0.0f0, σ0::T = 0.1f0) where {T}
# 	return σ * (σ^2 - σ0^2) - h
# 	# return σ * σ0 - h
# end
# export δUδσ

function δUδσ(σ; m2 = -1.0f0, λ= 1.0f0,J=0.0f0)
	return σ * ((λ*σ^2)/6 +m2) + J
	# return σ * σ0 - h
end
export δUδσ



function init_langevin_2d(xyd,f=(x,y)->randn())
	N = length(xyd)
	u = zeros(myT, N, N,M)
	for I in CartesianIndices((N, N,M))
		x = xyd[I[1]]
		y = xyd[I[2]]
		u[I] = f(x,y)
	end
	return u
end
export init_langevin_2d


function init_langevin_1d(xyd,f=(x)->randn())
	N = length(xyd)
	u = zeros(myT, N,M)
	for I in CartesianIndices((N,M))
		x = xyd[I[1]]
		u[I] = f(x)
	end
	return u
end
export init_langevin_1d
