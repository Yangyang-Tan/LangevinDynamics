using CUDA
function DGP(N)
    x = range(0, 1, N^2)
    return reshape(x, (N, N))
end
1
range(0, 1, 4^2)


function main(N)
    x = CuArray(DGP(N))
    V0 = CUDA.ones(Float32, N)
    idx = ()
    a = 0.5f0
    max_iter = 100
    iter = 0
    tmp = x .+ a * V0'
    while iter < max_iter
        V1 = V0
        tmp .= x .+ a * V1'
        V0, idx = findmax(tmp, dims = 2)
        iter += 1
    end
    return V0, idx, iter
end

@time CUDA.@sync main(2^15);
