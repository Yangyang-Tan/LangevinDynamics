using FFTW, CUDA
using CUDA.CUFFT
using CUDA: CUDA.randn

a = CUDA.randn(ComplexF32, 512, 512, 512)
b = CUDA.randn(Float32, 512, 512, 512)
CUDA.CUFFT.ifft(CUDA.CUFFT.fft(a))


@benchmark CUDA.@sync fft!($a)

function periodic_convolution(Z, u)
    du = zeros(size(Z)) # Initialize the output array

    N = size(u, 1) # Assuming u is a square filter of size NxN
    M, N = size(Z)

    for i = 1:M
        for j = 1:N
            for ip = i-N÷2:i+N÷2
                for jp = j-N÷2:j+N÷2
                    # Apply periodic boundary conditions
                    ip_wrap = mod(ip - 1, M) + 1
                    jp_wrap = mod(jp - 1, N) + 1

                    du[i, j] += Z[ip_wrap, jp_wrap] * u[ip-i+N÷2+1, jp-j+N÷2+1]
                end
            end
        end
    end

    return du
end

function mycut(i, N)
    if N ≥ i ≥ 1
        i
    elseif i < 1
        i + N
    elseif i > N
        i - N
    end
end
Ztdata = [1 / (x^2 + y^2 + 1.0f0^2) for x = 0:3, y = 0:3]


function periodic_convolution2(Z, u)
    du = zeros(size(Z)) # Initialize the output array

    N = size(u, 1) # Assuming u is a square filter of size NxN
    M, N = size(Z)

    for i = 1:M
        for j = 1:N
            for ip = i-N÷2+1:i+N÷2
                for jp = j-N÷2+1:j+N÷2
                    # Apply periodic boundary conditions
                    Z = 1 / (abs(i - ip)^2 + abs(j - jp)^2 + 1.0f0^2)
                    du[i, j] += Z * u[mycut(ip, N), mycut(jp, N)]
                end
            end
        end
    end

    return du
end


function myL2(i, j, N)
    if abs(i - j) ≤ N ÷ 2
        return (i - j)^2
    else
        return (abs(i - j) - N)^2
    end
end

myL2(0, 6, 5)

function periodic_convolution3(Z, u)
    du = zeros(size(Z)) # Initialize the output array

    N = size(u, 1) # Assuming u is a square filter of size NxN
    M, N = size(Z)

    for i = 1:M
        for j = 1:N
            for ip = 1:N
                for jp = 1:N
                    # Apply periodic boundary conditions
                    Z = 1 / (myL2(i, ip, N) + myL2(j, jp, N) + 1.0f0^2)
                    du[i, j] += Z * u[mycut(ip, N), mycut(jp, N)]
                end
            end
        end
    end

    return du
end

a = rand(6)
a_2D = rand(6, 6)

b = randn(4, 4)
periodic_convolution2(a, b)
periodic_convolution3_1D(a, a)
periodic_convolution3(a_3D, a_3D)

function periodic_convolution3_1D(Z, u)
    du = zeros(size(Z)) # Initialize the output array

    N = size(u, 1) # Assuming u is a square filter of size NxN

    for i = 1:N
        for ip = 1:N
            # Apply periodic boundary conditions
            Z = 1 / (myL2(i, ip, N) + 1.0f0^2)
            du[i] += Z * u[mycut(ip, N)]
        end
    end

    return du
end

Ztdata = [
    [1 / (x^2 + 1.0f0^2) for x = 0:2]
    [1 / (x^2 + 1.0f0^2) for x = 3:-1:1]
]

indexlist = [0:2^8-1; 2^8:-1:1]

Ztdata_2D =
    fft(cu([1 / (x^2 + y^2 + 1.0f0^2) for x in indexlist, y in indexlist, z = 1:512]), 1:2)
a_2D_catch = CuArray{ComplexF32}(undef, 2^9, 2^9, 512)
γσ_catch = CUDA.randn(2^9, 2^9, 512)
dσ_catch = CUDA.randn(2^9, 2^9, 512, 2)
σ_catch = CUDA.randn(2^9, 2^9, 512, 2)

32*32*32*32*32*32


sin(1, 2, 3, 4, 5)
@benchmark CUDA.@sync langevin_2d_loop_nonlocal_GPU(
    $dσ_catch,
    $σ_catch,
    $γσ_catch,
    [-1.0f0, 1.0f0, 0.0f0],
    $Ztdata_2D,
    $a_2D_catch,
)


a_2D
1

@btime fastconv($a_2D, $Ztdata_2D, dims = 1:2)

a = rand(10, 5)
stack([fft(a[:, i]) for i = 1:5]) ≈ fft(a, 1)
fft(a)
fft(a)

ifft(fft([(x)^2 for x = 1:9]) .* fft([sin(x) for x = 1:9]))
ndims(randn(10, 5))
