@inbounds function memcopy_KP!(A, B, C)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    du1[ix, iy] = dutmp[ix, iy] + dt
    return nothing
end


nx = ny = 2^10
nz=2^2
nk=2^7
CUDA.seed!(2)
A = CUDA.randn(Float32, nx, ny, nz,1);
B = CUDA.randn(Float32, nx, ny,nz);
C = CUDA.randn(Float32, nx, ny, nz);
dW1 = CUDA.randn(Float32, nx, ny, nz);

A

copy(A)

0.2f0 * sum(A) + sum(B)
0.2f0*(0.2f0 * sum(A) + sum(B))+sum(C)
axpy2!(0.2f0, A, B, C)
sum(C)
sum(B)
c2 * c3, dW, c1, du1
∇²u = -2N * u[i]
for uvec in unitvecs
    ∇²u += u[i+uvec] + u[i-uvec]
end
I
ntuple(i -> CartesianIndex(ntuple(==(i), Val(4))), Val(3))
ntuple(i -> CartesianIndex(ntuple(==(i), Val(3))), Val(3))

It = CartesianIndices(A)
Ifirst, Ilast = first(It), last(It)
I1 = oneunit(Ifirst)

prod(It[4].I)

LinearIndices(A)[1,12,1,1]


function axpy2_kernel(c1, c2, dt, dW, A, du1, u1,I,N)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    for i = ix:blockDim().x*gridDim().x:prod(size(A))
        x, y, z, k = Tuple(I[i])
        xp1, xm1 = limitbound(x + 1, N), limitbound(x - 1, N)
        yp1, ym1 = limitbound(y + 1, N), limitbound(y - 1, N)
        zp1, zm1 = limitbound(z + 1, N), limitbound(z - 1, N)
        ∇²u =-6 * A[x, y, z, k] +
            A[xp1, y, z, k] +
            A[xm1, y, z, k] +
            A[x, yp1, z, k] +
            A[x, ym1, z, k] +
            A[x, y, zp1, k] +
            A[x, y, zm1, k] - fun(A[x, y, z, k])
        @inbounds TMP1 = 2 * dt * ∇²u + du1[i]
        @inbounds TMP2 = dt * TMP1 + u1[i]
        @inbounds TMP3 = c1 * TMP1 + c2 * dW[i]
        @inbounds du1[i]= TMP3
        @inbounds u1[i] = TMP2 + TMP3 * dt
    end
    return nothing
end

function axpy2!(c1, c2, dt, dW, A, du1, u1, I, N)
    @cuda blocks = 2^8 threads = 1024 axpy2_kernel(c1, c2, dt, dW, A, du1, u1, I, N)
end
tv1,tv2,tv3= Tuple(CartesianIndices(A)[1])

@inbounds function memcopy_KP!(B,A)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    iz = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    B[ix, iy,iz] = A[ix, iy,iz]
    return nothing
end

threads = (256,1 , 1)
blocks  = (1, ny÷threads[2], nz÷threads[3])
blocks = (nx÷threads[1], ny ÷ threads[2], nz ÷ threads[3])
t_it = @belapsed CUDA.@sync @cuda blocks = 2^8 threads = 1024 memcopy2!(2f0,$C,$B, $A)
t_it2 = @belapsed CUDA.@sync @cuda blocks = $blocks threads = $threads memcopy_KP!($B, $A)
t_it = @belapsed CUDA.@sync axpy2!(2f0,2f0,2f0,$dW1,$C,$B, $A)
t_it = @belapsed CUDA.@sync axpy!(2.0f0, $B, $A)
20000 * t_it
T_tot = 4 * 1 / 1e9 * nx * ny * nz * sizeof(Float32) / t_it
@belapsed CUDA.@sync @cuda blocks = 1 threads = 1024 memcopy!(B, A)
CUDA.@sync @cuda blocks = 2 threads = 1024 memcopy2!(2f0,C,B, A)
sum(B)
CUDA.copyto!
axpy!
