using DrWatson

@time @quickactivate :LangevinDynamics
using BenchmarkTools
u0 = init_langevin_2d(LangevinDynamics.xyd_brusselator)
u0_GPU = CuArray(u0)
du = similar(u0_GPU)
using Plots
macro on_device(ex...)
    code = ex[end]
    kwargs = ex[1:end-1]

    @gensym kernel
    esc(quote
        let
            function $kernel()
                $code
                return
            end

            CUDA.@sync @cuda $(kwargs...) $kernel()
        end
    end)
end

@on_device shmem=10*10*10*sizeof(Float32) CuDynamicSharedArray(Float32, (10,10,10))

@time langevin_2d_loop_GPU(du, u0_GPU, p, 0.0f0)
langevin_2d_loop_shem_GPU(du, u0_GPU, p, 0.0f0)

langevin_2d_loop_flat_shem_GPU(du, u0_GPU, p, 0.0f0)

u0_GPU

kernel = @cuda launch=false update_2d_flat2_langevin!(du, u0_GPU, p...)
config = launch_configuration(kernel.fun)


p = LangevinDynamics.myT.((0.02, 0.1,0.0,step(LangevinDynamics.xyd_brusselator)))
t_it = @belapsed begin
    langevin_2d_loop_GPU($du, $u0_GPU, $p, 0.0f0)
    synchronize()
end






langevin_2d!(du, u0_GPU, p, 0.0f0)

t_it2 = @belapsed begin
    langevin_2d!($du, $u0_GPU, $p, 0.0f0)
    synchronize()
end

t_it2 = @belapsed begin
    langevin_2d_loop_flat_GPU($du, $u0_GPU, $p, 0.0f0)
    synchronize()
end

t_it3 = @belapsed begin
    langevin_2d_loop_flat2_GPU($du, $u0_GPU, $p, 0.0f0)
    synchronize()
end

t_it4 = @belapsed begin
    langevin_2d_loop_flat_shem_GPU($du, $u0_GPU, $p, 0.0f0)
    synchronize()
end
T_eff = (2) * 1 / 1e9 * nx *nx* nz*sizeof(Float32) / t_it
nx=32*2^3
nz=2^8

:(1+1)

function testf2(;u0=(x)->rand(5))
    f(x,y)=u0(y)
    f(1,1)-f(2,1)
end
testf2(;)
langevin_2d_ODE_prob()

device!(1)
u0 = randn(Float32,64,64,64,2^12,2)
sizeof(u0)/1024/1024/1024
u0_GPU = CuArray(u0)
du_GPU = similar(u0_GPU)
p = LangevinDynamics.myT.((1,-1,1,0))
@benchmark langevin_3d_tex_loop_GPU(du_GPU, u0_GPU,δUδσTextureTex, p, 0.1f0)

CartesianIndices((2, 2))[5]

@benchmark CUDA.@sync langevin_3d_tex_loop_GPU_2($du_GPU, $u0_GPU,δUδσTextureTex, p, 0.1f0)
using DelimitedFiles,Dierckx
Uσ_CPU_fun = Spline1D(
    readdlm("data/phi.dat")[:, 1] ./ (197.33),
    (readdlm("data/V.dat")[:, 1]) ./ (197.3) .^ 4,
)

# Uσ_CPU_fun = σ -> σ * (-(1 / 4) + σ^2)
σrng = 0.0:0.5:100

δUδσTextureMem = CuTextureArray(Float32.(Uσ_CPU_fun.(σrng)))

δUδσTextureMem = CuTextureArray(Float32.(derivative(Uσ_CPU_fun, σrng)))
# # copyto!(ImlambdaTextureMem, Imlambda)
# # copyto!(RelambdaTextureMem, Relambda)
δUδσTextureTex = CuTexture(δUδσTextureMem;)


@benchmark CUDA.@sync langevin_3d_tex_loop_GPU($du_GPU, $u0_GPU,δUδσTextureTex, p, 0.1f0)

@benchmark CUDA.@sync langevin_3d_tex_loop_GPU($du_GPU, $u0_GPU, $δUδσTextureTex, $p, 0.1f0)


@benchmark CUDA.@sync langevin_3d_loop_GPU(du_GPU, u0_GPU ,Uσ_CPU_fun, p, 0.1f0)






langevin_3d_tex_loop_GPU_2(du_GPU, u0_GPU,δUδσTextureTex, p, 0.1f0)


2^3

u0[]
GC.gc(true)
CUDA.reclaim()
CPU()

langevin_1d_loop_GPU(du_GPU, u0_GPU, p, 0.1f0)
@benchmark langevin_1d_tex_loop_GPU(du_GPU, u0_GPU,δUδσTextureTex, p, 0.1f0)

@benchmark langevin_1d_loop_GPU(du_GPU, u0_GPU, p, 0.1f0)

t_it = @benchmark begin
    langevin_1d_tex_loop_GPU($du_GPU, $u0_GPU,δUδσTextureTex, p, 0.1f0)
    synchronize()
end

t_it2 = @belapsed begin
    langevin_iso_loop_GPU($du, $u0_GPU, $p, 0.0f0)
    synchronize()
end
using CUDA,LinearAlgebra,BenchmarkTools
device!(0)
nx = ny = 2^5
nz=2^5
nk=2^7
A = CUDA.zeros(Float32, nx, ny,nz,nk);
B = CUDA.rand(Float32, nx, ny, nz,nk);
t_it1 =@belapsed CUDA.@sync copyto!($A, $B)
t_it2 = @belapsed CUDA.@sync axpy!(2f0,$A, $B)
T_tot1 = 2 * 1 / 1e9 * nx * ny*nz*nk * sizeof(Float32) / t_it1
@elapsed CUDA.@sync axpy!(2.0f0, A, B)
nx = ny = 2^12
A = CUDA.zeros(Float32, nx, ny);
B = CUDA.rand(Float32, nx, ny);
t_it3 = @belapsed CUDA.@sync copyto!($A, $B)
T_tot3 = 2 * 1 / 1e9 * nx * ny * sizeof(Float32) / t_it3






@time for i = 1:100000
    CUDA.@sync copyto!(A, B)
end
T_tot = 10000*2*1/1e9*nx*ny*sizeof(Float32)/t_it

@inbounds function memcopy_KP!(A,B,C)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    du1[ix,iy] = dutmp[ix,iy]+dt
    return nothing
end
max_threads  = attribute(device(),CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
thread_count = []
throughputs  = []
for pow = Int(log2(32)):Int(log2(max_threads))
    threads = (2^pow, 1)
    blocks  = (nx÷threads[1], ny)
    t_it = @belapsed begin @cuda blocks=$blocks threads=$threads memcopy_KP!($A, $B); synchronize() end
    T_tot = 2*1/1e9*nx*ny*sizeof(Float32)/t_it
    push!(thread_count, prod(threads))
    push!(throughputs, T_tot)
    println("(threads=$threads) T_tot = $(T_tot)")
end
thread_count = []
throughputs = []
for pow = 0:Int(log2(max_threads / 32))
    threads = (32, 2^pow)
    blocks = (nx ÷ threads[1], ny ÷ threads[2])
    t_it = @belapsed begin
        @cuda blocks = $blocks threads = $threads memcopy_KP!($A, $B)
        synchronize()
    end
    T_tot = 2 * 1 / 1e9 * nx * ny * sizeof(Float32) / t_it
    push!(thread_count, prod(threads))
    push!(throughputs, T_tot)
    println("(threads=$threads) T_tot = $(T_tot)")
end


T_tot_max, index = findmax(throughputs)
threads = (32, thread_count[index]÷32)
blocks  = (nx÷threads[1], ny÷threads[2])
t_it = @belapsed begin @cuda blocks=$blocks threads=$threads memcopy_triad_KP!($A, $B, $C, $s); synchronize() end
T_tot = 2*1/1e9*nx*ny*sizeof(Float64)/t_it
