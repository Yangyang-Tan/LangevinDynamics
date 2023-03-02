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


u0 = randn(128,128,2)
u0_GPU = CuArray(u0)
du = similar(u0_GPU)
p = LangevinDynamics.myT.((1,-1,1,0))

langevin_0d_tex_loop_GPU(du, u0_GPU,δUδσTextureTex, p, 0.1f0)



t_it = @belapsed begin
    langevin_iso_loop_GPU($du, $u0_GPU, $p, 0.0f0)
    synchronize()
end

t_it2 = @belapsed begin
    langevin_iso_loop_GPU($du, $u0_GPU, $p, 0.0f0)
    synchronize()
end
