using Distributed, CUDA
addprocs(length(devices()))
@everywhere using CUDA, DrWatson
@everywhere @time @quickactivate :LangevinDynamics

@everywhere u0 = randn(Float32,32,32,32,2^14,2)
@everywhere p = LangevinDynamics.myT.((1,-1,1,0))
@everywhere Uσ_CPU_fun = σ -> σ * (-(1 / 4) + σ^2)
@everywhere σrng = 0.0:0.05:100
sizeof(u0)/1024/1024/1024

tempd=1
"u0_GPU$tempd"

asyncmap((zip(workers(), devices()))) do (p, d)
    remotecall_wait(p) do
        @info "Worker $p uses $d"
        device!(d)
        testfun(u0)
    end
end


@everywhere function testfun(u0)
    δUδσTextureMem = CuTextureArray(Float32.(Uσ_CPU_fun.(σrng)))
    δUδσTextureTex = CuTexture(δUδσTextureMem; interpolation = CUDA.CubicInterpolation())
    u0_GPU = CuArray(u0)
    du_GPU = similar(u0_GPU)

    CUDA.@time for i in 1:5
    langevin_3d_tex_loop_GPU(du_GPU, u0_GPU,δUδσTextureTex, p, 0.1f0)
    end

    CUDA.@time for i in 1:1000
    langevin_3d_tex_loop_GPU(du_GPU, u0_GPU,δUδσTextureTex, p, 0.1f0)
    end

    @time for i in 1:500000
    langevin_3d_tex_loop_GPU(du_GPU, u0_GPU,δUδσTextureTex, p, 0.1f0)
    end
end


device!(2)
testfun(u0)



@sync begin
    @async begin
        device!(0)
        u0_GPU1 = CuArray(u0)
        du_GPU1 = similar(u0_GPU1)
        GC.gc(true)
        CUDA.reclaim()
    end
    @async begin
        device!(1)
        u0_GPU2 = CuArray(u0)
        du_GPU2 = similar(u0_GPU2)
        GC.gc(true)
        CUDA.reclaim()
    end
    @async begin
        device!(2)
        u0_GPU3 = CuArray(u0)
        du_GPU3 = similar(u0_GPU3)
        GC.gc(true)
        CUDA.reclaim()
    end
    @async begin
        device!(3)
        u0_GPU4 = CuArray(u0)
        du_GPU4 = similar(u0_GPU4)
        GC.gc(true)
        CUDA.reclaim()
    end
end

device!(1)
u0_GPU4 = CuArray(u0)
du_GPU4 = similar(u0_GPU1)

@sync begin
    @async begin
        device!(0)
        for i in 1:200000
        langevin_3d_loop_GPU(du_GPU1, u0_GPU1,Uσ_CPU_fun, p, 0.1f0)
        end
    end
    @async begin
        device!(1)
        for i in 1:200000
        langevin_3d_loop_GPU(du_GPU2, u0_GPU2,Uσ_CPU_fun, p, 0.1f0)
        end
    end
    @async begin
        device!(2)
        for i in 1:200000
        langevin_3d_loop_GPU(du_GPU3, u0_GPU3,Uσ_CPU_fun, p, 0.1f0)
        end
    end
    @async begin
        device!(3)
        for i in 1:200000
        langevin_3d_loop_GPU(du_GPU4, u0_GPU4,Uσ_CPU_fun, p, 0.1f0)
        end
    end
end
