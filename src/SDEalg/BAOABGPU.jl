# struct BAOABGPU{T} <: StochasticDiffEqAlgorithm
#     gamma::T
#     scale_noise::Bool
# end
# BAOABGPU(; gamma = 1.0, scale_noise = true) = BAOABGPU(gamma, scale_noise)

# @cache struct BAOABGPUCache{uType,uEltypeNoUnits,rateNoiseType,uTypeCombined} <:
#               StochasticDiffEqMutableCache
#     utmp::uType
#     dutmp::uType
#     k::uType
#     gtmp::uType
#     noise::rateNoiseType
#     half::uEltypeNoUnits
#     c1::uEltypeNoUnits
#     c2::uEltypeNoUnits
#     tmp::uTypeCombined
# end

# function alg_cache(
#     alg::BAOABGPU,
#     prob,
#     u,
#     ΔW,
#     ΔZ,
#     p,
#     rate_prototype,
#     noise_rate_prototype,
#     jump_rate_prototype,
#     ::Type{uEltypeNoUnits},
#     ::Type{uBottomEltypeNoUnits},
#     ::Type{tTypeNoUnits},
#     uprev,
#     f,
#     t,
#     dt,
#     ::Type{Val{true}},
# ) where {uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits}
#     dutmp = zero(u.x[1])
#     utmp = zero(u.x[2])
#     k = zero(rate_prototype.x[1])

#     gtmp = zero(rate_prototype.x[1])
#     noise = zero(rate_prototype.x[1])

#     half = uEltypeNoUnits(1 // 2)
#     c1 = exp(-alg.gamma * dt)
#     c2 = sqrt(1 - alg.scale_noise * c1^2) # if scale_noise == false, c2 = 1

#     tmp = zero(u)

#     BAOABGPUCache(
#         utmp,
#         dutmp,
#         k,
#         gtmp,
#         noise,
#         half,
#         uEltypeNoUnits(c1),
#         uEltypeNoUnits(c2),
#         tmp,
#     )
# end


# function verify_f2(f, res, p, q, pa, t, integrator, ::BAOABCache)
#     f(res, p, q, pa, t)
#     res != p && throwex(integrator)
# end
# function throwex(integrator)
#     algn = typeof(integrator.alg)
#     throw(ArgumentError("Algorithm $algn is not applicable if f2(p, q, t) != p"))
# end


# function initialize!(integrator, cache::BAOABGPUCache)
#     @unpack t, dt, uprev, u, p, W = integrator
#     du1 = integrator.uprev.x[1]
#     u1 = integrator.uprev.x[2]

#     integrator.f.f1(cache.k, du1, u1, p, t)
# end


# @muladd function perform_step!(integrator, cache::BAOABGPUCache)
#     @unpack t, dt, sqdt, uprev, u, p, W, f = integrator
#     @unpack utmp, dutmp, k, gtmp, noise, half, c1, c2 = cache
#     du1 = uprev.x[1]
#     u1 = uprev.x[2]

#     # B
#     @.. dutmp = du1 + half * dt * k

#     # A
#     @.. utmp = u1 + half * dt * dutmp

#     # O
#     integrator.g(gtmp, utmp, p, t + dt * half)
#     @.. noise = gtmp * W.dW / sqdt
#     @.. dutmp = c1 * dutmp + c2 * noise

#     # A
#     @.. u.x[2] = utmp + half * dt * dutmp

#     # B
#     f.f1(k, dutmp, u.x[2], p, t + dt)
#     @.. u.x[1] = dutmp + half * dt * k
# end
using BenchmarkTools
struct SimpleBAOABGPU{T} <: StochasticDiffEqAlgorithm
    eta::T
    noise::T
end
struct SimpleBAOABGPUIsing{T} <: StochasticDiffEqAlgorithm
    eta::T
    noise::T
end

struct SimpleSDEProblem{T<:AbstractFloat,N}
    f1::Function
    v0::AbstractArray{T,N}
    u0::AbstractArray{T,N}
    tspan::Tuple{T,T}
    p::Union{SciMLBase.NullParameters,NTuple}
end

SimpleSDEProblem(f1, v0, u0, tspan) =
    SimpleSDEProblem(f1, v0, u0, tspan, SciMLBase.NullParameters())
export SimpleSDEProblem


SimpleBAOABGPU(; eta = 1.0, noise = error("noise not provided")) =
    SimpleBAOABGPU(eta, noise)
export SimpleBAOABGPU


SimpleBAOABGPUIsing(; eta = 1.0, noise = error("noise not provided")) =
    SimpleBAOABGPUIsing(eta, noise)
export SimpleBAOABGPUIsing
# @muladd function DiffEqBase.solve(
#     prob::SimpleSDEProblem,
#     alg::SimpleBAOABGPU,
#     args...;
#     dt = error("dt required for SimpleEM"),
#     savefun::Function=x->nothing,
# )
#     GC.gc(true)
#     CUDA.reclaim()
#     f1 = prob.f1
#     # f2 = prob.f.f2
#     # g = prob.g
#     tspan = prob.tspan
#     p = prob.p

#     du1 = CuArray(prob.v0)
#     u1 = CuArray(prob.u0)
#     # dvtmp = similar(du1)
#     # gtmp = zero(u0)
#     dW = zero(u1)
#     c1 = exp(-alg.eta * dt)
#     c2 = sqrt(1 - c1^2)
#     c3 =  alg.noise
#     @inbounds begin
#         n = Int((tspan[2] - tspan[1]) / dt) + 1
#         t = [tspan[1] + i * dt for i = 0:(n-1)]
#         sqdt = sqrt(dt)
#     end
#     savefun(u1)
#     CUDA.@sync f1(dW, u1)
#     @inbounds for i = 2:n
#         # B
#         tprev = t[i-1]
#         println("1=",@belapsed CUDA.@sync axpy!($dt / 2, $dW, $du1))
#         # A
#         println("2=", @belapsed CUDA.@sync axpy!($dt / 2, $du1, $u1))
#         # O
#         # g(gtmp, u1, p, tprev + dt / 2)
#         # CUDA.randn!(dW)
#         println("rand=", @belapsed CUDA.@sync randn!($dW))
#         # @.. dW = c3 * dW
#         println("3=", @belapsed CUDA.@sync axpby!($c2 * $c3, $dW, $c1, $du1))
#         # A
#         println("4=", @belapsed CUDA.@sync axpy!($dt / 2, $du1, $u1))
#         # B
#         println("5=", @belapsed CUDA.@sync $f1($dW, $u1))
#         println("6=", @belapsed CUDA.@sync axpy!($dt / 2, $dW, $du1))
#         println("7=", @belapsed $savefun($u1))
#     end
#     # return (Array(du1), Array(u1))
#     # sol = DiffEqBase.build_solution(prob, alg, t, u, calculate_error = false)
# end


# @muladd function DiffEqBase.solve(
#     prob::SimpleSDEProblem,
#     alg::SimpleBAOABGPU,
#     args...;
#     dt = error("dt required for SimpleEM"),
#     savefun::Function=x->nothing,
#     fun=nothing,
# )
#     GC.gc(true)
#     CUDA.reclaim()
#     f1 = prob.f1
#     # f2 = prob.f.f2
#     # g = prob.g
#     tspan = prob.tspan
#     p = prob.p

#     du1 = CuArray(prob.v0)
#     u1 = CuArray(prob.u0)
#     # dvtmp = similar(du1)
#     # gtmp = zero(u0)
#     dW = zero(u1)
#     c1 = exp(-alg.eta * dt)
#     c2 = sqrt(1 - c1^2)
#     c3 =  alg.noise
#     @inbounds begin
#         n = Int((tspan[2] - tspan[1]) / dt) + 1
#         t = [tspan[1] + i * dt for i = 0:(n-1)]
#         sqdt = sqrt(dt)
#     end
#     savefun(u1)
#     CUDA.@sync f1(dW, u1)
#     @time @inbounds for i = 2:n
#         # B
#         tprev = t[i-1]
#         CUDA.@sync axpy!(dt / 2, dW, du1)
#         # A
#         CUDA.@sync axpy!(dt / 2, du1, u1)
#         # O
#         # g(gtmp, u1, p, tprev + dt / 2)
#         # CUDA.randn!(dW)
#         CUDA.@sync randn!(dW)
#         # @.. dW = c3 * dW
#         CUDA.@sync axpby!(c2 * c3, dW, c1, du1)
#         # A
#         CUDA.@sync axpy!(dt / 2, du1, u1)
#         # B
#         CUDA.@sync f1(dW, u1)
#         CUDA.@sync axpy!(dt / 2, dW, du1)
#         savefun(u1)
#     end
#     return (Array(du1), Array(u1))
#     # sol = DiffEqBase.build_solution(prob, alg, t, u, calculate_error = false)
# end


@muladd function DiffEqBase.solve(
    prob::SimpleSDEProblem,
    alg::SimpleBAOABGPU,
    args...;
    dt = error("dt required for SimpleEM"),
    savefun::Function=x->nothing,
    fun = error("fun required for SimpleEM"),
)
    GC.gc(true)
    CUDA.reclaim()
    f1 = prob.f1
    # f2 = prob.f.f2
    # g = prob.g
    tspan = prob.tspan
    p = prob.p
    du1 = copy(prob.v0)
    u1 = copy(prob.u0)
    dW = zero(u1)
    dutmp = copy(prob.u0)


    CUDA.@sync randn!(dW)
    # CUDA.@sync randn!(dW_1)
    # CUDA.@sync randn!(dW_2)

    c1 = exp(-alg.eta * dt)
    c2 = sqrt(1 - c1^2)
    c3 =  alg.noise
    @inbounds begin
        n = Int((tspan[2] - tspan[1]) / dt) + 1
        t = [tspan[1] + i * dt for i = 0:(n-1)]
        sqdt = sqrt(dt)
    end
    m_1=zero(t)
    m_2=zero(t)
    # m_1[1],m_2[1] = savefun(u1)
    m_1[1]=mean(u1)
    CUDA.@sync f1(dutmp, u1)
    # CUDA.@sync f1(dutmp_1, u1_1)
    # CUDA.@sync f1(dutmp_2, u1_2)
    # B
    CUDA.@sync axpy!(dt / 2, dutmp, du1)
    # CUDA.@sync axpy!(dt / 2, dutmp_1, du1_1)
    # CUDA.@sync axpy!(dt / 2, dutmp_2, du1_2)
    # A
    CUDA.@sync axpy!(dt / 2, du1, u1)
    # CUDA.@sync axpy!(dt / 2, du1_1, u1_1)
    # CUDA.@sync axpy!(dt / 2, du1_2, u1_2)
    # O
    # g(gtmp, u1, p, tprev + dt / 2)
    # CUDA.randn!(dW)
    # @.. dW = c3 * dW
    CUDA.@sync axpby!(c2 * c3, dW, c1, du1)
    # CUDA.@sync axpby!(c2 * c3, dW_1, c1, du1_1)
    # CUDA.@sync axpby!(c2 * c3, dW_2, c1, du1_2)
    # A
    CUDA.@sync axpy!(dt / 2, du1, u1)
    # m_1[2], m_2[2] = savefun(u1)
    m_1[2]=mean(u1)
    # CUDA.@sync axpy!(dt / 2, du1_1, u1_1)
    # CUDA.@sync axpy!(dt / 2, du1_2, u1_2)
    # CUDA.@sync randn!(dW)
    # CUDA.@sync randn!(dW_1)
    # CUDA.@sync randn!(dW_2)
    I1 = CartesianIndices(u1)
    N=size(u1,1)
    # M=prod(size(u1))÷4
    M=2^7
    N_4=size(u1,4)
    N_scale=N_4÷M
    M1 = prod(size(u1)) ÷ N_scale
    # WT = CUDA.randn(size(u1,1),size(u1,2),size(u1,3),size(u1,4)*2^5 ÷4)
    CUDA.@sync dW1 = CUDA.randn(size(u1, 1), size(u1, 2), size(u1, 3), M)
    for s = 0:N_scale-1
        dutmp1 = @view dutmp[:, :, :, 1+s*M:M+s*M]
        du11 = @view du1[:, :, :, 1+s*M:M+s*M]
        u11 = @view u1[:, :, :, 1+s*M:M+s*M]
        for i = 3:n
            # CUDA.@sync axpy2!(c1, c2 * c3, dt / 2, dW, dutmp, du1, u1)
            # m_1[i], m_2[i] = (savefun(u1) ./ N_scale) .+ (m_1[i], m_2[i])
            f1(dutmp1, u11)
            axpy2!(c1, c2 * c3, dt / 2, dW1, dutmp1, du11, u11, M1, n)
            CUDA.randn!(dW1)
            m_1[i] += mean(u11) / N_scale
            # synchronize()
            # f1(dutmp_1, u1_1)
            # f1(dutmp_2, u1_2)

            # for s in 0:3
            #    CUDA.@sync axpy2!(c1, c2 * c3, dt / 2, dW, dutmp, du1, u1, M, s)
            # end
            # CUDA.@sync axpy2!(c1, c2 * c3, dt / 2, dW, dutmp, du1, u1, M, 0)
            # CUDA.@sync axpy2!(c1, c2 * c3, dt / 2, dW, dutmp, du1, u1, M, 1)
            # CUDA.@sync axpy2!(c1, c2 * c3, dt / 2, dW, dutmp, du1, u1, M, 2)

            # axpy2!(c1, c2 * c3, dt / 2, dW, dutmp, du1, u1, M, 1)
            # axpy2!(c1, c2 * c3, dt / 2, dW_1, dutmp_1, du1_1, u1_1)
            # axpy2!(c1, c2 * c3, dt / 2, dW_2, dutmp_2, du1_2, u1_2)
            # copyto!(dutmp,u1)
            # axpy2!(c1, c2 * c3, dt / 2, dW, dutmp, du1, u1, I1, N, M, fun)
            # CUDA.randn!(dW1)

            # synchronize()
            # f1(dutmp_1, u1_1)
            # axpy2!(c1, c2 * c3, dt / 2, dW_1, dutmp_1, du1_1, u1_1)
            # randn!(dW_1)
            # synchronize()
            # f1(dutmp_2, u1_2)
            # axpy2!(c1, c2 * c3, dt / 2, dW_2, dutmp_2, du1_2, u1_2)
            # randn!(dW_2)
            # randn!(dW_1)
            # randn!(dW_2)
            # copyto!(dutmp,u1)
            # synchronize()
            # CUDA.@sync axpy!(dt / 2, dutmp, du1)
            # synchronize()
            #B
            # tprev = t[i-1]
            # CUDA.@sync axpy!(dt / 2, dutmp, du1)
            # # # A
            # CUDA.@sync axpy!(dt / 2, du1, u1)
            # #O
            # CUDA.@sync axpby!(c2 * c3, dW, c1, du1)
            # # A
            # CUDA.@sync axpy!(dt / 2, du1, u1)
            # B
            # randn!(dW)
            # f1(dutmp, u1)
            # synchronize()
            # CUDA.@sync axpy!(dt / 2, dutmp, du1)
            # savefun(u1)
        end
    end

    CUDA.@sync f1(dutmp, u1)
    # f1(dutmp_1, u1_1)
    # f1(dutmp_2, u1_2)
    CUDA.@sync axpy!(dt / 2, dutmp, du1)
    # CUDA.@sync axpy!(dt / 2, dutmp_1, du1_1)
    # CUDA.@sync axpy!(dt / 2, dutmp_2, du1_2)
    return [m_1, m_2]
    # sol = DiffEqBase.build_solution(prob, alg, t, u, calculate_error = false)
end


@muladd function DiffEqBase.solve(
    prob::SimpleSDEProblem,
    alg::SimpleBAOABGPUIsing,
    args...;
    dt = error("dt required for SimpleEM"),
    savefun::Function = x -> nothing,
    fun = error("fun required for SimpleEM"),
)
    GC.gc(true)
    CUDA.reclaim()
    f1 = prob.f1
    tspan = prob.tspan
    p = prob.p
    du1 = copy(prob.v0)
    u1 = copy(prob.u0)
    dW = zero(u1)
    dutmp = copy(prob.u0)


    CUDA.@sync randn!(dW)
    c1 = exp(-alg.eta * dt)
    c2 = sqrt(1 - c1^2)
    c3 = alg.noise
    @inbounds begin
        n = unsafe_trunc(Int64,(tspan[2] - tspan[1]) / dt) + 1
        t = [tspan[1] + i * dt for i = 0:(n-1)]
        sqdt = sqrt(dt)
    end
    m_1 = zero(t)
    m_2 = zero(t)
    m_1[1] = mean(u1)
    CUDA.@sync f1(dutmp, u1)
    # B
    CUDA.@sync axpy!(dt / 2, dutmp, du1)
    # A
    CUDA.@sync axpy!(dt / 2, du1, u1)
    CUDA.@sync axpby!(c2 * c3, dW, c1, du1)
    CUDA.@sync axpy!(dt / 2, du1, u1)
    m_1[2] = mean(u1)
    I1 = CartesianIndices(u1)
    N = size(u1, 1)
    M = 2^7
    N_4 = size(u1, 4)
    N_scale = N_4 ÷ M
    M1 = prod(size(u1)) ÷ N_scale
    CUDA.@sync dW1 = CUDA.randn(size(u1, 1), size(u1, 2), size(u1, 3), M)
    for s = 0:N_scale-1
        dutmp1 = @view dutmp[:, :, :, 1+s*M:M+s*M]
        du11 = @view du1[:, :, :, 1+s*M:M+s*M]
        u11 = @view u1[:, :, :, 1+s*M:M+s*M]
        for i = 3:9*n÷10
            f1(dutmp1, u11)
            axpy2!(c1, c2 * c3, dt / 2, dW1, dutmp1, du11, u11, M1, n)
            CUDA.randn!(dW1)
            m_1[i], m_2[i] = (m_1[i], m_2[i]) .+ (savefun(u11) ./ N_scale)
        end
        for i2 in (9 * n ÷ 10)+1:n
            f1(dutmp1, u11)
            axpy2!(c1, c2 * c3, dt / 2, dW1, dutmp1, du11, u11, M1, n)
            CUDA.randn!(dW1)
            m_1[i2], m_2[i2] =(m_1[i2], m_2[i2]).+(savefun(u11) ./ N_scale)
        end
    end

    CUDA.@sync f1(dutmp, u1)
    CUDA.@sync axpy!(dt / 2, dutmp, du1)
    return [m_1, m_2]
end


function axpy2_kernel(c1, c2, dt, dW, A, du1, u1, I, N,M,fun)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    for i = ix:blockDim().x*gridDim().x:M
        for j in (0, M,2M,3M)
        x, y, z, k = Tuple(I[i+j])
        xp1, xm1 = limitbound(x + 1, N), limitbound(x - 1, N)
        yp1, ym1 = limitbound(y + 1, N), limitbound(y - 1, N)
        zp1, zm1 = limitbound(z + 1, N), limitbound(z - 1, N)
        ∇²u =
            -6 * A[x, y, z, k] +
            A[xp1, y, z, k] +
            A[xm1, y, z, k] +
            A[x, yp1, z, k] +
            A[x, ym1, z, k] +
            A[x, y, zp1, k] +
            A[x, y, zm1, k] - fun(A[x, y, z, k])
        @inbounds TMP1 = 2 * dt * ∇²u + du1[i+j]
        @inbounds TMP2 = dt * TMP1 + u1[i+j]
        @inbounds TMP3 = c1 * TMP1 + c2 * dW[i+j]
        @inbounds du1[i+j] = TMP3
        @inbounds u1[i+j] = TMP2 + TMP3 * dt
        end
    end
    # for i = prod(size(A))÷2+ix:blockDim().x*gridDim().x:prod(size(A))
    #     x, y, z, k = Tuple(I[i])
    #     xp1, xm1 = limitbound(x + 1, N), limitbound(x - 1, N)
    #     yp1, ym1 = limitbound(y + 1, N), limitbound(y - 1, N)
    #     zp1, zm1 = limitbound(z + 1, N), limitbound(z - 1, N)
    #     ∇²u =
    #         -6 * u1[x, y, z, k] +
    #         u1[xp1, y, z, k] +
    #         u1[xm1, y, z, k] +
    #         u1[x, yp1, z, k] +
    #         u1[x, ym1, z, k] +
    #         u1[x, y, zp1, k] +
    #         u1[x, y, zm1, k] - fun(u1[x, y, z, k])
    #     @inbounds TMP1 = 2 * dt * ∇²u + du1[i]
    #     @inbounds TMP2 = dt * TMP1 + u1[i]
    #     @inbounds TMP3 = c1 * TMP1 + c2 * dW[i]
    #     @inbounds du1[i] = TMP3
    #     @inbounds u1[i] = TMP2 + TMP3 * dt
    # end
    return nothing
end

function axpy2!(c1, c2, dt, dW, A, du1, u1, I, N,M,fun)
    @cuda blocks = 2^8 threads = 1024 axpy2_kernel(c1, c2, dt, dW, A, du1, u1, I, N,M,fun)
end


function axpy2_kernel(c1, c2, dt, dW, A, du1, u1, M, out)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    # iy = (blockIdx().y - 1) * blockDim().y + (threadIdx().y-1)
    for i = ix:blockDim().x*gridDim().x:M
            # i = j+iy*M
            @inbounds TMP1 = 2 * dt * A[i] + du1[i]
            @inbounds TMP2 = dt * TMP1 + u1[i]
            @inbounds TMP3 = c1 * TMP1 + c2 * dW[i]
            @inbounds TMP4 = TMP2 + TMP3 * dt
            # @inbounds acc += TMP4
            @inbounds du1[i] = TMP3
            @inbounds u1[i] = TMP4
    end
    # CUDA.@atomic out[] += acc
    return nothing
end


function axpy2!(c1, c2, dt, dW, A, du1, u1,M,n)
    @cuda blocks = 2^8 threads = 1024 axpy2_kernel(c1, c2, dt, dW, A, du1, u1, M,n)
end
