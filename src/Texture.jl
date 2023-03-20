using DelimitedFiles,Dierckx

Uσ_CPU_fun = Spline1D(
    readdlm("data/phi.dat")[:, 1] ./ (197.33),
    (readdlm("data/V.dat")[:, 1]) ./ (197.3) .^ 4,
)

Uσ_CPU_fun = σ -> σ * (-(1 / 4) + σ^2)
σrng = 0.0:0.05:100

δUδσTextureMem = CuTextureArray(Float32.(Uσ_CPU_fun.(σrng)))


δUδσTextureMem = CuTextureArray(Float32.(derivative(Uσ_CPU_fun, σrng)))
# copyto!(ImlambdaTextureMem, Imlambda)
# copyto!(RelambdaTextureMem, Relambda)
δUδσTextureTex = CuTexture(δUδσTextureMem; interpolation = CUDA.CubicInterpolation())
plot(derivative(Uσ_CPU_fun, σrng)[1:10])
sign.(CUDA.randn(10, 10))

# struct Langevin3D
#     γ::Float32
#     m2::Float32
#     λ::Float32
#     J::Float32
#     tspan::Tuple{Float32, Float32}
#     T::Float32
#     tex::CuTexture{Float32, 1}
#     u0fun::Function
# end

struct TaylorParameters{T}
    λ1::T
    λ2::T
    λ3::T
    λ4::T
    λ5::T
    ρ0::T
    c::T
end

TaylorParameters(λ::Array) = TaylorParameters(λ[1], λ[2], λ[3], λ[4], λ[5], λ[6], λ[7])



function funout(λ::TaylorParameters)
    σ ->
        σ * λ.λ1 +
        σ * (-λ.ρ0 + σ^2 / 2) * λ.λ2 +
        (σ * (-λ.ρ0 + σ^2 / 2)^2 * λ.λ3) / 2 +
        (σ * (-λ.ρ0 + σ^2 / 2)^3 * λ.λ4) / 6 +
        (σ * (-λ.ρ0 + σ^2 / 2)^4 * λ.λ5) / 24 - c
end

funout(TaylorParameters([1, 1, 2, 3, 4, 5]))(1)
readdlm("data/eQCD_Input/buffer/lam1.dat")
