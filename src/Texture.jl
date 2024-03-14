using DelimitedFiles,Dierckx

Uσ_CPU_fun = Spline1D(
    readdlm("data/phi.dat")[:, 1] ./ (197.33),
    (readdlm("data/V.dat")[:, 1]) ./ (197.3) .^ 4,
)

# Uσ_CPU_fun = σ -> σ * (-(1 / 4) + σ^2)
σrng = 0.0:0.05:100

# δUδσTextureMem = CuTextureArray(Float32.(Uσ_CPU_fun.(σrng)))


# δUδσTextureMem = CuTextureArray(Float32.(derivative(Uσ_CPU_fun, σrng)))
# copyto!(ImlambdaTextureMem, Imlambda)
# copyto!(RelambdaTextureMem, Relambda)
# δUδσTextureTex = CuTexture(δUδσTextureMem; interpolation = CUDA.CubicInterpolation())


function TexSpline1D(x_grid::StepRangeLen, δUδσ_grid)
    x_ini=x_grid[1]
    x_step=x_grid[2]-x_grid[1]
    δUδσTextureMem = CuTextureArray(δUδσ_grid)
    δUδσTextureTex = CuTexture(δUδσTextureMem; interpolation = CUDA.CubicInterpolation())
    function Texfun(σ)
        δUδσTextureTex[(σ-x_ini)/x_step +1]
    end
end
export TexSpline1D
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

struct QCDModelParameters{T1}
    U::TaylorParameters{T1}
    T::T1
    γ::T1
end

QCDModelParameters(U::Vector, T::Vector, γ::Vector) =
    QCDModelParameters.(U, T, γ)

QCDModelParameters(U::Vector, T::Vector) = QCDModelParameters.(U, T)
QCDModelParameters(U::TaylorParameters{T1}, T) where {T1} = QCDModelParameters(U, T, T1(0))
QCDModelParameters(U::TaylorParameters{T1}, T,γ) where {T1} = QCDModelParameters(U, T1(T), T1(γ))

export QCDModelParameters

TaylorParameters(λ::Matrix) =
    mapslices(
    x ->
        TaylorParameters(x),λ,
        dims = 2,
    )
TaylorParameters(λ::Vector) = TaylorParameters(λ...)

TaylorParameters(T::DataType,λ::Array)=TaylorParameters(T.(λ))
TaylorParameters(x1::T, x2::T) where {T} = TaylorParameters(x1,x2,T.([0,0,0,0,0])...)
TaylorParameters(x1::T, x2::T, x3::T) where {T} =
    TaylorParameters(x1, x2, T(0), T(0), T(0), x3, T(0))

TaylorParameters(x1::T, x2::T, x3::T,x4::T) where {T} =
    TaylorParameters(x1, x2, T(0), T(0), T(0), x3, x4)

TaylorParameters(x1::T, x2::T, x5::T,x3::T, x4::T) where {T} =
    TaylorParameters(x1, x2, x5, T(0), T(0), x3, x4)

TaylorParameters(x1::T, x2::T, x5::T, x6::T, x3::T, x4::T) where {T} =
    TaylorParameters(x1, x2, x5, x6, T(0), x3, x4)

export TaylorParameters




function funout(λ::TaylorParameters)
    σ ->
        σ * λ.λ1 +
        σ * (-λ.ρ0 + σ^2 / 2) * λ.λ2 +
        (σ * (-λ.ρ0 + σ^2 / 2)^2 * λ.λ3) / 2 +
        (σ * (-λ.ρ0 + σ^2 / 2)^3 * λ.λ4) / 6 +
        (σ * (-λ.ρ0 + σ^2 / 2)^4 * λ.λ5) / 24 - λ.c
end

function funout_cut(λ::TaylorParameters;α=100f0)
    σ ->
        -λ.c +
        σ * λ.λ1 +
        (σ * λ.λ2 * (σ^2 - 2 * λ.ρ0)) / 2 +
        (
            α *
            σ *
            (-78 + λ.λ3) *
            (σ^2 - 2 * λ.ρ0)^3 *
            (
                logisticsigmoid((α * (σ^2 - 2 * λ.ρ0)) / 2) -
                logisticsigmoid(-1f0 / 2 * (α * (σ^2 + 2 * λ.ρ0)))
            ) *
            (
                -1 +
                logisticsigmoid((α * (σ^2 - 2 * λ.ρ0)) / 2) +
                logisticsigmoid(-1.0f0 / 2 * (α * (σ^2 + 2 * λ.ρ0)))
            )
        ) / 48 -
        (
            σ *
            (σ^2 - 2 * λ.ρ0)^2 *
            (
                -λ.λ3 +
                (-78 + λ.λ3) * logisticsigmoid((α * (σ^2 - 2 * λ.ρ0)) / 2) +
                (-78 + λ.λ3) * logisticsigmoid(-1.0f0 / 2 * (α * (σ^2 + 2 * λ.ρ0)))
            )
        ) / 8
end


function funout_cut2(λ::TaylorParameters; α = 100.0f0)
    σ ->
        (
            -3840 * λ.c + 3840 * σ * λ.λ1 + 1920 * σ * λ.λ2 * (σ^2 - 2 * λ.ρ0) -
            80 *
            σ *
            λ.λ4 *
            (σ^2 - 2 * λ.ρ0)^3 *
            (
                -1 +
                logisticsigmoid((α * (σ^2 - 2 * λ.ρ0)) / 2) +
                logisticsigmoid(-1f0 / 2 * (α * (σ^2 + 2 * λ.ρ0)))
            ) -
            10 *
            σ *
            λ.λ5 *
            (σ^2 - 2 * λ.ρ0)^4 *
            (
                -1 +
                logisticsigmoid((α * (σ^2 - 2 * λ.ρ0)) / 2) +
                logisticsigmoid(-1f0 / 2 * (α * (σ^2 + 2 * λ.ρ0)))
            ) -
            480 *
            σ *
            (σ^2 - 2 * λ.ρ0)^2 *
            (
                -λ.λ3 +
                (-78 + λ.λ3) * logisticsigmoid((α * (σ^2 - 2 * λ.ρ0)) / 2) +
                (-78 + λ.λ3) * logisticsigmoid(-1f0 / 2 * (α * (σ^2 + 2 * λ.ρ0)))
            ) +
            80 *
            α *
            σ *
            (-78 + λ.λ3) *
            (σ^2 - 2 * λ.ρ0)^3 *
            (
                -logisticsigmoid((α * (σ^2 - 2 * λ.ρ0)) / 2) +
                logisticsigmoid((α * (σ^2 - 2 * λ.ρ0)) / 2)^2 +
                logisticsigmoid(-1f0 / 2 * (α * (σ^2 + 2 * λ.ρ0))) -
                logisticsigmoid(-1f0 / 2 * (α * (σ^2 + 2 * λ.ρ0)))^2
            ) +
            10 *
            α *
            σ *
            λ.λ4 *
            (σ^2 - 2 * λ.ρ0)^4 *
            (
                -logisticsigmoid((α * (σ^2 - 2 * λ.ρ0)) / 2) +
                logisticsigmoid((α * (σ^2 - 2 * λ.ρ0)) / 2)^2 +
                logisticsigmoid(-1f0 / 2 * (α * (σ^2 + 2 * λ.ρ0))) -
                logisticsigmoid(-1f0 / 2 * (α * (σ^2 + 2 * λ.ρ0)))^2
            ) +
            α *
            σ *
            λ.λ5 *
            (σ^2 - 2 * λ.ρ0)^5 *
            (
                -logisticsigmoid((α * (σ^2 - 2 * λ.ρ0)) / 2) +
                logisticsigmoid((α * (σ^2 - 2 * λ.ρ0)) / 2)^2 +
                logisticsigmoid(-1f0 / 2 * (α * (σ^2 + 2 * λ.ρ0))) -
                logisticsigmoid(-1f0 / 2 * (α * (σ^2 + 2 * λ.ρ0)))^2
            )
        ) / 3840
end



function funout_cut2_64(λ::TaylorParameters; α = 100.0)
    σ ->
        (
            -3840 * λ.c + 3840 * σ * λ.λ1 + 1920 * σ * λ.λ2 * (σ^2 - 2 * λ.ρ0) -
            80 *
            σ *
            λ.λ4 *
            (σ^2 - 2 * λ.ρ0)^3 *
            (
                -1 +
                logisticsigmoid((α * (σ^2 - 2 * λ.ρ0)) / 2) +
                logisticsigmoid(-1/ 2 * (α * (σ^2 + 2 * λ.ρ0)))
            ) -
            10 *
            σ *
            λ.λ5 *
            (σ^2 - 2 * λ.ρ0)^4 *
            (
                -1 +
                logisticsigmoid((α * (σ^2 - 2 * λ.ρ0)) / 2) +
                logisticsigmoid(-1 / 2 * (α * (σ^2 + 2 * λ.ρ0)))
            ) -
            480 *
            σ *
            (σ^2 - 2 * λ.ρ0)^2 *
            (
                -λ.λ3 +
                (-78 + λ.λ3) * logisticsigmoid((α * (σ^2 - 2 * λ.ρ0)) / 2) +
                (-78 + λ.λ3) * logisticsigmoid(-1 / 2 * (α * (σ^2 + 2 * λ.ρ0)))
            ) +
            80 *
            α *
            σ *
            (-78 + λ.λ3) *
            (σ^2 - 2 * λ.ρ0)^3 *
            (
                -logisticsigmoid((α * (σ^2 - 2 * λ.ρ0)) / 2) +
                logisticsigmoid((α * (σ^2 - 2 * λ.ρ0)) / 2)^2 +
                logisticsigmoid(-1 / 2 * (α * (σ^2 + 2 * λ.ρ0))) -
                logisticsigmoid(-1 / 2 * (α * (σ^2 + 2 * λ.ρ0)))^2
            ) +
            10 *
            α *
            σ *
            λ.λ4 *
            (σ^2 - 2 * λ.ρ0)^4 *
            (
                -logisticsigmoid((α * (σ^2 - 2 * λ.ρ0)) / 2) +
                logisticsigmoid((α * (σ^2 - 2 * λ.ρ0)) / 2)^2 +
                logisticsigmoid(-1 / 2 * (α * (σ^2 + 2 * λ.ρ0))) -
                logisticsigmoid(-1 / 2 * (α * (σ^2 + 2 * λ.ρ0)))^2
            ) +
            α *
            σ *
            λ.λ5 *
            (σ^2 - 2 * λ.ρ0)^5 *
            (
                -logisticsigmoid((α * (σ^2 - 2 * λ.ρ0)) / 2) +
                logisticsigmoid((α * (σ^2 - 2 * λ.ρ0)) / 2)^2 +
                logisticsigmoid(-1 / 2 * (α * (σ^2 + 2 * λ.ρ0))) -
                logisticsigmoid(-1 / 2 * (α * (σ^2 + 2 * λ.ρ0)))^2
            )
        ) / 3840
end


function Uσfunout(λ::TaylorParameters)
    σ ->
        (-λ.ρ0 + σ^2 / 2) * λ.λ1 +
        (-λ.ρ0 + σ^2 / 2)^2 * λ.λ2/2 +
        (-λ.ρ0 + σ^2 / 2)^3 * λ.λ3 / 6 +
        (-λ.ρ0 + σ^2 / 2)^4 * λ.λ4 / 24 +
        (-λ.ρ0 + σ^2 / 2)^5 * λ.λ5 / 120 - λ.c*σ
end
logisticsigmoid(x::Real) = inv(exp(-x) + one(x))
