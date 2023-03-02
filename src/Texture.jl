using Dierckx

Uσ_CPU_fun=Spline1D(readdlm("data/phi.dat")[:,1]./(197.33),(readdlm("data/V.dat")[:,1])./(197.3).^4)
σrng=0.0:0.005:1.5

δUδσTextureMem = CuTextureArray(Float32.(derivative(Uσ_CPU_fun,σrng)))
# copyto!(ImlambdaTextureMem, Imlambda)
# copyto!(RelambdaTextureMem, Relambda)
δUδσTextureTex =CuTexture(δUδσTextureMem;interpolation = CUDA.CubicInterpolation())
plot(derivative(Uσ_CPU_fun,σrng)[1:10])
sign.(CUDA.randn(10,10))
