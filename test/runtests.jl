using DrWatson, Test
@quickactivate "LangevinDynamics"

# Here you include files using `srcdir`
# include(srcdir("file.jl"))

# Run test suite
println("Starting tests")
ti = time()

@testset "LangevinDynamics tests" begin
    @test 1 == 1
end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti/60, digits = 3), " minutes")

@inline function fun(x)
    for i in x
        for j in x
            if i^3 + j^3 == 278029380032146
                println("x=",i," y=",j)
           end
        end
    end
end

@btime fun(1:100000)
@time fun(1:100000)
fun(1:100000)
using BenchmarkTools
278029380032146|>typeof
unsafe_trunc(Int32, 278029380032146)
