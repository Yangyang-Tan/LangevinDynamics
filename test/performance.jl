using DrWatson
@quickactivate :LangevinDynamics
using BenchmarkTools
u0 = init_langevin_2d(LangevinDynamics.xyd_brusselator)
u0_GPU = CuArray(u0)
du = similar(u0_GPU)

langevin_2d_loop_GPU(du, u0_GPU, p, 0.0f0)
langevin_2d_loop_shem_GPU(du, u0_GPU, p, 0.0f0)
p = LangevinDynamics.myT.((0.02, step(LangevinDynamics.xyd_brusselator), 0.1))
t_it = @belapsed begin
    langevin_2d_loop_GPU($du, $u0_GPU, $p, 0.0f0)
    synchronize()
end

t_it2 = @belapsed begin
    @cuda blocks = $blocks threads = $threads shmem = prod($threads .+ 2) * sizeof(Float64) update_temperature!(
        $T2, $T, $Ci, $lam, $dt, $_dx, $_dy
    )
    synchronize()
end
T_eff = (2 * 1 + 1) * 1 / 1e9 * nx * ny * sizeof(Float32) / t_it
