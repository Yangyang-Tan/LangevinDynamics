device!(1)
@time sol_3D_ODE = solve(
    langevin_3d_ODE_prob(;
        γ = 2.0f0,
        m2 = -1.0f0,
        λ = 5.0f0,
        J = 0.0f0,
        tspan = (0.0f0, 1.0f0),
        T = 100.0f0/197.33f0,
        para=taylordata[99],
        u0=CUDA.rand(3,3,3,2).*0f0,
        du0=CUDA.randn(3,3,3,2),
        # u0fun = x -> 0.1f0*CUDA.randn(32,32,32,2^9, 2),
    ),
    [VelocityVerlet(),
    Nystrom4(),
    IRKN4(),
    KahanLi6(),
    KahanLi8()
    ][end-1],
    # dtmax = 0.01f0,
    # split=true,
    # trajectories = 1,
    dt=0.02f0,
    # saveat = 0.0:50.0:200.0,
    save_everystep = false,
    save_start=true,
    save_end=true,
    # dense = false,
    # save_on=false,
    # initialize_save=false,
    calck=false,
    # callback=cb,
    abstol = 1e-1,
    reltol = 1e-1,
)
GC.gc(true)
CUDA.reclaim()
stack(sol_3D_ODE.u[1])

phiv0=sol_3D_ODE.u[1][2,:,:,:,:]
piv0=sol_3D_ODE.u[1][1,:,:,:,:]
phiv1=sol_3D_ODE.u[2][2,:,:,:,:]
piv1=sol_3D_ODE.u[2][1,:,:,:,:]
(sum(piv0.^2)/2)

getU(phiv0,LangevinDynamics.Uσfunout(taylordata[99]))

LangevinDynamics.funout(taylordata[99])
