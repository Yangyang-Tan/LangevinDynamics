using DelimitedFiles
u0_1 = fill(0.6f0, 32, 32, 32, 2^6)

function MF_dataloader(i; file = "Updata")
    readdlm("data/MF/$file.dat")[i, :] ./ 1.0f0
end


function MS_dataloader(i; file = "msdata")
    readdlm("data/MF/$file.dat")[i]
end
myT=Float32
sol_O1 = modelA_3d_SDE_Simple_tex_prob(;
    u0 = u0_1,
    γ = 0.1f0,
    tspan = myT.((0.0, 2.0)),
    T = 160.0f0 / 197.33,
    dt = 0.01f0,
    x_grid = -2.5f0:5/197.33f0:2.5f0,
    δUδσ_grid = MF_dataloader(110),
    noise = sqrt(1f0),
)

##MFFit

###############################
## const noise
###############################
function MF_relaxtime_datasaver(
    j::Int,#Temperature
    u0::AbstractArray{Float32,4};
    file = "MFFit_c0",
    savefile = "relax_time_MF_Origin_muB=205_gamma8.0",
    γ = 8.0f0,
    α=0.0f0,
)
    δUδσ_grid = Float32.(MF_dataloader(j, file = "Updata_$file"))
    ms = Float32(MS_dataloader(j, file = "msdata_$file"))
    # u0_1 = fill(0.6f0, 32, 32, 32, 2^8)
    T0 = j / 197.33f0
    sol_3D_SDE = modelA_3d_SDE_Simple_tex_prob(;
        u0 = u0,
        γ = γ,
        tspan = myT.((0.0, 20.0*γ)),
        T = T0,
        dt = 0.01f0,
        x_grid = -2.5f0:5/197.33f0:2.5f0,
        δUδσ_grid = δUδσ_grid,
        noise = sqrt(α/γ),
    )
    writedlm("sims/eqcd_relax_phase/$savefile/T=$j.dat", sol_3D_SDE)
end

function etafunfit(x)
    exp(-0.0008813364606557306f0 * (-158 + x)^2)
end

plot(readdlm("data/MF/msdata_MFFit_c0.dat")[:,1])
plot!(etafunfit,0,250)


for alp in 0.5:0.5:2.5 
    mkdir("sims/eqcd_relax_phase/MFFit/MFFit_constnoise=$alp-gamma=gaussianfun_c=c0")
    for Tm = 1:2:250
        u0_1 = fill(1.6f0, 32, 32, 32, 2^6)
        MF_relaxtime_datasaver(
            Tm,
            u0_1,
            γ = Float32(etafunfit(Tm)),
            file = "MFFit_c0",
            savefile = "MFFit/MFFit_constnoise=$alp-gamma=gaussianfun_c=c0",
            α=alp,
        )
    end 
end



###c=c0, noise=sqrt(2*T0/γ)
function MF_relaxtime_datasaver(
    j::Int,#Temperature
    u0::AbstractArray{Float32,4};
    file = "MFFit_c0",
    savefile = "relax_time_MF_Origin_muB=205_gamma8.0",
    γ = 8.0f0,
)
    δUδσ_grid = Float32.(MF_dataloader(j, file = "Updata_$file"))
    ms = Float32(MS_dataloader(j, file = "msdata_$file"))
    # u0_1 = fill(0.6f0, 32, 32, 32, 2^8)
    T0 = j / 197.33f0
    sol_3D_SDE = modelA_3d_SDE_Simple_tex_prob(;
        u0 = u0,
        γ = γ,
        tspan = myT.((0.0, 20.0*γ)),
        T = T0,
        dt = 0.01f0,
        x_grid = -2.5f0:5/197.33f0:2.5f0,
        δUδσ_grid = δUδσ_grid,
        noise = sqrt(2 * T0 / γ),
    )
    writedlm("sims/eqcd_relax_phase/$savefile/T=$j.dat", sol_3D_SDE)
end



mkdir("sims/eqcd_relax_phase/MFFit/MFFit_2gammaT_gamma=gaussianfun_c=c0")
for Tm = 1:2:250
    u0_1 = fill(1.6f0, 32, 32, 32, 2^6)
    MF_relaxtime_datasaver(
        Tm,
        u0_1,
        γ = Float32(etafunfit(Tm)),
        file = "MFFit_c0",
        savefile = "MFFit/MFFit_2gammaT_gamma=gaussianfun_c=c0",
    )
end


mkdir("sims/eqcd_relax_phase/MFFit/MFGL_2gammaT_gamma=gaussianfun_c=c0")
for Tm = 1:2:250
    u0_1 = fill(1.6f0, 32, 32, 32, 2^6)
    MF_relaxtime_datasaver(
        Tm,
        u0_1,
        γ = Float32(etafunfit(Tm)),
        file = "MFGL_c0",
        savefile = "MFFit/MFGL_2gammaT_gamma=gaussianfun_c=c0",
    )
end



###c=c0, noise=sqrt(2*T0/γ) ini condition
function MF_relaxtime_datasaver(
    j::Int,#Temperature
    u0::AbstractArray{Float32,4};
    file = "MFFit_c0",
    savefile = "relax_time_MF_Origin_muB=205_gamma8.0",
    γ = 8.0f0,
)
    δUδσ_grid = Float32.(MF_dataloader(j, file = "Updata_$file"))
    ms = Float32(MS_dataloader(j, file = "msdata_$file"))
    # u0_1 = fill(0.6f0, 32, 32, 32, 2^8)
    T0 = j / 197.33f0
    sol_3D_SDE = modelA_3d_SDE_Simple_tex_prob(;
        u0 = u0,
        γ = γ,
        tspan = myT.((0.0, 20.0*γ)),
        T = T0,
        dt = 0.01f0,
        x_grid = -2.5f0:5/197.33f0:2.5f0,
        δUδσ_grid = δUδσ_grid,
        noise = sqrt(2 * T0 / γ),
    )
    writedlm("sims/eqcd_relax_phase/$savefile/T=$j.dat", sol_3D_SDE)
end


for ini in 0.0f0:0.4f0:1.6f0 
    mkdir("sims/eqcd_relax_phase/MFFit/MFFit_2gammaT_constini=$ini-gamma=gaussianfun_c=c0")
    for Tm = 1:2:250
        u0_1 = fill(ini, 32, 32, 32, 2^6)
        MF_relaxtime_datasaver(
            Tm,
            u0_1,
            γ = Float32(etafunfit(Tm)),
            file = "MFFit_c0",
            savefile = "MFFit/MFFit_2gammaT_constini=$ini-gamma=gaussianfun_c=c0",
        )
    end 
end


for ini in 0.0f0:0.4f0:1.6f0 
    mkdir("sims/eqcd_relax_phase/MFFit/MFFit_2gammaT_randini=$ini-gamma=gaussianfun_c=c0")
    for Tm = 1:2:250
        u0_1 = randn(Float32, 32, 32, 32, 2^8) .+ ini
        MF_relaxtime_datasaver(
            Tm,
            u0_1,
            γ = Float32(etafunfit(Tm)),
            file = "MFFit_c0",
            savefile = "MFFit/MFFit_2gammaT_randini=$ini-gamma=gaussianfun_c=c0",
        )
    end 
end