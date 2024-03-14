using DelimitedFiles
u0_1 = fill(0.6f0, 32, 32, 32, 2^6)

function MF_dataloader(i; file = "Updata")
    readdlm("data/MF/$file.dat")[i, :] ./ 1.0f0
end


function MS_dataloader(i; file = "msdata")
    readdlm("data/MF/$file.dat")[i]
end

sol_O1 = modelA_3d_SDE_Simple_tex_prob(;
    u0 = u0_1,
    γ = 8.0f0,
    tspan = myT.((0.0, 200.0)),
    T = 160.0f0 / 197.33,
    dt = 0.01f0,
    x_grid = -2.5f0:5/197.33f0:2.5f0,
    δUδσ_grid = MF_dataloader(110),
    ms = MS_dataloader(110),
)

plot(sol_O1[:, 1], sol_O1[:, 2])

##MFFit

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
        tspan = myT.((0.0, 200.0)),
        T = T0,
        dt = 0.01f0,
        x_grid = -2.5f0:5/197.33f0:2.5f0,
        δUδσ_grid = δUδσ_grid,
        noise = sqrt(2 * T0 / γ),
    )
    writedlm("sims/eqcd_relax_phase/$savefile/T=$j.dat", sol_3D_SDE)
end
mkdir("sims/eqcd_relax_phase/MFFit/MFFit_2gammaT_gamma=8.0_c=c0")
for Tm = 1:2:250
    u0_1 = fill(0.6f0, 32, 32, 32, 2^6)
    MF_relaxtime_datasaver(
        Tm,
        u0_1,
        γ = 8.0,
        file = "MFFit_c0",
        savefile = "MFFit/MFFit_2gammaT_gamma=8.0_c=c0",
    )
end


##c=c0, noise=sqrt(2*T0/γ), GL
mkdir("sims/eqcd_relax_phase/MFFit/MFGL_2gammaT_gamma=8.0_c=c0")
for Tm = 1:2:250
    u0_1 = fill(1.6f0, 32, 32, 32, 2^6)
    MF_relaxtime_datasaver(
        Tm,
        u0_1,
        γ = 8.0,
        file = "MFGL_c0",
        savefile = "MFFit/MFGL_2gammaT_gamma=8.0_c=c0",
    )
end


##c=c0, noise=sqrt(2*T0/γ)

mkdir("sims/eqcd_relax_phase/MFFit/MFFit_2gammaT_gamma=8.0_c=0.1c0")
for Tm = 1:2:250
    u0_1 = fill(0.6f0, 32, 32, 32, 2^6)
    MF_relaxtime_datasaver(
        Tm,
        u0_1,
        γ = 8.0,
        file = "MFFit_0.1c0",
        savefile = "MFFit/MFFit_2gammaT_gamma=8.0_c=0.1c0",
    )
end

###c=c0, noise=ms*coth(ms/2T)/γ
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
        tspan = myT.((0.0, 200.0)),
        T = T0,
        dt = 0.01f0,
        x_grid = -2.5f0:5/197.33f0:2.5f0,
        δUδσ_grid = δUδσ_grid,
        noise = sqrt(ms*coth(ms/(2*T0))/γ),
    )
    writedlm("sims/eqcd_relax_phase/$savefile/T=$j.dat", sol_3D_SDE)
end
mkdir("sims/eqcd_relax_phase/MFFit/MFFit_coth_gamma=8.0_c=c0")
for Tm = 1:2:250
    u0_1 = fill(0.6f0, 32, 32, 32, 2^6)
    MF_relaxtime_datasaver(
        Tm,
        u0_1,
        γ = 8.0,
        file = "MFFit_c0",
        savefile = "MFFit/MFFit_coth_gamma=8.0_c=c0",
    )
end
##c=c0, noise=ms*coth(ms/2T)/γ, GL
mkdir("sims/eqcd_relax_phase/MFFit/MFGL_coth_gamma=8.0_c=c0")
for Tm = 1:2:250
    u0_1 = fill(1.6f0, 32, 32, 32, 2^6)
    MF_relaxtime_datasaver(
        Tm,
        u0_1,
        γ = 8.0,
        file = "MFGL_c0",
        savefile = "MFFit/MFGL_coth_gamma=8.0_c=c0",
    )
end

###############################
##
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
        tspan = myT.((0.0, 200.0)),
        T = T0,
        dt = 0.01f0,
        x_grid = -2.5f0:5/197.33f0:2.5f0,
        δUδσ_grid = δUδσ_grid,
        noise = (1.592f0/1.951)*((1-α)*sqrt(ms*coth(ms/(2*T0))/γ)+α*sqrt(2 * T0 / γ)),
    )
    writedlm("sims/eqcd_relax_phase/$savefile/T=$j.dat", sol_3D_SDE)
end

for alp in 0.0:0.2:1.0 
    mkdir("sims/eqcd_relax_phase/MFFit/MFGL_alpha=$alp-gamma=8.0_c=c0")
    for Tm = 1:4:250
        u0_1 = fill(1.6f0, 32, 32, 32, 2^5)
        MF_relaxtime_datasaver(
            Tm,
            u0_1,
            γ = 8.0,
            file = "MFFit_c0",
            savefile = "MFFit/MFGL_alpha=$alp-gamma=8.0_c=c0",
            α=alp,
        )
    end 
end

mkdir("sims/eqcd_relax_phase/MFFit/MFGL_rescale_alpha=0-gamma=8.0_c=c0")
    for Tm = 1:4:250
        u0_1 = fill(1.6f0, 32, 32, 32, 2^5)
        MF_relaxtime_datasaver(
            Tm,
            u0_1,
            γ = 8.0,
            file = "MFFit_c0",
            savefile = "MFFit/MFGL_rescale_alpha=0-gamma=8.0_c=c0",
            α=0.0f0,
        )
    end


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
        tspan = myT.((0.0, 200.0)),
        T = T0,
        dt = 0.01f0,
        x_grid = -2.5f0:5/197.33f0:2.5f0,
        δUδσ_grid = δUδσ_grid,
        noise = sqrt(α/γ),
    )
    writedlm("sims/eqcd_relax_phase/$savefile/T=$j.dat", sol_3D_SDE)
end

for alp in 0.5:0.5:2.5 
    mkdir("sims/eqcd_relax_phase/MFFit/MFFit_constnoise=$alp-gamma=8.0_c=c0")
    for Tm = 1:2:250
        u0_1 = fill(1.6f0, 32, 32, 32, 2^6)
        MF_relaxtime_datasaver(
            Tm,
            u0_1,
            γ = 8.0,
            file = "MFFit_c0",
            savefile = "MFFit/MFFit_constnoise=$alp-gamma=8.0_c=c0",
            α=alp,
        )
    end 
end

for alp in 0.5:0.5:2.5 
    mkdir("sims/eqcd_relax_phase/MFFit/MFFit_constnoise=$alp-gamma=4.0_c=c0")
    for Tm = 1:2:250
        u0_1 = fill(1.6f0, 32, 32, 32, 2^6)
        MF_relaxtime_datasaver(
            Tm,
            u0_1,
            γ = 4.0,
            file = "MFFit_c0",
            savefile = "MFFit/MFFit_constnoise=$alp-gamma=4.0_c=c0",
            α=alp,
        )
    end 
end

for alp in 0.5:0.5:2.5 
    mkdir("sims/eqcd_relax_phase/MFFit/MFFit_constnoise=$alp-gamma=2.0_c=c0")
    for Tm = 1:2:250
        u0_1 = fill(1.6f0, 32, 32, 32, 2^6)
        MF_relaxtime_datasaver(
            Tm,
            u0_1,
            γ = 2.0,
            file = "MFFit_c0",
            savefile = "MFFit/MFFit_constnoise=$alp-gamma=2.0_c=c0",
            α=alp,
        )
    end 
end

for alp in 0.5:0.5:2.5 
    mkdir("sims/eqcd_relax_phase/MFFit/MFFit_constnoise=$alp-gamma=1.0_c=c0")
    for Tm = 1:2:250
        u0_1 = fill(1.6f0, 32, 32, 32, 2^6)
        MF_relaxtime_datasaver(
            Tm,
            u0_1,
            γ = 1.0,
            file = "MFFit_c0",
            savefile = "MFFit/MFFit_constnoise=$alp-gamma=1.0_c=c0",
            α=alp,
        )
    end 
end
