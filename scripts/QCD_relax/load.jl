using DelimitedFiles
function eqcd_potential_dataloader(i;dim=3)
    lam1data =
        readdlm("data/eQCD_Input/eqcd_potential_data/lamdata/Tem$i/lam1_cut3.dat")[:, 1] .*
        1.6481059699913014^2
    lam2data =
        readdlm("data/eQCD_Input/eqcd_potential_data/lamdata/Tem$i/lam2_cut3.dat")[:, 1] .*
        1.6481059699913014^4
    lam3data =
        readdlm("data/eQCD_Input/eqcd_potential_data/lamdata/Tem$i/lam3_cut3.dat")[:, 1] .*
        1.6481059699913014^6
    lam4data =
        readdlm("data/eQCD_Input/eqcd_potential_data/Tem$i/buffer/lam4_nc.dat")[:, 1] .*
        1.6481059699913014^8
    lam5data =
        readdlm("data/eQCD_Input/eqcd_potential_data/Tem$i/buffer/lam5_nc.dat")[:, 1] .*
        1.6481059699913014^10
    rho0data =
        readdlm("data/eQCD_Input/eqcd_potential_data/Tem$i/buffer/rho0.dat")[:, 1] ./
        1.6481059699913014^2
    cdata =
        0f0*readdlm("data/eQCD_Input/eqcd_potential_data/Tem$i/buffer/c.dat")[:, 1] .*
        1.6481059699913014
    Tdata= readdlm("data/eQCD_Input/eqcd_potential_data/Tem$i/buffer/TMeV.dat")[:, 1]./197.33
    taylordata = TaylorParameters(
        Float32,
        [lam1data lam2data lam3data lam4data lam5data rho0data cdata][:,[1:dim...,6,7]],
    )[:,1]
    return QCDModelParameters(taylordata, Tdata)
end

function eqcd_Zt_dataloader(muB)
    readdlm("data/Ztdata/mub$muB.dat")[:, 1]
end
