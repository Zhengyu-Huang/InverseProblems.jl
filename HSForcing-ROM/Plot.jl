using JLD2
using LinearAlgebra
using NNGCM
include("../Plot.jl")
include("../RUKI.jl")

@load "UKI_Obj_Ite_20.dat" rukiobj


"""
dG = [-1/(1 + |θ1|)^2, 0, 0, 0]
[-1/(1 + |θ1|)^2, -1/(1 + |θ2|)^2, 0, 0]
[0, 0, 1, 0]
[0, 0, 0, 1]
"""
function θθ_cov_trans(θ_bar_raw_arr::Array{Float64, 2}, θθ_cov_raw::Vector{Array{Float64, 2}})
    θθ_cov = Array{Float64,2}[]
    for i = 1:length(θθ_cov_raw)
        θ1, θ2 = θ_bar_raw_arr[1, i], θ_bar_raw_arr[2, i]
        dG  = [-1/(1 + abs(θ1))^2   0                  0 0; 
        -1/(1 + abs(θ1))^2   -1/(1 + abs(θ2))^2 0 0;
        0 0 1 0;
        0 0 0 1]
        push!(θθ_cov, dG*θθ_cov_raw[i]*dG')
    end
    return θθ_cov
end

"""
ka = 1/(1 + |θ1|)
ks = 1/(1 + |θ1|) + 1/(1 + |θ2|)
ΔT = |θ3|
Δθ = |θ4|
"""
function θ_bar_trans(θ_bar_raw_arr::Array{Float64, 2})
    θ_bar_arr = similar(θ_bar_raw_arr)
    θ_bar_arr[1, :] = 1.0 ./ (1 .+ abs.(θ_bar_raw_arr[1, :]))
    θ_bar_arr[2, :] = 1.0 ./ (1 .+ abs.(θ_bar_raw_arr[1, :])) + 1.0 ./ (1 .+ abs.(θ_bar_raw_arr[2, :]))
    θ_bar_arr[3, :] = θ_bar_raw_arr[3, :]
    θ_bar_arr[4, :] = θ_bar_raw_arr[4, :]
    
    return θ_bar_arr
end

function visualize(uki, θ_ref::Array{Float64, 1}, file_name::String)
    
    θ_bar_raw_arr = hcat(uki.θ_bar...)
    θ_bar_arr = θ_bar_trans(θ_bar_raw_arr)
    θθ_cov = θθ_cov_trans( θ_bar_raw_arr, uki.θθ_cov)
    
    N_θ, N_ite = size(θ_bar_arr,1), size(θ_bar_arr,2)-1
    ites = Array(LinRange(1, N_ite+1, N_ite+1))
    
    
    
    
    θθ_cov_arr = zeros(Float64, (N_θ, N_ite+1))
    for i = 1:N_ite+1
        for j = 1:N_θ
            θθ_cov_arr[j, i] = sqrt(θθ_cov[i][j,j])
        end
    end
    
    @info "final result is ", θ_bar_arr[:,end],  θθ_cov[end]
    
    errorbar(ites, θ_bar_arr[1,:], yerr=3.0*θθ_cov_arr[1,:], fmt="--o",fillstyle="none", label=L"k_a")
    #semilogy(ites, θ_bar_arr[1,:], "--o", fillstyle="none", label=L"k_a")
    semilogy(ites, fill(θ_ref[1], N_ite+1), "--", color="gray")
    
    errorbar(ites, θ_bar_arr[2,:], yerr=3.0*θθ_cov_arr[2,:], fmt="--o",fillstyle="none", label=L"k_s")
    #semilogy(ites, θ_bar_arr[2,:], "--o", fillstyle="none", label=L"k_s")
    semilogy(ites, fill(θ_ref[2], N_ite+1), "--", color="gray")
    
    errorbar(ites, θ_bar_arr[3,:], yerr=3.0*θθ_cov_arr[3,:], fmt="--o",fillstyle="none", label=L"\Delta T_y")
    #semilogy(ites, θ_bar_arr[3,:], "--o", fillstyle="none", label=L"ΔT_y")
    semilogy(ites, fill(θ_ref[3], N_ite+1), "--", color="gray")
    
    errorbar(ites, θ_bar_arr[4,:], yerr=3.0*θθ_cov_arr[4,:], fmt="--o",fillstyle="none", label=L"\Delta \theta_y")
    #semilogy(ites, θ_bar_arr[4,:], "--o", fillstyle="none", label=L"Δθ_z")
    semilogy(ites, fill(θ_ref[4], N_ite+1), "--", color="gray")
    
    
    xlabel("Iterations")
    legend()
    grid("on")
    tight_layout()
    savefig(file_name)
    close("all")
end

function contourf()
    Tmin, Tmax = 180.0, 315.0
    Umin, Umax = -30.0, 36.0
    n_level = 10
    
    Tlevels = LinRange(Tmin, Tmax, n_level)
    Ulevels = LinRange(Umin, Umax, n_level)
    @load "Data/HS_OpM_T42.dat"
    t_zonal_mean, u_zonal_mean = output_manager.t_zonal_mean, output_manager.u_zonal_mean 
    Sigma_Zonal_Mean_Contourf(output_manager, t_zonal_mean, "Contourf_T42.png"; levels=Tlevels, cmap="jet")
    Sigma_Zonal_Mean_Contourf(output_manager, u_zonal_mean, "ContourU_T42.png"; levels=Ulevels, cmap="jet")
    
    
    @load "Data/HS_OpM_T21.dat"
    t_zonal_mean, u_zonal_mean = output_manager.t_zonal_mean, output_manager.u_zonal_mean 
    Sigma_Zonal_Mean_Contourf(output_manager, t_zonal_mean, "Contourf_T21.png"; levels=Tlevels, cmap="jet")
    Sigma_Zonal_Mean_Contourf(output_manager, u_zonal_mean, "ContourU_T21.png"; levels=Ulevels, cmap="jet")
    
    
end
contourf()

θ_ref = [1.0/40.0; 1.0/4.0; 60.0; 10.0]
visualize(rukiobj, θ_ref, "GCM_Obj.pdf") 



