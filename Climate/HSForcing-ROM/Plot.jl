using JLD2
using LinearAlgebra
using NNGCM

include("../../Inversion/Plot.jl")
# include("HS.jl")
include("../../Inversion/KalmanInversion.jl")


@load "UKI_Obj_Ite_10.dat" ukiobj


"""
dG = [-1/(1 + |θ1|)^2, 0, 0, 0]
[-1/(1 + |θ1|)^2, -1/(1 + |θ2|)^2, 0, 0]
[0, 0, 1, 0]
[0, 0, 0, 1]
"""
function θθ_cov_trans(θ_mean_raw_arr::Array{Float64, 2}, θθ_cov_raw::Vector{Array{Float64, 2}})
    θθ_cov = Array{Float64,2}[]
    for i = 1:length(θθ_cov_raw)
        θ1, θ2, θ3, θ4 = θ_mean_raw_arr[:, i]
        dG  = [-exp(θ1)/(1 + exp(θ1))^2   0    0  0; 
               0     -exp(θ2)/(1 + exp(θ2))^2   0  0;
               0     0       -100*exp(θ3)/(1 + exp(θ3))^2             0;
               0     0         0                        -100*exp(θ4)/(1 + exp(θ4))^2]
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
function θ_mean_trans(θ_mean_raw_arr::Array{Float64, 2})
    θ_mean_arr = similar(θ_mean_raw_arr)
    for j = 1:size(θ_mean_raw_arr,2)
    θ_mean_arr[1,j] = 1.0/(1 + exp(θ_mean_raw_arr[1,j]))
    θ_mean_arr[2,j] = 1.0/(1 + exp(θ_mean_raw_arr[2,j]))
    θ_mean_arr[3,j] = 100.0/(1 + exp(θ_mean_raw_arr[3,j]))
    θ_mean_arr[4,j] = 100.0/(1 + exp(θ_mean_raw_arr[4,j]))
    end
    
    return θ_mean_arr
end

function visualize(uki, θ_ref::Array{Float64, 1}, file_name::String)
    
    θ_mean_raw_arr = hcat(uki.θ_mean...)
    θ_mean_arr = θ_mean_trans(θ_mean_raw_arr)
    θθ_cov = θθ_cov_trans( θ_mean_raw_arr, uki.θθ_cov)
    
    N_θ, N_ite = size(θ_mean_arr,1), size(θ_mean_arr,2)-1
    ites = Array(LinRange(0, N_ite, N_ite+1))
    
    
    
    
    θθ_cov_arr = zeros(Float64, (N_θ, N_ite+1))
    for i = 1:N_ite+1
        for j = 1:N_θ
            θθ_cov_arr[j, i] = sqrt(θθ_cov[i][j,j])
        end
    end
    
    @info "final result is ", θ_mean_arr[:,end],  θθ_cov[end]
    
    figure(figsize = (7.5, 4.8))
    errorbar(ites, θ_mean_arr[1,:], yerr=3.0*θθ_cov_arr[1,:], fmt="--o",fillstyle="none", label=L"k_a\ (day^{-1})")
    semilogy(ites, fill(θ_ref[1], N_ite+1), "--", color="gray")

    errorbar(ites, θ_mean_arr[2,:], yerr=3.0*θθ_cov_arr[2,:], fmt="--o",fillstyle="none", label=L"k_s\ (day^{-1})")
    semilogy(ites, fill(θ_ref[2], N_ite+1), "--", color="gray")

    errorbar(ites, θ_mean_arr[3,:], yerr=3.0*θθ_cov_arr[3,:], fmt="--o",fillstyle="none", label=L"\Delta T_y\ (K)")
    semilogy(ites, fill(θ_ref[3], N_ite+1), "--", color="gray")

    errorbar(ites, θ_mean_arr[4,:], yerr=3.0*θθ_cov_arr[4,:], fmt="--o",fillstyle="none", label=L"\Delta \theta_y\ (K)")
    semilogy(ites, fill(θ_ref[4], N_ite+1), "--", color="gray")

    xlabel("Iterations")
    legend(bbox_to_anchor=(0.95, 0.8))

    tight_layout()
    savefig(file_name)
    close("all")
end


θ_ref = [1.0/40.0; 1.0/4.0; 60.0; 10.0]
visualize(ukiobj, θ_ref, "GCM_Obj.pdf") 



