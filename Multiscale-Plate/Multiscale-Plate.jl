using NNFEM
using JLD2
using Statistics
using LinearAlgebra

stress_scale = 1.0e+5
strain_scale = 1

force_scale = 5.0
tid = 300


include("../Plot.jl")
include("CommonFuncs.jl")
include("../RExKI.jl")

mutable struct Params
    θ_name::Array{String, 1}
    θ_scale::Array{Float64,1}
    ρ::Float64
    tids::Array{Int64,1}
    force_scale::Float64
    
    NT::Int64
    T::Float64

    n_tids::Int64
    n_obs_point::Int64
    n_obs::Int64
    n_data::Int64
end

function Params(tids::Array{Int64,1}, n_obs_point::Int64 = 2, n_obs_time::Int64 = 200, T::Float64 = 200.0, NT::Int64 = 200)
    θ_name = ["E", "nu", "sigmaY", "K"] 
    θ_scale = [1.0e+5, 1, 1.0e+5, 1]
    
    fiber_fraction = 0.25
    ρ = 4.5*(1 - fiber_fraction) + 3.2*fiber_fraction
    
    force_scale = 5.0
    n_tids = length(tids)
    n_data = n_obs_point * n_obs_time * n_tids
    
    return Params(θ_name, θ_scale, ρ, tids, force_scale, NT, T,  n_tids, n_obs_point, n_obs_point * n_obs_time, n_data)
end

function Foward(phys_params::Params, θ::Array{Float64,1})
    θ_scale, ρ, tids, force_scale, n_data = phys_params.θ_scale, phys_params.ρ, phys_params.tids, phys_params.force_scale, phys_params.n_data

    n_obs = div(n_data, length(tids))
    obs = zeros(Float64, n_data)

    for tid = 1:length(tids)
        _, data = Run_Homogenized(θ, θ_scale, ρ, tids[tid], force_scale)
        obs[(tid-1)*n_obs+1:tid*n_obs] = data[:]
    end

    return obs
end


function Ensemble(phys_params::Params,  params_i::Array{Float64, 2})
    n_data = phys_params.n_data
    
    N_ens,  N_θ = size(params_i)
    
    g_ens = zeros(Float64, N_ens,  n_data)
    
    Threads.@threads for i = 1:N_ens 
        # g: N_ens x N_data
        g_ens[i, :] .= Foward(phys_params, params_i[i, :])
    end
    
    return g_ens
end



function ExKI(phys_params::Params, 
    t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
    θ0_bar::Array{Float64,1}, θθ0_cov::Array{Float64,2}, 
    α_reg::Float64, N_iter::Int64 = 100)
    
    
    parameter_names = ["E"]
    
    ens_func(θ_ens) = Ensemble(phys_params, θ_ens)
    
    
    exkiobj = ExKIObj(parameter_names,
    θ0_bar, 
    θθ0_cov,
    t_mean, # observation
    t_cov,
    α_reg)
    
    update_cov = 1
    for i in 1:N_iter
        
        
        update_ensemble!(exkiobj, ens_func) 
        
        if (update_cov > 0) && (i%update_cov == 0) 
            exkiobj.θθ_cov[1] = copy(exkiobj.θθ_cov[end])
        end
        
    end
    
    return exkiobj
end

function Multiscale_Test(phys_params::Params, 
    t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
    θ0_bar::Array{Float64,1}, θθ0_cov::Array{Float64,2}, 
    α_reg::Float64, 
    N_iter::Int64,
    ki_file = nothing)
    
    # optimization related plots
    fig_ite, ax_ite = PyPlot.subplots(ncols = 2, nrows=1, sharex=false, sharey=false, figsize=(12,6))
    
    if ki_file === nothing
        kiobj = ExKI(phys_params, t_mean, t_cov,  θ0_bar, θθ0_cov, α_reg, N_iter)
        @save "exkiobj.dat" kiobj
    else
        @load "exkiobj.dat" kiobj
    end

    ites = Array(LinRange(1, N_iter, N_iter))
    errors = zeros(Float64, (3, N_iter))
    for i = 1:N_iter
        errors[1, i] = NaN
        errors[2, i] = 0.5*(kiobj.g_bar[i] - kiobj.g_t)'*(kiobj.obs_cov\(kiobj.g_bar[i] - kiobj.g_t))
        errors[3, i] = norm(kiobj.θθ_cov[i])   
      
    end
    errors[3, 1] = norm(θθ0_cov) 
    ax_ite[1].semilogy(ites, errors[2, :], "-o", fillstyle="none", markevery=2 )
    ax_ite[2].semilogy(ites, errors[3, :], "-o", fillstyle="none", markevery=2 )
    
    ax_ite[1].set_xlabel("Iterations")
    ax_ite[1].set_ylabel("Optimization error")
    ax_ite[1].grid(true)
    ax_ite[2].set_xlabel("Iterations")
    ax_ite[2].set_ylabel("Frobenius norm")
    ax_ite[2].grid(true)
  
    fig_ite.tight_layout()
    fig_ite.savefig("Plate-error.png")


    # parameter plot
    N_θ = length(θ0_bar)
    θθ_std = zeros(Float64, (N_θ, N_iter+1))
    for i = 1:N_iter+1
        for j = 1:N_θ
            θθ_std[j, i] = sqrt(kiobj.θθ_cov[i][j,j])
        end
    end
    errorbar(ites, θ_bar_arr[1,:], yerr=3.0*θθ_std[1,:], fmt="--o",fillstyle="none", label="E")
    errorbar(ites, θ_bar_arr[2,:], yerr=3.0*θθ_std[2,:], fmt="--o",fillstyle="none", label="ν")
    errorbar(ites, θ_bar_arr[3,:], yerr=3.0*θθ_std[3,:], fmt="--o",fillstyle="none", label="σY")
    errorbar(ites, θ_bar_arr[4,:], yerr=3.0*θθ_std[4,:], fmt="--o",fillstyle="none", label="K")
    semilogy()
    xlabel("Iterations")
    legend()
    grid("on")
    tight_layout()
    savefig("Plate_theta.png")
    close("all")

    # test on 300
    tid = 300
    θ = kiobj.θ_bar[end]
    θ_scale, ρ, force_scale, n_tids, n_obs = phys_params.θ_scale, phys_params.ρ, phys_params.force_scale, phys_params.n_tids, phys_params.n_obs
    domain, obs = Run_Homogenized(θ, θ_scale, ρ, tid, force_scale)

    
    # only visulize the first point
    NT, T = phys_params.NT, phys_params.T
    ts = LinRange(0, T, NT)
    plot(ts, obs[:,1], label="dx")
    plot(ts, obs[:,2], label="dy")
end




