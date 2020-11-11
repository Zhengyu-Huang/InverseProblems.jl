using JLD2
using Statistics
using LinearAlgebra
using Distributions
using Random
using SparseArrays
include("../Plot.jl")
include("Darcy-1D.jl")
include("../RExKI.jl")
include("../RWMCMC.jl")


function run_Darcy()
    N, L = 256, 1.0
    obs_ΔN = 10
    α = 2.0
    τ = 3.0
    KL_trunc = 64
    darcy = Param_Darcy(N, obs_ΔN, L, KL_trunc, α, τ)
    
    κ = exp.(darcy.logκ)
    h = solve_GWF(darcy, κ)
    plot_field(darcy, h, true, "Figs/Darcy-1D-obs-ref.pdf")
    plot_field(darcy, darcy.logκ, false, "Figs/Darcy-1D-logk-ref.pdf")
end  




function run_Darcy_ensemble(darcy::Param_Darcy, params_i::Array{Float64, 2})
    N_ens,  N_θ = size(params_i)
    
    g_ens = zeros(Float64, N_ens, darcy.n_data)
    
    for i = 1:N_ens
        
        logκ = compute_logκ(darcy, params_i[i, :])
        κ = exp.(logκ)
        
        h = solve_GWF(darcy, κ)
        
        obs = compute_obs(darcy, h)
        
        # g: N_ens x N_data
        g_ens[i,:] .= obs 
    end
    
    return g_ens
end


function ExKI_Run(t_mean, t_cov, θ_bar, θθ_cov,  darcy::Param_Darcy,  α_reg::Float64, N_iter::Int64 = 100)
    parameter_names = ["logκ"]
    
    ens_func(θ_ens) = run_Darcy_ensemble(darcy, θ_ens)
    
    exkiobj = ExKIObj(parameter_names,
    θ_bar, 
    θθ_cov,
    t_mean, # observation
    t_cov,
    α_reg)
    
    update_cov = 1
    
    for i in 1:N_iter
        
        params_i = deepcopy(exkiobj.θ_bar[end])
        
        @info "L₂ error of params_i :", norm(darcy.u_ref[1:length(params_i)] - params_i), " / ",  norm(darcy.u_ref[1:length(params_i)])
        
        logκ_i = compute_logκ(darcy, params_i)
        
        @info "F error of logκ :", norm(darcy.logκ - logκ_i), " / ",  norm(darcy.logκ )
        
        
        update_ensemble!(exkiobj, ens_func) 
        
        @info "F error of data_mismatch :", (exkiobj.g_bar[end] - exkiobj.g_t)'*(exkiobj.obs_cov\(exkiobj.g_bar[end] - exkiobj.g_t))
        
        @info "F norm of the covariance: ", norm(exkiobj.θθ_cov[end])
        
        if (update_cov > 0) && (i%update_cov == 0) 
            exkiobj.θθ_cov[1] = copy(exkiobj.θθ_cov[end])
        end
        
    end
    
    return exkiobj
    
end



function f_posterior(u::Array{Float64,1}, darcy, obs::Array{Float64,1}, obs_cov::Array{Float64,2}, μ0::Array{Float64,1}, cov0::Array{Float64,2})
    
    logκ = compute_logκ(darcy, u)
    κ = exp.(logκ)  
    h = solve_GWF(darcy, κ)
    Gu = compute_obs(darcy, h)
    
    
    Φ = - 0.5*(obs - Gu)'/obs_cov*(obs - Gu) - 0.5*(u - μ0)'/cov0*(u - μ0)

    return Φ
end




function Darcy_1d_uq()
N, L = 512, 1.0
obs_ΔN = 8
α = 1.0
τ = 3.0
N_θ = 32
θ_ind = Array(1:N_θ)
darcy = Param_Darcy(N, obs_ΔN, L, N_θ, α, τ)

u_ref = darcy.u_ref
κ = exp.(darcy.logκ)
h = solve_GWF(darcy, κ)

obs = compute_obs(darcy, h)
obs_cov = Array(Diagonal(fill(0.1^2, length(obs))))
θ0_bar = zeros(Float64, N_θ)  # mean 
θθ0_cov = Array(Diagonal(fill(1.0^2.0, N_θ)))

fig, ax = PyPlot.subplots(figsize=(18,6))

N_ite = 50
kiobj = ExKI_Run(obs, obs_cov, θ0_bar, θθ0_cov, darcy,  1.0, N_ite)
xx = darcy.xx
ki_θ_bar  = kiobj.θ_bar[end]
ki_θθ_cov = kiobj.θθ_cov[end]
ki_θθ_std = sqrt.(diag(kiobj.θθ_cov[end]))
ax.plot(θ_ind , ki_θ_bar,"-*", color="red", fillstyle="none",  label="UKI")
ax.plot(θ_ind , ki_θ_bar + 3.0*ki_θθ_std, color="red")
ax.plot(θ_ind , ki_θ_bar - 3.0*ki_θθ_std, color="red")
# errorbar(θ_ind, ki_θ_bar, 3.0*ki_θθ_std, color="red",  label="UKI")


# compute posterior distribution by MCMC
θθ0_cov = Array(Diagonal(fill(10.0^2.0, N_θ)))
f_density(u) = f_posterior(u, darcy, obs, obs_cov, θ0_bar , θθ0_cov) 
step_length = 1.0e-3# .0
n_ite , n_burn_in= 50000000, 10000000
# n_ite , n_burn_in= 50000, 10000
# us = RWMCMC(f_density, θ0_bar, step_length, n_ite; seed=42)
us = RWMCMC(f_density, u_ref, step_length, n_ite; seed=42)

mcmc_mean = mean(us[n_burn_in:n_ite, :], dims=1)[:]
mcmc_std = std(us[n_burn_in:n_ite, :], dims=1)[:]
ax.plot(θ_ind , mcmc_mean,"-s", color="C2", fillstyle="none" , label="MCMC")
ax.plot(θ_ind , mcmc_mean + 3.0*mcmc_std, color ="C2")
ax.plot(θ_ind , mcmc_mean - 3.0*mcmc_std, color ="C2")
# errorbar(θ_ind, mcmc_mean, 3.0*mcmc_std, color="C2",  label="MCMC")


ax.plot(θ_ind , u_ref, "--o", color="grey", fillstyle="none", label="Reference")
ax.legend()
# plot MCMC results 
ax.set_xlabel("θ indices")
fig.savefig("Darcy-1d-uq.png")




########################################### Covaraince 
fig, ax = PyPlot.subplots(figsize=(18,6))
θ_cov_ind = Array(1:div(N_θ*(N_θ+1), 2))
ki_θiθj_cov, mcmc_θiθj_cov = zeros(Float64, div(N_θ*(N_θ+1), 2)), zeros(Float64, div(N_θ*(N_θ+1), 2))
ind_ij = 1
for i = 1:N_θ
    for j = i:N_θ
        ki_θiθj_cov[ind_ij]   = ki_θθ_cov[i,j]
        mcmc_θiθj_cov[ind_ij] = (us[n_burn_in:n_ite, i] .- mcmc_mean[i])' * (us[n_burn_in:n_ite, j] .- mcmc_mean[j])/(n_ite - n_burn_in)

        ind_ij += 1
    end
end

ax.plot(θ_cov_ind , ki_θiθj_cov,  "*", color="red", fillstyle="none",  label="UKI")
ax.plot(θ_cov_ind , mcmc_θiθj_cov, "s", color="C2",  fillstyle="none" , label="MCMC")


ax.legend()
# plot MCMC results 
ax.set_xlabel("Cov(θᵢ, θⱼ)")
fig.savefig("Darcy-1d-cov.png")
########################################## 








fig, (ax1, ax2) = PyPlot.subplots(ncols=2, figsize=(14,6))
ites = Array(LinRange(1, N_ite, N_ite))
errors = zeros(Float64, (2, N_ite))
for i = 1:N_ite
    
    errors[1, i] = norm(darcy.logκ - compute_logκ(darcy, kiobj.θ_bar[i]))/norm(darcy.logκ)
    errors[2, i] = 0.5*(kiobj.g_bar[i] - kiobj.g_t)'*(kiobj.obs_cov\(kiobj.g_bar[i] - kiobj.g_t))
    
end

ax1.semilogy(ites, errors[1, :], "-o", fillstyle="none", markevery=10, label= "UKI")
ax2.set_xlabel("Iterations")
ax1.set_ylabel("Relative L₂ norm error")
ax1.legend()
ax1.grid(true)
ax2.semilogy(ites, errors[2, :], "-o", fillstyle="none", markevery=10, label= "UKI")
ax2.set_xlabel("Iterations")
ax2.set_ylabel("Optimization error")
ax2.grid(true)
ax2.legend()
fig.savefig("Darcy-1d-opt.png")
end

Darcy_1d_uq()