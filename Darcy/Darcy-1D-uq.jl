using LinearAlgebra
using Distributions
using Random
using SparseArrays
include("../Inversion/Plot.jl")
include("../Inversion/KalmanInversion.jl")
include("../Inversion/RWMCMC.jl")
include("Darcy-1D.jl")







    N, L = 512, 1.0
    obs_ΔN = 8
    d = 1.0
    τ = 3.0
    N_θ = 32
    θ_ind = Array(1:N_θ)
    darcy = Setup_Param(N_x, L, 
    N_KL, obs_ΔN, 
    N_θ, d, τ)
    N_y = darcy.N_y
    
    θ_ref = darcy.θ_ref
    κ = exp.(darcy.logκ)
    h = solve_Darcy_1D(darcy, κ)
    
    PyPlot.figure()
    plot_field(darcy, h, true)
    PyPlot.title("observation")
    PyPlot.figure()
    plot_field(darcy, darcy.logκ, false)
    PyPlot.title("logκ field")
    
    
    y = compute_obs(darcy, h)
    Σ_η = Array(Diagonal(fill(0.1^2, N_y)))
    
    
    # UKI 
    θ0_mean = zeros(Float64, N_θ) 
    θθ0_cov = Array(Diagonal(fill(1.0^2.0, N_θ)))
    N_iter = 20
    α_reg = 1.0
    update_freq = 1
    ukiobj = UKI_Run(darcy, forward, θ0_mean, θθ0_cov, y, Σ_η, α_reg, update_freq, N_iter)
    
    
    fig, (ax1, ax2, ax3) = PyPlot.subplots(ncols=3, figsize=(18,6))
    ites = Array(LinRange(1, N_iter, N_iter))
    errors = zeros(Float64, (2, N_iter))
    for i = 1:N_iter
        errors[1, i] = norm(darcy.logκ - compute_logκ(darcy, ukiobj.θ_mean[i]))/norm(darcy.logκ)
        errors[2, i] = 0.5*(ukiobj.g_mean[i] - ukiobj.y)'*(ukiobj.Σ_η\(ukiobj.g_mean[i] - ukiobj.y))
        errors[3, i] = norm(ukiobj.θθ_cov[i])
    end
    
    ax1.semilogy(ites, errors[1, :], "-o", fillstyle="none", markevery=2, label= "UKI")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Relative L₂ norm error of logκ")
    ax1.legend()
    ax1.grid(true)
    
    ax2.semilogy(ites, errors[2, :], "-o", fillstyle="none", markevery=2, label= "UKI")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Optimization error")
    ax2.grid(true)
    ax2.legend()
    
    ax3.semilogy(ites, errors[3, :], "-o", fillstyle="none", markevery=2, label= "UKI")
    ax3.set_xlabel("Iterations")
    ax3.set_ylabel("Frobenius norm of covariance")
    ax3.grid(true)
    ax3.legend()
    
    
    
    # fig, ax = PyPlot.subplots(figsize=(18,6))
    # xx = darcy.xx
    # ki_θ_mean  = ukiobj.θ_mean[end]
    # ki_θθ_cov = ukiobj.θθ_cov[end]
    # ki_θθ_std = sqrt.(diag(ukiobj.θθ_cov[end]))
    # ax.plot(θ_ind , ki_θ_mean,"-*", color="red", fillstyle="none",  label="UKI")
    # ax.plot(θ_ind , ki_θ_mean + 3.0*ki_θθ_std, color="red")
    # ax.plot(θ_ind , ki_θ_mean - 3.0*ki_θθ_std, color="red")
    # # errorbar(θ_ind, ki_θ_mean, 3.0*ki_θθ_std, color="red",  label="UKI")
    
    
    # # compute posterior distribution by MCMC
    # θθ0_cov = Array(Diagonal(fill(10.0^2.0, N_θ)))
    # f_density(u) = f_posterior(u, darcy, obs, obs_cov, θ0_mean , θθ0_cov) 
    # step_length = 1.0e-3# .0
    # n_ite , n_burn_in= 200000000, 50000000
    # # n_ite , n_burn_in= 50000, 10000
    # # us = RWMCMC(f_density, θ0_mean, step_length, n_ite; seed=42)
    # us = RWMCMC(f_density, θ_ref, step_length, n_ite; seed=42)
    
    # mcmc_mean = mean(us[n_burn_in:n_ite, :], dims=1)[:]
    # mcmc_std = std(us[n_burn_in:n_ite, :], dims=1)[:]
    # ax.plot(θ_ind , mcmc_mean,"-s", color="C2", fillstyle="none" , label="MCMC")
    # ax.plot(θ_ind , mcmc_mean + 3.0*mcmc_std, color ="C2")
    # ax.plot(θ_ind , mcmc_mean - 3.0*mcmc_std, color ="C2")
    # # errorbar(θ_ind, mcmc_mean, 3.0*mcmc_std, color="C2",  label="MCMC")
    
    
    # ax.plot(θ_ind , θ_ref, "--o", color="grey", fillstyle="none", label="Reference")
    # ax.legend()
    # # plot MCMC results 
    # ax.set_xlabel("θ indices")
    # fig.tight_layout()
    # fig.savefig("Darcy-1d-uq.png")
    
    
    
    
    # ########################################### Covaraince 
    # fig, ax = PyPlot.subplots(figsize=(18,6))
    # θ_cov_ind = Array(1:div(N_θ*(N_θ+1), 2))
    # ki_θiθj_cov, mcmc_θiθj_cov = zeros(Float64, div(N_θ*(N_θ+1), 2)), zeros(Float64, div(N_θ*(N_θ+1), 2))
    # ind_ij = 1
    # for i = 1:N_θ
    #     for j = i:N_θ
    #         ki_θiθj_cov[ind_ij]   = ki_θθ_cov[i,j]
    #         mcmc_θiθj_cov[ind_ij] = (us[n_burn_in:n_ite, i] .- mcmc_mean[i])' * (us[n_burn_in:n_ite, j] .- mcmc_mean[j])/(n_ite - n_burn_in)
            
    #         ind_ij += 1
    #     end
    # end
    
    # ax.plot(θ_cov_ind , ki_θiθj_cov,  "*", color="red", fillstyle="none",  label="UKI")
    # ax.plot(θ_cov_ind , mcmc_θiθj_cov, "s", color="C2",  fillstyle="none" , label="MCMC")
    
    
    # ax.legend()
    # # plot MCMC results 
    # ax.set_xlabel("Cov(θ₍ᵢ₎, θ₍ⱼ₎)")
    # fig.tight_layout()
    # fig.savefig("Darcy-1d-cov.png")
    # ########################################## 
    
    
    
    
    
    
    
    
    # fig, (ax1, ax2) = PyPlot.subplots(ncols=2, figsize=(14,6))
    # ites = Array(LinRange(1, N_iter, N_iter))
    # errors = zeros(Float64, (2, N_iter))
    # for i = 1:N_iter
        
    #     errors[1, i] = norm(darcy.logκ - compute_logκ(darcy, ukiobj.θ_mean[i]))/norm(darcy.logκ)
    #     errors[2, i] = 0.5*(ukiobj.g_mean[i] - ukiobj.g_t)'*(ukiobj.obs_cov\(ukiobj.g_mean[i] - ukiobj.g_t))
        
    # end
    
    # ax1.semilogy(ites, errors[1, :], "-o", fillstyle="none", markevery=10, label= "UKI")
    # ax2.set_xlabel("Iterations")
    # ax1.set_ylabel("Relative L₂ norm error")
    # ax1.legend()
    # ax1.grid(true)
    # ax2.semilogy(ites, errors[2, :], "-o", fillstyle="none", markevery=10, label= "UKI")
    # ax2.set_xlabel("Iterations")
    # ax2.set_ylabel("Optimization error")
    # ax2.grid(true)
    # ax2.legend()
    # fig.tight_layout()
    # fig.savefig("Darcy-1d-opt.png")