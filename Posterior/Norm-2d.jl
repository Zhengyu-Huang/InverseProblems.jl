using Random
using Distributions
using PyPlot
using LinearAlgebra
include("../Inversion/Plot.jl")
include("../Inversion/KalmanInversion.jl")
include("../Inversion/RWMCMC.jl")

mutable struct Setup_Param{IT<:Int}
    θ_names::Array{String,1}
    N_θ::IT
    N_y::IT
end

function Setup_Param(N_θ::IT, N_y::IT) where {IT<:Int}
    return Setup_Param(["θ"], N_θ, N_y)
end

x1_ref, x2_ref = -1.0, 2.0
lower_bound, upper_bound = -2.0, 2.0
c_to_u = (x -> log((x - lower_bound) / (upper_bound - x)))
# jacobian = (x -> 1.0 / (upper_bound - x) + 1.0 / (x - lower_bound))
u_to_c = (x -> upper_bound - (upper_bound - lower_bound) / (exp(x) + 1))
    
function forward(s_param, θ::Array{Float64,1})
    
    x1, x2 = u_to_c(θ[1]), θ[2]
    Δx = [x1 - x1_ref; x2 - x2_ref] 

    # output = [norm(Δx, 1); norm(Δx, 2); maximum(Δx)]#l^1, l^2, l^inf norm
    output = [Δx[1]; Δx[2]; maximum(Δx)]#l^1, l^2, l^inf norm
    
    # output = [sum(Δx); norm(Δx, 2); maximum(Δx)]#l^1, l^2, l^inf norm
    
    return output
end



function reg_forward(s_param, θ::Array{Float64,1})
    
    output = forward(s_param, θ)

    return [output; θ]
end





function Ellitic_Posterior_Plot(update_freq::Int64 = 1)
    # observation and observation error covariance
    y = forward(nothing, [x1_ref, x2_ref])
    Σ_η = Matrix(Diagonal(ones(3) * 0.1^2))
    
    # prior mean and covariance
    μ0 = [0.0; 3.0] 
    Σ0    = [0.5^2  0.0; 0.0 0.5^2]
    
    s_param = Setup_Param(2, 3)
    
    # compute posterior distribution by MCMC
    logρ(θ) = log_bayesian_posterior(s_param, θ, forward, y, Σ_η, μ0, Σ0)
    step_length = 1.0
    N_iter , n_burn_in= 5000000, 1000000
    us = RWMCMC_Run(logρ, μ0, step_length, N_iter)
    @info "posterior mean = ", sum(us[n_burn_in:end, :], dims=1)/size(us[n_burn_in:end, :], 1)
    # UKI initialization compute posterior distribution by UKI
    θ0_mean = μ0
    θθ0_cov = Σ0
    α_reg,  N_iter = 1.0, 20
    uki_Δt = 0.5
    ukiobj = UKI_Run(s_param, forward, θ0_mean, θθ0_cov, θ0_mean, θθ0_cov, y, Σ_η, uki_Δt, α_reg, update_freq, N_iter)
    
    # plot UKI results at 5th, 10th, and 15th iterations
    ncols = 3
    fig, ax = PyPlot.subplots(ncols=ncols, sharex=false, sharey=true, figsize=(15,5))
    for icol = 1:ncols
        # plot UKI results 
        ites = 5*icol
        Nx = 100; Ny = 200
        uki_θ_mean = ukiobj.θ_mean[ites]
        uki_θθ_cov = ukiobj.θθ_cov[ites]
        X,Y,Z = Gaussian_2d(uki_θ_mean, uki_θθ_cov, Nx, Ny)
        ax[icol].contour(X, Y, Z, 50)
        
        # plot MCMC results 
        everymarker = 1
        ax[icol].scatter(us[n_burn_in:everymarker:end, 1], us[n_burn_in:everymarker:end, 2], s = 1)
        
    end
    
    fig.tight_layout()
end




function reg_Ellitic_Posterior_Plot(update_freq::Int64 = 1)
    # observation and observation error covariance
    y = forward(nothing, [x1_ref, x2_ref])
    Σ_η = Matrix(Diagonal(ones(3) * 0.1^2))
    
    # prior mean and covariance
    μ0 = [0.0; 3.0] 
    Σ0    = [0.5^2  0.0; 0.0 0.5^2]

    Nθ, Ny = 2, 3
    s_param = Setup_Param(Nθ, Ny)
    
    
    reg_Σ_η = [Σ_η zeros(Ny, Nθ); zeros(Nθ, Ny) Σ0]
    reg_y = [y ; μ0]
    reg_s_param = Setup_Param(Nθ, Ny + Nθ)
    
    # compute posterior distribution by MCMC
    logρ(θ) = log_bayesian_posterior(s_param, θ, forward, y, Σ_η, μ0, Σ0)
    step_length = 1.0
    N_iter , n_burn_in= 500000, 100000
    us = RWMCMC_Run(logρ, μ0, step_length, N_iter)
    @info "posterior mean = ", sum(us[n_burn_in:end, :], dims=1)/size(us[n_burn_in:end, :], 1)
    # UKI initialization compute posterior distribution by UKI
    θ0_mean = μ0
    θθ0_cov = Σ0
    α_reg,  N_iter = 1.0, 100
    uki_Δt = 0.5
    ukiobj = UKI_Run(reg_s_param, reg_forward, θ0_mean, θθ0_cov, θ0_mean, θθ0_cov, reg_y, reg_Σ_η, uki_Δt, α_reg, update_freq, N_iter; unscented_transform = "modified-2n+1")
    
    # plot UKI results at 5th, 10th, and 15th iterations
    ncols = 3
    fig, ax = PyPlot.subplots(ncols=ncols, sharex=false, sharey=true, figsize=(15,5))
    for icol = 1:ncols
        # plot UKI results 
        ites = 5*icol
        Nx = 100; Ny = 200
        uki_θ_mean = ukiobj.θ_mean[ites]
        uki_θθ_cov = ukiobj.θθ_cov[ites]
        X,Y,Z = Gaussian_2d(uki_θ_mean, uki_θθ_cov, Nx, Ny)
        ax[icol].contour(X, Y, Z, 50)
        
        # plot MCMC results 
        everymarker = 1
        ax[icol].scatter(us[n_burn_in:everymarker:end, 1], us[n_burn_in:everymarker:end, 2], s = 1)
        
    end
    
    fig.tight_layout()
end

reg_Ellitic_Posterior_Plot()