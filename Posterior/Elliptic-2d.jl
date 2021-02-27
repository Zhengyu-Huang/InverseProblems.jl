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


function forward(s_param, θ::Array{Float64,1})
    x1, x2 = 0.25, 0.75
    θ1, θ2 = θ
    p = (x) -> θ2*x + exp(-θ1)*(-x^2/2 + x/2)
    return [p(x1) ; p(x2)]
end





function Ellitic_Posterior_Plot(update_freq::Int64 = 1)
    # observation and observation error covariance
    y = [27.5; 79.7]
    Σ_η = [0.1^2   0.0; 0.0  0.1^2]
    
    # (uninformative) prior mean and covariance
    μ0 = [0.0; 0.0] 
    Σ0    = [10.0^2  0.0; 0.0 10.0^2]
    
    s_param = Setup_Param(2, 2)
    
    # compute posterior distribution by MCMC
    logρ(θ) = log_bayesian_posterior(s_param, θ, forward, y, Σ_η, μ0, Σ0)
    step_length = 1.0
    N_iter , n_burn_in= 5000000, 1000000
    us = RWMCMC_Run(logρ, μ0, step_length, N_iter)
    
    # UKI initialization compute posterior distribution by UKI
    θ0_mean = μ0
    θθ0_cov = [1.0  0.0; 0.0 100.0]
    α_reg,  N_iter = 1.0, 20
    ukiobj = UKI_Run(s_param, forward, θ0_mean, θθ0_cov, y, Σ_η, α_reg, update_freq, N_iter)
    
    
    # plot UKI results at 5th, 10th, and 15th iterations
    fig, ax = PyPlot.subplots(ncols=3, sharex=false, sharey=true, figsize=(15,5))
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


Ellitic_Posterior_Plot()