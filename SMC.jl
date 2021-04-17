using Random
using Statistics
using Distributions
using LinearAlgebra
using StatsBase
"""
The implementation follows

@article{kantas2014sequential,
  tInt64le={Sequential Monte Carlo methods for high-dimensional inverse problems: A case study for the Navier--Stokes equations},
  author={Kantas, Nikolas and Beskos, Alexandros and Jasra, Ajay},
  journal={SIAM/ASA Journal on Uncertainty Quantification},
  volume={2},
  number={1},
  pages={464--489},
  year={2014},
  publisher={SIAM}
}


@article{beskos2015sequential,
  tInt64le={Sequential Monte Carlo methods for Bayesian elliptic inverse problems},
  author={Beskos, Alexandros and Jasra, Ajay and Muzaffer, Ege A and Stuart, Andrew M},
  journal={Statistics and Computing},
  volume={25},
  number={4},
  pages={727--737},
  year={2015},
  publisher={Springer}
}

When the densInt64y function is Φ/Z, 
The f_densInt64y function return log(Φ) instead of Φ
"""



mutable struct SMCObj
    "vector of parameter names"
    unames::Vector{String}
    "a vector of arrays of size N_ensemble x N_parameters containing the parameters (in each SMC Int64eration a new array of parameters is added)"
    θ::Vector{Array{Float64, 2}}
    "weights for each particles"
    weights::Vector{Array{Float64, 1}}
    "Prior mean"
    θ0_bar::Array{Float64,1}
    "Prior convariance"
    θθ0_cov::Array{Float64, 2}
    "observation"
    g_t::Vector{Float64}
    "covariance of the observational noise, which is assumed to be normally distributed"
    obs_cov::Array{Float64, 2}
    "function -1/2 (θ-θ₀)^T Σ₀^{-1} (y-θ₀)"   
    f_log_prior::Function
    "parameter size"   
    N_θ::Int64
    "ensemble size"
    N_ens::Int64
    "step_length for random walk MCMC"
    step_length
    "threshold for resampling"
    M_threshold
    "time step"
    Δt::Float64
    "current time"
    t::Float64
end

# outer constructors
function SMCObj(
    parameter_names::Vector{String},
    θ0_bar::Array{Float64,1},
    θθ0_cov::Array{Float64,2},
    g_t::Array{Float64,1},
    obs_cov::Array{Float64,2},
    N_ens::Int64,
    step_length::Float64,
    M_threshold::Float64,
    N_t::Int64,
    T::Float64 = 1.0) 
    
    Δt = T/N_t
    t = 0.0

    Int64 = typeof(N_ens)
    # parameters
    θ = Array{Float64, 2}[] # array of Array{Float64, 2}'s

    N_θ = length(θ0_bar)
   
    θ0 = rand(MvNormal(θ0_bar, θθ0_cov), N_ens)'
    push!(θ, θ0) # insert parameters at end of array (in this case just 1st entry)

    f_log_prior = (θ::Array{Float64,1}) -> (-0.5*(θ - θ0_bar)'/θθ0_cov*(θ - θ0_bar))
    weights = Array{Float64,1}[]  # array of Array{Float64, 2}'s
    weights0 = zeros(Float64, N_ens)
    fill!(weights0, 1.0/N_ens)

    push!(weights, weights0)

    
    SMCObj(parameter_names, θ, weights, θ0_bar, θθ0_cov,  g_t, obs_cov, f_log_prior, N_θ, N_ens, step_length, M_threshold,  Δt, t)
end




function time_advance!(smc::SMCObj, ens_func::Function, Δt::Float64=smc.Δt)
    t = smc.t
    
    N_θ, N_ens = smc.N_θ, smc.N_ens
    step_length = smc.step_length
    g_t, obs_cov, f_log_prior = smc.g_t, smc.obs_cov, smc.f_log_prior
    # prepare to update states and weights
    θ_p = smc.θ[end]
    weights_p  = smc.weights[end]
    θ_n = copy(θ_p)
    weights_n = copy(weights_p)
    

    g_p = ens_func(θ_p)
    for i = 1:N_ens
        θ_n[i, :] = θ_p[i, :] + step_length * rand(Normal(0, 1), N_θ)
    end
    g_n = ens_func(θ_n)

    # to avoid Inf in the update weights step
    log_dweights = zeros(Float64, N_ens)
    for i = 1:N_ens
        
        log_dweights[i] = -0.5*(g_t - g_p[i,:])'/obs_cov*(g_t - g_p[i,:]) * Δt
    end
    log_dweight_scale = maximum(log_dweights)
 

    # update weights
    for i = 1:N_ens
        weights_n[i] = weights_p[i] * exp(-0.5*(g_t - g_p[i,:])'/obs_cov*(g_t - g_p[i,:]) * Δt - log_dweight_scale)
    end 

    weights_n .= weights_n/sum(weights_n)

    
    for i = 1:N_ens

        f_p = (t+Δt)*(-0.5*(g_t - g_p[i,:])'/obs_cov*(g_t - g_p[i,:])) + f_log_prior(θ_p[i, :])  
        f_n = (t+Δt)*(-0.5*(g_t - g_n[i,:])'/obs_cov*(g_t - g_n[i,:])) + f_log_prior(θ_n[i, :]) 
        
        
        α = min(1.0, exp(f_n - f_p))

        θ_n[i, :] = (rand(Bernoulli(α)) ? θ_n[i, :] : θ_p[i, :])
        
    end

    

    
    # Update and Analysis and Resample
    ESS = sum(weights_n)^2/sum(weights_n.^2)
    if ESS < smc.M_threshold
        θ_p = copy(θ_n)
        for i = 1:N_ens
            θ_n[i, :] .= θ_p[sample(Weights(weights_n)), :]
        end

        weights_n .= 1/N_ens
    end

    # update 
    smc.t += Δt
    push!(smc.θ, θ_n) 
    push!(smc.weights, weights_n)
    
end


