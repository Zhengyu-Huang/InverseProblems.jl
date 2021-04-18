using Random
using Statistics
using Distributions
using LinearAlgebra
using StatsBase

include("Utility.jl")
"""
The implementation follows

@article{kantas2014sequential,
  tITle={Sequential Monte Carlo methods for high-dimensional inverse problems: A case study for the Navier--Stokes equations},
  author={Kantas, Nikolas and Beskos, Alexandros and Jasra, Ajay},
  journal={SIAM/ASA Journal on Uncertainty Quantification},
  volume={2},
  number={1},
  pages={464--489},
  year={2014},
  publisher={SIAM}
}


@article{beskos2015sequential,
  tITle={Sequential Monte Carlo methods for Bayesian elliptic inverse problems},
  author={Beskos, Alexandros and Jasra, Ajay and Muzaffer, Ege A and Stuart, Andrew M},
  journal={Statistics and Computing},
  volume={25},
  number={4},
  pages={727--737},
  year={2015},
  publisher={Springer}
}

When the densITy function is Φ/Z, 
The f_densITy function return log(Φ) instead of Φ
"""



mutable struct SMCObj{FT<:AbstractFloat, IT<:Int}
    "vector of parameter names"
    θ_names::Vector{String}
    "a vector of arrays of size N_ensemble x N_parameters containing the parameters (in each SMC ITeration a new array of parameters is added)"
    θ::Vector{Array{FT, 2}}
    "weights for each particles"
    weights::Vector{Array{FT, 1}}
    "observation"
    y::Vector{FT}
    "covariance of the observational noise, which is assumed to be normally distributed"
    Σ_η
    "function -1/2 (θ-θ₀)^T Σ₀^{-1} (y-θ₀)"   
    log_prior::Function
    "parameter size"   
    N_θ::IT
    "ensemble size"
    N_ens::IT
    "step_length for random walk MCMC"
    step_length::FT
    "threshold for resampling"
    M_threshold::FT
    "time step"
    Δt::FT
    "current time"
    t::FT
end

# outer constructors
function SMCObj(
    θ_names::Vector{String},
    # assume the prior is Gaussian 
    θ0_mean::Array{FT,1},
    θθ0_cov::Array{FT,2},
    y::Array{FT,1},
    Σ_η::Array{FT,2},
    N_ens::IT,
    step_length::FT,
    M_threshold::FT,
    N_t::IT;
    T::FT = 1.0) where {FT<:AbstractFloat, IT<:Int}
    
    Δt = T/N_t
    t = 0.0
    
    N_θ = length(θ0_mean)
    θ = Array{FT, 2}[] # array of Array{FT, 2}'s
    θ0 = rand(MvNormal(θ0_mean, θθ0_cov), N_ens)'
    push!(θ, θ0) # insert parameters at end of array (in this case just 1st entry)
    
    
    weights = Array{FT,1}[]  # array of Array{FT, 2}'s
    weights0 = zeros(FT, N_ens)
    fill!(weights0, 1.0/N_ens)
    push!(weights, weights0)
    
    log_prior = (θ::Array{FT,1}) -> (-0.5*(θ - θ0_mean)'/θθ0_cov*(θ - θ0_mean))
    
    
    SMCObj(θ_names, θ, weights, y, Σ_η, log_prior, N_θ, N_ens, step_length, M_threshold,  Δt, t)
end




function update_ensemble!(smc::SMCObj, ens_func::Function, Δt::FT=smc.Δt) where {FT<:AbstractFloat}
    t = smc.t
    
    N_θ, N_ens = smc.N_θ, smc.N_ens
    step_length = smc.step_length
    y, Σ_η, log_prior = smc.y, smc.Σ_η, smc.log_prior
    
    
    # prepare to update states and weights
    θ = smc.θ[end]
    weights  = smc.weights[end]
    θ_n = copy(θ)
    weights_n = copy(weights)


    # update weights: ω̂ᵐ_n = l_{n-1}(uᵐ_{n-1}) ω̂ᵐ_{n-1}
    g = ens_func(θ)


    # to avoid Inf in the update weights step, scale the weights during the computation
    # the step does not change the results but avoids numerical instability
    log_dweights = zeros(Float64, N_ens)
    for i = 1:N_ens
        log_dweights[i] = -0.5*(y - g[i,:])'/Σ_η*(y - g[i,:]) * Δt
    end
    log_dweight_scale = maximum(log_dweights)

    
    for i = 1:N_ens
        weights_n[i] = weights[i] * exp(log_dweights[i] - log_dweight_scale)
    end 
    weights_n .= weights_n/sum(weights_n)
    
    
    # sample uᵐ_{n} from K_n(uᵐ_{n-1}, ⋅)
    for i = 1:N_ens
        θ_n[i, :] = θ[i, :] + step_length * rand(Normal(0, 1), N_θ)
    end
    g_n = ens_func(θ_n)
    
    for i = 1:N_ens
        
        f = (t+Δt)*(-0.5*(y - g[i,:])'/Σ_η*(y - g[i,:])) + log_prior(θ[i, :])  
        f_n = (t+Δt)*(-0.5*(y - g_n[i,:])'/Σ_η*(y - g_n[i,:])) + log_prior(θ_n[i, :]) 
        α = min(1.0, exp(f_n - f))
        
        θ_n[i, :] = (rand(Bernoulli(α)) ? θ_n[i, :] : θ[i, :])
        
    end
    
    
    # Update and Analysis and Resample
    ESS = sum(weights_n)^2/sum(weights_n.^2)
    if ESS < smc.M_threshold
        θ = copy(θ_n)
        for i = 1:N_ens
            θ_n[i, :] .= θ[sample(Weights(weights_n)), :]
        end
        
        weights_n .= 1/N_ens
    end
    
    # update 
    smc.t += Δt
    push!(smc.θ, θ_n) 
    push!(smc.weights, weights_n)
    
end


function SMC_Run(s_param, forward::Function,
    θ0_mean::Array{FT,1}, θθ0_cov::Array{FT,2}, 
    y::Array{FT,1}, Σ_η,
    N_ens::IT, 
    step_length::FT,
    M_threshold::FT,
    N_t::IT;
    T::FT = 1.0) where {FT<:AbstractFloat, IT<:Int}
    
    θ_names = s_param.θ_names
    
    smcobj = SMCObj(θ_names , 
    θ0_mean , θθ0_cov , y , Σ_η, 
    N_ens , step_length, M_threshold, 
    N_t; T = T) 
    
    ens_func(θ_ens) = ensemble(s_param, θ_ens, forward)
    
    
    for i in 1:N_t
        
        update_ensemble!(smcobj, ens_func)
        
        
    end
    
    return smcobj
end