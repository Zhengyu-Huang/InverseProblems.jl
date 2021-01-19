using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions


"""
UKSObj{FT<:AbstractFloat, IT<:Int}
Struct that is used in Continous UKI, including UKI in the continous time limit and UKS

"""
mutable struct UKSObj{FT<:AbstractFloat, IT<:Int}
    "vector of parameter names (never used)"
     unames::Vector{String}
    "a vector of arrays of size N_ensemble x N_parameters containing the mean of the parameters (in each uks iteration a new array of mean is added)"
     θ_bar::Vector{Array{FT}}
     "a vector of arrays of size N_ensemble x (N_parameters x N_parameters) containing the covariance of the parameters (in each uks iteration a new array of cov is added)"
     θθ_cov::Vector{Array{FT, 2}}
     "vector of observations (length: N_data); mean of all observation samples"
     g_t::Vector{FT}
     "covariance of the observational noise"
     obs_cov::Array{FT, 2}
     "a vector of arrays of size N_ensemble x N_g containing the predicted observation (in each uks iteration a new array of predicted observation is added)"
     g_bar::Vector{Array{FT}}
     "ensemble size"
     N_ens::IT
     "parameter size g = G(θ)"
     N_θ::IT
     "observation size g = G(θ)"
     N_g::IT
     "weights in uks"
     sample_weights::Array{FT}
     μ_weights::Array{FT} 
     cov_weights::Array{FT} 
     "Time stepping"
     Δt0::FT
     t::FT

end



"""
UKSObj Constructor 
parameter_names::Vector{String} : parameter name vector
θ0_bar::Array{FT} : prior mean
θθ0_cov::Array{FT, 2} : prior covariance
g_t::Vector{FT} : observation 
obs_cov::Array{FT, 2} : observation covariance
α_reg::Float64 : regularization parameter toward θ0 (0 < α_reg <= 1), default should be 1, without regulariazion
"""
function UKSObj(parameter_names::Vector{String},
                θ0_bar::Array{FT}, 
                θθ0_cov::Array{FT, 2},
                g_t::Vector{FT}, # observation
                obs_cov::Array{FT, 2},
                UKF_modify::Bool,
                Δt0::FT) where {FT<:AbstractFloat}

    # ensemble size
    N_θ = size(θ0_bar,1)
    N_g = size(g_t, 1)

    N_ens = 2*N_θ+1
    IT = typeof(N_ens)

 
    

    

    θ_bar = Array{FT}[]  # array of Array{FT, 2}'s
    push!(θ_bar, θ0_bar) # insert parameters at end of array (in this case just 1st entry)
    θθ_cov = Array{FT,2}[] # array of Array{FT, 2}'s
    push!(θθ_cov, θθ0_cov) # insert parameters at end of array (in this case just 1st entry)
    g_bar = Array{FT}[]  # array of Array{FT, 2}'s
   

    # todo parameters λ, α, β
    sample_weights = zeros(FT, N_θ)
    μ_weights = zeros(FT, N_ens)
    cov_weights = zeros(FT, N_ens)
    # α, β = 1.0e-3, 2.0
    κ = 0.0
    β = 2.0
    α = min(sqrt(4/(N_θ + κ)), 1.0)
    
    λ = α^2*(N_θ + κ) - N_θ

    sample_weights[1:N_θ]     .=  sqrt(N_θ + λ)

    μ_weights[1] = λ/(N_θ + λ)
    μ_weights[2:N_ens] .= 1/(2(N_θ + λ))

    cov_weights[1] = λ/(N_θ + λ) + 1 - α^2 + β
    cov_weights[2:N_ens] .= 1/(2(N_θ + λ))
    if UKF_modify
        μ_weights[1] = 1
        μ_weights[2:N_ens] .= 0
    end



    UKSObj{FT,IT}(parameter_names, θ_bar, θθ_cov, g_t, obs_cov, g_bar, N_ens, N_θ, N_g, 
                  sample_weights, μ_weights, cov_weights, 
                  Δt0, 0.0)

end


"""
construct_sigma_ensemble
Construct the sigma ensemble, based on the mean x_bar, and covariance x_cov
"""
function construct_sigma_ensemble(uks::UKSObj{FT}, x_bar::Array{FT}, x_cov::Array{FT,2}) where {FT<:AbstractFloat}
    N_ens = uks.N_ens
    N_x = size(x_bar,1)
    @assert(N_ens == 2*N_x+1)

    sample_weights = uks.sample_weights


    chol_xx_cov = cholesky(Hermitian(x_cov)).L

    x = zeros(Float64, 2*N_x+1, N_x)
    x[1, :] = x_bar
    for i = 1: N_x
        x[i+1,     :] = x_bar + sample_weights[i]*chol_xx_cov[:,i]
        x[i+1+N_x, :] = x_bar - sample_weights[i]*chol_xx_cov[:,i]
    end

    return x
end


"""
construct_mean x_bar from ensemble x
"""
function construct_mean(uks::UKSObj{FT}, x::Array{FT,2}) where {FT<:AbstractFloat}
    N_ens, N_x = size(x)

    @assert(uks.N_ens == N_ens)

    x_bar = zeros(Float64, N_x)

    μ_weights = uks.μ_weights

    
    for i = 1: N_ens
        x_bar += μ_weights[i]*x[i,:]
    end

    return x_bar
end

"""
construct_cov xx_cov from ensemble x and mean x_bar
"""
function construct_cov(uks::UKSObj{FT}, x::Array{FT,2}, x_bar::Array{FT}) where {FT<:AbstractFloat}
    N_ens, N_x = uks.N_ens, size(x_bar,1)
    
    cov_weights = uks.cov_weights

    xx_cov = zeros(FT, N_x, N_x)

    for i = 1: N_ens
        xx_cov .+= cov_weights[i]*(x[i,:] - x_bar)*(x[i,:] - x_bar)'
    end

    return xx_cov
end

"""
construct_cov xy_cov from ensemble x and mean x_bar, ensemble y and mean y_bar
"""
function construct_cov(uks::UKSObj{FT}, x::Array{FT,2}, x_bar::Array{FT}, y::Array{FT,2}, y_bar::Array{FT}) where {FT<:AbstractFloat}
    N_ens, N_x, N_y = uks.N_ens, size(x_bar,1), size(y_bar,1)
    
    cov_weights = uks.cov_weights

    xy_cov = zeros(FT, N_x, N_y)

    for i = 1: N_ens
        xy_cov .+= cov_weights[i]*(x[i,:] - x_bar)*(y[i,:] - y_bar)'
    end

    return xy_cov
end



"""
update uks struct
ens_func: The function g = G(θ)
define the function as 
    ens_func(θ_ens) = MyG(phys_params, θ_ens, other_params)
"""
function update_ensemble!(uks::UKSObj{FT}, ens_func::Function) where {FT}

    N_ens, N_g  = uks.N_ens, uks.N_g
    

    θ_bar  = copy(uks.θ_bar[end])
    θθ_cov = copy(uks.θθ_cov[end])

    θ0_bar  = copy(uks.θ_bar[1])
    θθ0_cov = copy(uks.θθ_cov[1])


    Σ_η  = uks.obs_cov
    Σ_0 = θθ0_cov

    Δt = uks.Δt0

    θ = construct_sigma_ensemble(uks, θ_bar, θθ_cov)
    g = zeros(FT, N_ens, N_g)
    g .= ens_func(θ)
    g_bar = construct_mean(uks, g)
    θg_cov = construct_cov(uks, θ, θ_bar, g, g_bar)

    
    θ_bar  .=  (I + Δt*θθ_cov/Σ_0)\(θ_bar + Δt * (θg_cov*(Σ_η\(uks.g_t - g_bar)) + θθ_cov*(Σ_0\θ0_bar))) 
    θθ_cov .=  (θθ_cov - 2Δt * (θg_cov*(Σ_η\θg_cov') + θθ_cov*(Σ_0\θθ_cov)))/(1 - 2Δt)
  
    

    # store new parameters (and observations)
    push!(uks.g_bar,   g_bar) # N_ens x N_data
    push!(uks.θ_bar,   θ_bar) # N_ens x N_params
    push!(uks.θθ_cov, θθ_cov) # N_ens x N_data

    uks.t += Δt

end


