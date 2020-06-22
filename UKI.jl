using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions


"""
    UKIObj{FT<:AbstractFloat, IT<:Int}
Structure that is used in Ensemble Kalman Inversion (EKI)
#Fields
$(DocStringExtensions.FIELDS)
"""
struct UKIObj{FT<:AbstractFloat, IT<:Int}
    "vector of parameter names"
     unames::Vector{String}
    "a vector of arrays of size N_ensemble x N_parameters containing the parameters (in each EKI iteration a new array of parameters is added)"
     θ_bar::Vector{Array{FT}}
     "a vector of arrays of size N_ensemble x N_parameters containing the parameters (in each EKI iteration a new array of parameters is added)"
     θθ_cov::Vector{Array{FT, 2}}
     "vector of observations (length: N_data); mean of all observation samples"
     g_t::Vector{FT}
     "covariance of the observational noise, which is assumed to be normally distributed"
     obs_cov::Array{FT, 2}
     "ensemble size"
     N_ens::IT
     "g = G(u), z = [u, g]"
     N_θ::IT
     N_g::IT
     "weights"
     sample_weights::Array{FT}
     μ_weights::Array{FT} 
     cov_weights::Array{FT} 
end



# outer constructors
function UKIObj(parameter_names::Vector{String},
                θ0_bar::Array{FT}, 
                θθ0_cov::Array{FT, 2},
                g_t::Vector{FT}, # observation
                obs_cov::Array{FT, 2}) where {FT<:AbstractFloat}

    # ensemble size
    N_θ = size(θ0_bar,1)
    N_g = size(g_t, 1)

    N_ens = 2*N_θ+1
    IT = typeof(N_ens)

 
    sample_weights = zeros(FT, N_θ)
    μ_weights = zeros(FT, N_ens)
    cov_weights = zeros(FT, N_ens)

    # todo parameters λ, α, β
    # α, β = 1.0e-3, 2.0
    α, β = 1.0, 2.0
    κ = 0.0
    λ = α^2*(N_θ + κ) - N_θ

    θ_bar = Array{FT}[]  # array of Array{FT, 2}'s
    push!(θ_bar, θ0_bar) # insert parameters at end of array (in this case just 1st entry)
    θθ_cov = Array{FT,2}[] # array of Array{FT, 2}'s
    push!(θθ_cov, θθ0_cov) # insert parameters at end of array (in this case just 1st entry)
   

    sample_weights[1:N_θ]     .=  sqrt(N_θ + λ)

    μ_weights[1] = λ/(N_θ + λ)
    μ_weights[2:N_ens] .= 1/(2(N_θ + λ))

    cov_weights[1] = λ/(N_θ + λ) + 1 - α^2 + β
    cov_weights[2:N_ens] .= 1/(2(N_θ + λ))

    #error(">_<")

    UKIObj{FT,IT}(parameter_names, θ_bar, θθ_cov, g_t, obs_cov, N_ens, N_θ, N_g, 
                  sample_weights, μ_weights, cov_weights)

end


"""
    construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
Construct the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_sigma_ensemble(uki::UKIObj{FT}, x_bar::Array{FT}, x_cov::Array{FT,2}) where {FT<:AbstractFloat}
    N_ens = uki.N_ens
    N_x = size(x_bar,1)
    @assert(N_ens == 2*N_x+1)

    sample_weights = uki.sample_weights


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
    construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
Construct the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_mean(uki::UKIObj{FT}, x::Array{FT,2}) where {FT<:AbstractFloat}
    N_ens, N_x = size(x)

    @assert(uki.N_ens == N_ens)

    x_bar = zeros(Float64, N_x)

    μ_weights = uki.μ_weights

    
    for i = 1: N_ens
        x_bar += μ_weights[i]*x[i,:]
    end

    return x_bar
end

"""
    construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
Construct the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_cov(uki::UKIObj{FT}, x::Array{FT,2}, x_bar::Array{FT}) where {FT<:AbstractFloat}
    N_ens, N_x = uki.N_ens, size(x_bar,1)
    
    cov_weights = uki.cov_weights

    xx_cov = zeros(FT, N_x, N_x)

    for i = 1: N_ens
        xx_cov .+= cov_weights[i]*(x[i,:] - x_bar)*(x[i,:] - x_bar)'
    end

    return xx_cov
end

"""
    construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
Construct the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_cov(uki::UKIObj{FT}, x::Array{FT,2}, x_bar::Array{FT}, y::Array{FT,2}, y_bar::Array{FT}) where {FT<:AbstractFloat}
    N_ens, N_x, N_y = uki.N_ens, size(x_bar,1), size(y_bar,1)
    
    cov_weights = uki.cov_weights

    xy_cov = zeros(FT, N_x, N_y)

    for i = 1: N_ens
        xy_cov .+= cov_weights[i]*(x[i,:] - x_bar)*(y[i,:] - y_bar)'
    end

    return xy_cov
end



function reset_θθ0_cov!(uki::UKIObj)
    uki.θθ_cov[1] = copy(uki.θθ_cov[end])
end


function update_ensemble!(uki::UKIObj{FT}, ens_func::Function) where {FT}
    
    θ_bar  = copy(uki.θ_bar[end])
    θθ_cov = copy(uki.θθ_cov[end])
    ############# Prediction 
    # Generate sigma points, and time step update 
    
    
    θ_p_bar  = θ_bar 
    θθ_p_cov = θθ_cov + uki.θθ_cov[1]
    ############# Update
    # Generate sigma points
    N_θ, N_g, N_ens = uki.N_θ, uki.N_g, uki.N_ens
    θ_p = construct_sigma_ensemble(uki, θ_p_bar, θθ_p_cov)
    θ_p_bar  = construct_mean(uki, θ_p)
    θθ_p_cov = construct_cov(uki, θ_p, θ_p_bar)

    @info "min θ_p", minimum(θ_p)
    g = zeros(FT, N_ens, N_g)
    g .= ens_func(θ_p)
    g_bar = construct_mean(uki, g)

    
    gg_cov = construct_cov(uki, g, g_bar) + 2uki.obs_cov
    θg_cov = construct_cov(uki, θ_p, θ_p_bar, g, g_bar)

    tmp = θg_cov/gg_cov
    θ_bar =  θ_p_bar + tmp*(uki.g_t - g_bar)

    @info "norm(uki.g_t - g_bar)", norm(uki.g_t - g_bar), "/", norm(uki.g_t)
    @info "norm(θθ_cov)", norm(θθ_cov)
    
    θθ_cov =  θθ_p_cov - tmp*θg_cov' 


    # store new parameters (and observations)
    push!(uki.θ_bar, θ_bar) # N_ens x N_params
    push!(uki.θθ_cov, θθ_cov) # N_ens x N_data

end


