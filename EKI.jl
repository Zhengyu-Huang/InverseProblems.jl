using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions


"""
EKIObj{FT<:AbstractFloat, IT<:Int}
Structure that is used in Ensemble Kalman Inversion (EKI)
#Fields
$(DocStringExtensions.FIELDS)
"""
struct EKIObj{FT<:AbstractFloat, IT<:Int}
    "a vector of arrays of size N_ensemble x N_parameters containing the parameters (in each EKI iteration a new array of parameters is added)"
     θ::Vector{Array{FT, 2}}
     "Prior convariance"
     θθ0_cov::Array{FT, 2}
     "vector of parameter names"
     unames::Vector{String}
     "vector of observations (length: N_data); mean of all observation samples"
     g_t::Vector{FT}
     "covariance of the observational noise, which is assumed to be normally distributed"
     obs_cov::Array{FT, 2}
     "ensemble size"
     N_ens::IT
     "function size"
     N_g::IT
end

# outer constructors
function EKIObj(parameter_names::Vector{String},
                θ0::Array{FT, 2},
                θθ0_cov::Array{FT, 2},
                g_t,
                obs_cov::Array{FT, 2}) where {FT<:AbstractFloat}

    # ensemble size
    N_ens = size(θ0)[1]
    IT = typeof(N_ens)
    # parameters
    θ = Array{FT, 2}[] # array of Array{FT, 2}'s
    push!(θ, θ0) # insert parameters at end of array (in this case just 1st entry)
    # observations
    g = Vector{FT}[]
    N_g = size(g_t, 1)

    EKIObj{FT,IT}(θ, θθ0_cov, parameter_names, g_t, obs_cov, N_ens, N_g)
end


"""
construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
Construct the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
    N_params = length(priors)
    params = zeros(N_ens, N_params)
    # Ensuring reproducibility of the sampled parameter values
    Random.seed!(rng_seed)
    for i in 1:N_params
        prior_i = priors[i]
        params[:, i] = rand(prior_i, N_ens)
    end

    return params
end

"""
    construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
Construct the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_cov(eki::EKIObj{FT}, x::Array{FT,2}, x_bar::Array{FT}, y::Array{FT,2}, y_bar::Array{FT}) where {FT<:AbstractFloat}
    N_ens, N_x, N_y = eki.N_ens, size(x_bar,1), size(y_bar,1)

    xy_cov = zeros(FT, N_x, N_y)

    for i = 1: N_ens
        xy_cov .+= (x[i,:] - x_bar)*(y[i,:] - y_bar)'
    end

    return xy_cov/(N_ens - 1)
end



function update_ensemble!(eki::EKIObj{FT}, ens_func::Function) where {FT}
    # θ: N_ens x N_params
    N_ens, N_params = size(eki.θ[1])
    N_g = eki.N_g
    ############# Prediction 
    θ_p = copy(eki.θ[end])
    noise = rand(MvNormal(zeros(N_params), eki.θθ0_cov), N_ens) # N_ens

    for j = 1:N_ens
        θ_p[j, :] += noise[:, j]
    end

    θ_p_bar = dropdims(mean(θ_p, dims=1), dims=1)

    ############# Update and Analysis
    
    g = zeros(FT, N_ens, N_g)
    g .= ens_func(θ_p)

    g_bar = dropdims(mean(g, dims=1), dims=1)

    gg_cov = construct_cov(eki, g, g_bar, g, g_bar) + 2eki.obs_cov
    θg_cov = construct_cov(eki, θ_p, θ_p_bar, g, g_bar)
    

    tmp = θg_cov/gg_cov


    y = zeros(FT, N_ens, N_g)
    noise = rand(MvNormal(zeros(N_g), 2*eki.obs_cov), N_ens) # N_ens
    for j = 1:N_ens
        y[j, :] = g[j, :] + noise[:, j]
    end
    
    θ = copy(θ_p) 
    for j = 1:N_ens
        θ[j,:] += tmp*(eki.g_t - y[j, :]) # N_ens x N_params
    end

    # store new parameters (and observations)
    push!(eki.θ, θ) # N_ens x N_params

end


