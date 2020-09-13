using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions


"""
GAUKIObj{FT<:AbstractFloat, IT<:Int}
Struct that is used in Unscented Kalman Inversion (EKI)
"""
mutable struct GAUKIObj{FT<:AbstractFloat, IT<:Int}
    "vector of parameter names (never used)"
     unames::Vector{String}
    "a vector of arrays of size N_ensemble x N_parameters containing the mean of the parameters (in each GAUKI iteration a new array of mean is added)"
     θ_bar::Vector{Array{FT}}
     "a vector of arrays of size N_ensemble x (N_parameters x N_parameters) containing the covariance of the parameters (in each GAUKI iteration a new array of cov is added)"
     θθ_cov::Vector{Array{FT, 2}}
     "vector of observations (length: N_data); mean of all observation samples"
     g_t::Vector{FT}
     "covariance of the observational noise"
     obs_cov::Array{FT, 2}
     "a vector of arrays of size N_ensemble x N_g containing the predicted observation (in each GAUKI iteration a new array of predicted observation is added)"
     g_bar::Vector{Array{FT}}
     "ensemble size"
     N_ens::IT
     "parameter size g = G(θ)"
     N_θ::IT
     "observation size g = G(θ)"
     N_g::IT
     "weights in GAUKI"
     sample_weights::Array{FT}
     μ_weights::Array{FT} 
     cov_weights::Array{FT} 
     Δt0::FT
     t::FT

end



"""
GAUKIObj Constructor 
parameter_names::Vector{String} : parameter name vector
θ0_bar::Array{FT} : prior mean
θθ0_cov::Array{FT, 2} : prior covariance
g_t::Vector{FT} : observation 
obs_cov::Array{FT, 2} : observation covariance
α_reg::Float64 : regularization parameter toward θ0 (0 < α_reg <= 1), default should be 1, without regulariazion
"""
function GAUKIObj(parameter_names::Vector{String},
                θ0_bar::Array{FT}, 
                θθ0_cov::Array{FT, 2},
                g_t::Vector{FT}, # observation
                obs_cov::Array{FT, 2},
                Δt0::Float64) where {FT<:AbstractFloat}

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
    κ = 0.0
    β = 2.0

    α = min(sqrt(4/(N_θ + κ)), 1.0)
    
    λ = α^2*(N_θ + κ) - N_θ

    

    θ_bar = Array{FT}[]  # array of Array{FT, 2}'s
    push!(θ_bar, θ0_bar) # insert parameters at end of array (in this case just 1st entry)
    θθ_cov = Array{FT,2}[] # array of Array{FT, 2}'s
    push!(θθ_cov, θθ0_cov) # insert parameters at end of array (in this case just 1st entry)

    g_bar = Array{FT}[]  # array of Array{FT, 2}'s
   

    sample_weights[1:N_θ]     .=  sqrt(N_θ + λ)

    μ_weights[1] = λ/(N_θ + λ)
    μ_weights[2:N_ens] .= 1/(2(N_θ + λ))

    cov_weights[1] = λ/(N_θ + λ) + 1 - α^2 + β
    cov_weights[2:N_ens] .= 1/(2(N_θ + λ))


    GAUKIObj{FT,IT}(parameter_names, θ_bar, θθ_cov, g_t, obs_cov, g_bar, N_ens, N_θ, N_g, 
                  sample_weights, μ_weights, cov_weights, Δt0, 0.0)

end


"""
construct_sigma_ensemble
Construct the sigma ensemble, based on the mean x_bar, and covariance x_cov
"""
function construct_sigma_ensemble(gauki::GAUKIObj{FT}, x_bar::Array{FT}, x_cov::Array{FT,2}) where {FT<:AbstractFloat}
    N_ens = gauki.N_ens
    N_x = size(x_bar,1)
    @assert(N_ens == 2*N_x+1)

    sample_weights = gauki.sample_weights


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
function construct_mean(gauki::GAUKIObj{FT}, x::Array{FT,2}) where {FT<:AbstractFloat}
    N_ens, N_x = size(x)

    @assert(gauki.N_ens == N_ens)

    x_bar = zeros(Float64, N_x)

    μ_weights = gauki.μ_weights

    
    for i = 1: N_ens
        x_bar += μ_weights[i]*x[i,:]
    end

    return x_bar
end

"""
construct_cov xx_cov from ensemble x and mean x_bar
"""
function construct_cov(gauki::GAUKIObj{FT}, x::Array{FT,2}, x_bar::Array{FT}) where {FT<:AbstractFloat}
    N_ens, N_x = gauki.N_ens, size(x_bar,1)
    
    cov_weights = gauki.cov_weights

    xx_cov = zeros(FT, N_x, N_x)

    for i = 1: N_ens
        xx_cov .+= cov_weights[i]*(x[i,:] - x_bar)*(x[i,:] - x_bar)'
    end

    return xx_cov
end

"""
construct_cov xy_cov from ensemble x and mean x_bar, ensemble y and mean y_bar
"""
function construct_cov(gauki::GAUKIObj{FT}, x::Array{FT,2}, x_bar::Array{FT}, y::Array{FT,2}, y_bar::Array{FT}) where {FT<:AbstractFloat}
    N_ens, N_x, N_y = gauki.N_ens, size(x_bar,1), size(y_bar,1)
    
    cov_weights = gauki.cov_weights

    xy_cov = zeros(FT, N_x, N_y)

    for i = 1: N_ens
        xy_cov .+= cov_weights[i]*(x[i,:] - x_bar)*(y[i,:] - y_bar)'
    end

    return xy_cov
end



function compute_Φ!(gauki::GAUKIObj{FT}, g::Array{Float64, 2}) where {FT}
    N_ens = size(g, 1)
    Φ = zeros(FT, N_ens)
    for i = 1:N_ens
        Φ[i] = 0.5 * (gauki.g_t - g[i, :])' * (gauki.obs_cov\(gauki.g_t - g[i, :]))
    end
    return Φ
end

"""
update GAUKI struct
ens_func: The function g = G(θ)
define the function as 
    ens_func(θ_ens) = MyG(phys_params, θ_ens, other_params)
"""
function update_ensemble!(gauki::GAUKIObj{FT}, ens_func::Function) where {FT}
    
    N_ens, N_g  = gauki.N_ens, gauki.N_g

    θ_bar  = copy(gauki.θ_bar[end])
    θθ_cov = copy(gauki.θθ_cov[end])
    ############# Prediction 
    # Generate sigma points, and time step update 
    
   
    

    θ_p = construct_sigma_ensemble(gauki, θ_bar, θθ_cov)
    
    g = zeros(FT, N_ens, N_g)
    g .= ens_func(θ_p)
    g_bar = construct_mean(gauki, g)
    Φ = compute_Φ!(gauki, g)
    
    μ_weights = gauki.μ_weights
    dθ_bar = similar(θ_bar);    dθ_bar  .= 0.0
    dθθ_cov = similar(θθ_cov);  dθθ_cov .= 0.0
    Φ_bar  = 0.0

    for i = 1: N_ens
        Φ_bar    += μ_weights[i] * Φ[i]
    end

    K = Φ_bar

    dt = min(1.0 - gauki.t, gauki.Δt0/maximum(abs.(Φ .- K)))

    for i = 1: N_ens
        dθ_bar  .+= μ_weights[i] * (Φ[i] - K) * θ_p[i, :]
    end
    

    for i = 1: N_ens
        dθθ_cov .+= μ_weights[i] * (Φ[i] - K) * (θ_bar - θ_p[i, :]) * (θ_bar - θ_p[i, :])'
    end

    θ_bar  .=  (θ_bar - dt * dθ_bar) /(1 - dt*(Φ_bar - K))
    θθ_cov .=  (θθ_cov - dt * dθθ_cov)/(1 - dt*(Φ_bar - K))
   

    @info "maximum(abs.(Φ .- K)) : ",maximum(abs.(Φ .- K))
    @info "θ_bar = ", θ_bar
    @info "dθ_bar = ", dθ_bar
    @info "θθ_cov = ", θθ_cov
    @info "dθθ_cov = ", dθθ_cov

    
    
    

    # store new parameters (and observations)
    push!(gauki.g_bar, g_bar) # N_ens x N_data
    push!(gauki.θ_bar, θ_bar) # N_ens x N_params
    push!(gauki.θθ_cov, θθ_cov) # N_ens x N_data

    gauki.t += dt

end


