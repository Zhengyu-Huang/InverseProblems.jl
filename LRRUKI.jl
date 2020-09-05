using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions


"""
LRRUKIObj{FT<:AbstractFloat, IT<:Int}
Struct that is used in Unscented Kalman Inversion (EKI)
"""
struct LRRUKIObj{FT<:AbstractFloat, IT<:Int}
    "vector of parameter names (never used)"
     unames::Vector{String}
    "a vector of arrays of size N_ensemble x N_parameters containing the mean of the parameters (in each LRRUKI iteration a new array of mean is added)"
     θ_bar::Vector{Array{FT}}
     "a vector of arrays of size N_ensemble x (N_parameters x N_parameters) containing the covariance of the parameters (in each LRRUKI iteration a new array of cov is added)"
     θθ_cov::Vector{Array{FT, 2}}
     "vector of observations (length: N_data); mean of all observation samples"
     g_t::Vector{FT}
     "covariance of the observational noise"
     obs_cov::Array{FT, 2}
     "a vector of arrays of size N_ensemble x N_g containing the predicted observation (in each LRRUKI iteration a new array of predicted observation is added)"
     g_bar::Vector{Array{FT}}
     "low rank dimension, ensemble size is 2N_r+1 "
     N_r::IT
     "parameter size g = G(θ)"
     N_θ::IT
     "observation size g = G(θ)"
     N_g::IT
     "weights in LRRUKI"
     sample_weights::Array{FT}
     μ_weights::Array{FT} 
     cov_weights::Array{FT} 
     "regularization parameter"
     α_reg::Float64
     "Covariance matrix of the evolution error"
     Σ_ω::Array{FT, 2}
     "Covariance matrix of the observation error"
     Σ_ν::Array{FT, 2} 
end



"""
LRRUKIObj Constructor 
parameter_names::Vector{String} : parameter name vector
θ0_bar::Array{FT} : prior mean
θθ0_cov::Array{FT, 2} : prior covariance
g_t::Vector{FT} : observation 
obs_cov::Array{FT, 2} : observation covariance
α_reg::Float64 : regularization parameter toward θ0 (0 < α_reg <= 1), default should be 1, without regulariazion
"""
function LRRUKIObj(parameter_names::Vector{String},
                N_r::Int64,
                θ0_bar::Array{FT}, 
                θθ0_cov::Array{FT, 2},
                g_t::Vector{FT}, # observation
                obs_cov::Array{FT, 2},
                α_reg::Float64) where {FT<:AbstractFloat}

    # ensemble size
    N_θ = size(θ0_bar,1)
    N_g = size(g_t, 1)

    N_r = min(N_r, N_θ)
    N_ens = 2*N_r+1
    IT = typeof(N_ens)

 
    sample_weights = zeros(FT, N_θ)
    μ_weights = zeros(FT, N_ens)
    cov_weights = zeros(FT, N_ens)

    θ_bar = Array{FT}[]  # array of Array{FT, 2}'s
    push!(θ_bar, θ0_bar) # insert parameters at end of array (in this case just 1st entry)
    θθ_cov = Array{FT,2}[] # array of Array{FT, 2}'s
    push!(θθ_cov, θθ0_cov) # insert parameters at end of array (in this case just 1st entry)

    g_bar = Array{FT}[]   # array of Array{FT, 2}'s



    # TODO parameters λ, α, β
    # α, β = 1.0e-3, 2.0
    κ = 0.0
    β = 2.0

    α = min(sqrt(4/(N_r + κ)), 1.0)
    
    λ = α^2*(N_r + κ) - N_r

    sample_weights[1:N_r]     .=  sqrt(N_r + λ)

    μ_weights[1] = λ/(N_r + λ)
    μ_weights[2:N_ens] .= 1/(2(N_r + λ))

    cov_weights[1] = λ/(N_r + λ) + 1 - α^2 + β
    cov_weights[2:N_ens] .= 1/(2(N_r + λ))

    Σ_ω, Σ_ν =  (2-α_reg^2)*θθ0_cov, 2*obs_cov
    

    LRRUKIObj{FT,IT}(parameter_names, θ_bar, θθ_cov, g_t, obs_cov, g_bar, N_r, N_θ, N_g, 
                  sample_weights, μ_weights, cov_weights, α_reg, Σ_ω, Σ_ν)

end


"""
construct_sigma_ensemble
Construct the sigma ensemble, based on the mean x_bar, and covariance x_cov
"""
function construct_sigma_ensemble(lrruki::LRRUKIObj{FT}, x_bar::Array{FT}, x_cov::Array{FT,2}) where {FT<:AbstractFloat}
    N_r = lrruki.N_r
    N_x = size(x_bar, 1)
    sample_weights = lrruki.sample_weights

    
    # use svd decomposition
    svd_xx_cov = svd(x_cov)

    # @info "energy : ", sum(svd_xx_cov.S[1:N_r])/sum(svd_xx_cov.S)
    # @info "svd_xx_cov.S : ", svd_xx_cov.S

    # @info "svd_xx_cov.U : ", svd_xx_cov.U
    # @info "x_cov : ", x_cov

    x = zeros(Float64, 2*N_r+1, N_x)
    x[1, :] = x_bar
    for i = 1: N_r
        x[i+1,     :] = x_bar + sample_weights[i]*sqrt(svd_xx_cov.S[i])*svd_xx_cov.U[:, i]
        x[i+1+N_r, :] = x_bar - sample_weights[i]*sqrt(svd_xx_cov.S[i])*svd_xx_cov.U[:, i]
    end



    # chol_xx_cov = cholesky(Hermitian(x_cov)).L
    # x = zeros(Float64, 2*N_r+1, N_x)
    # x[1, :] = x_bar
    # for i = 1: N_r
    #     x[i+1,     :] = x_bar + sample_weights[i]*chol_xx_cov[:,i]
    #     x[i+1+N_r, :] = x_bar - sample_weights[i]*chol_xx_cov[:,i]
    # end

    return x
end


"""
construct_mean x_bar from ensemble x
"""
function construct_mean(lrruki::LRRUKIObj{FT}, x::Array{FT,2}) where {FT<:AbstractFloat}
    N_ens, N_x = size(x)

    @assert(2*lrruki.N_r + 1 == N_ens)

    x_bar = zeros(Float64, N_x)

    μ_weights = lrruki.μ_weights

    
    for i = 1: N_ens
        x_bar += μ_weights[i]*x[i,:]
    end

    return x_bar
end

"""
construct_cov xx_cov from ensemble x and mean x_bar
"""
function construct_cov(lrruki::LRRUKIObj{FT}, x::Array{FT,2}, x_bar::Array{FT}) where {FT<:AbstractFloat}
    N_ens, N_x = 2*lrruki.N_r + 1, size(x_bar,1)
    
    cov_weights = lrruki.cov_weights

    xx_cov = zeros(FT, N_x, N_x)

    for i = 1: N_ens
        xx_cov .+= cov_weights[i]*(x[i,:] - x_bar)*(x[i,:] - x_bar)'
    end

    return xx_cov
end

"""
construct_cov xy_cov from ensemble x and mean x_bar, ensemble y and mean y_bar
"""
function construct_cov(lrruki::LRRUKIObj{FT}, x::Array{FT,2}, x_bar::Array{FT}, y::Array{FT,2}, y_bar::Array{FT}) where {FT<:AbstractFloat}
    N_ens, N_x, N_y = 2*lrruki.N_r + 1, size(x_bar,1), size(y_bar,1)
    
    cov_weights = lrruki.cov_weights

    xy_cov = zeros(FT, N_x, N_y)

    for i = 1: N_ens
        xy_cov .+= cov_weights[i]*(x[i,:] - x_bar)*(y[i,:] - y_bar)'
    end

    return xy_cov
end



"""
update LRRUKI struct
ens_func: The function g = G(θ)
define the function as 
    ens_func(θ_ens) = MyG(phys_params, θ_ens, other_params)
use G(θ_bar) instead of FG(θ)
"""
function update_ensemble!(lrruki::LRRUKIObj{FT}, ens_func::Function) where {FT}
    
    θ_bar  = copy(lrruki.θ_bar[end])
    θθ_cov = copy(lrruki.θθ_cov[end])

    # @info "θθ_cov ", diag(θθ_cov)
    ############# Prediction 
    # Generate sigma points, and time step update 
    
    α_reg = lrruki.α_reg
    Σ_ω, Σ_ν = lrruki.Σ_ω, lrruki.Σ_ν
    

    θ_p_bar  = α_reg*θ_bar + (1-α_reg)*lrruki.θ_bar[1]
    θθ_p_cov = α_reg^2*θθ_cov + Σ_ω
    


    ############# Update
    # Generate sigma points
    N_θ, N_g, N_ens = lrruki.N_θ, lrruki.N_g, 2*lrruki.N_r+1
    θ_p = construct_sigma_ensemble(lrruki, θ_p_bar, θθ_p_cov)
    θ_p_bar_  = construct_mean(lrruki, θ_p)
    θθ_p_cov = construct_cov(lrruki, θ_p, θ_p_bar)
    #@show  norm(θ_p_bar_ - θ_p_bar), norm(θθ_p_cov - θθ_p_cov_)
    #@info "θθ_p_cov_: ", diag(θθ_p_cov_)
    
    g = zeros(FT, N_ens, N_g)
    g .= ens_func(θ_p)
    g_bar = g[1,:] #construct_mean(lrruki, g)

    # @info "θ_p: ", θ_p

    
    gg_cov = construct_cov(lrruki, g, g_bar) + Σ_ν
    θg_cov = construct_cov(lrruki, θ_p, θ_p_bar, g, g_bar)

    #Kalman Gain
    K = θg_cov/gg_cov
    # use G(θ_bar) instead of FG(θ)
    θ_bar =  θ_p_bar + K*(lrruki.g_t - g[1,:])

    # K = θθ'G' (Gθθ'G' + Σ_ν)G(θ_ref - θ)
    @info norm(lrruki.g_t - g[1,:]), norm(K*(lrruki.g_t - g[1,:]))
    
    θθ_cov =  θθ_p_cov - K*θg_cov' 

    # @info " gg_cov ", diag(gg_cov)
    # @info " θg_cov ", diag(θg_cov)
    # @info "θ_bar : ", θ_bar
    # @info "θθ_p_cov : ", diag(θθ_p_cov)
    # @info "K*θg_cov' : ", diag(K*θg_cov')
    # @info "θθ_cov' ", θθ_cov # diag(θθ_cov')


    # @info " K*θg_cov' : ", K*θg_cov'

    # @info " θθ_p_cov : ", θθ_p_cov
    # @info " θθ_cov : ", θθ_cov

    # # Test
    # nθ = 40
    # G = zeros(nθ, nθ)
    #     for i = 1:nθ
    #         for j = 1:nθ
    #             if i == j
    #                 G[i,j] = 2
    #             elseif i == j-1 || i == j+1
    #                 G[i,j] = -1
    #             end
    #         end
    #     end
    # θθ_cov = inv(G'/Σ_ν *G)


    # store new parameters (and observations  G(θ_bar) instead of FG(θ))
    push!(lrruki.g_bar, g[1,:]) # N_ens x N_data
    push!(lrruki.θ_bar, θ_bar) # N_ens x N_params
    push!(lrruki.θθ_cov, θθ_cov) # N_ens x N_data

    

end


