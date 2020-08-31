using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions


"""
EnKIObj{FT<:AbstractFloat, IT<:Int}
Structure that is used in Ensemble Kalman Inversion (EnKI)
#Fields
$(DocStringExtensions.FIELDS)
"""
struct EnKIObj{FT<:AbstractFloat, IT<:Int}
    "filter_type type"
    filter_type::String
    "vector of parameter names"
    unames::Vector{String}
    "a vector of arrays of size N_ensemble x N_parameters containing the parameters (in each EnKI iteration a new array of parameters is added)"
    θ::Vector{Array{FT, 2}}
    "Prior mean"
    θ0_bar::Array{FT}
    "Prior convariance"
    θθ0_cov::Array{FT, 2}
    "observation"
    g_t::Vector{FT}
    "covariance of the observational noise, which is assumed to be normally distributed"
    obs_cov::Array{FT, 2}
    "predicted g_bar"   
    g_bar::Vector{Array{FT}}
    "ensemble size"
    N_ens::IT
    "function size"
    N_g::IT
    "regularization parameter"
    α_reg::Float64
    "Covariance matrix of the evolution error"
    Σ_ω::Array{FT, 2}
    "Covariance matrix of the observation error"
    Σ_ν::Array{FT, 2} 
end

# outer constructors
function EnKIObj(filter_type::String,
    parameter_names::Vector{String},
    θ0::Array{FT, 2},
    θ0_bar::Array{FT},
    θθ0_cov::Array{FT, 2},
    g_t,
    obs_cov::Array{FT, 2},
    α_reg::Float64) where {FT<:AbstractFloat}
    
    # ensemble size
    N_ens = size(θ0)[1]
    IT = typeof(N_ens)
    # parameters
    θ = Array{FT, 2}[] # array of Array{FT, 2}'s
    push!(θ, θ0) # insert parameters at end of array (in this case just 1st entry)
    
    g_bar = Array{FT}[]  # array of Array{FT, 2}'s
    # observations
    g = Vector{FT}[]
    N_g = size(g_t, 1)


    Σ_ω, Σ_ν =  (2-α_reg^2)*θθ0_cov, 2*obs_cov
    # Σ_ω, Σ_ν =  1.0e-15*θθ0_cov, obs_cov
    
    EnKIObj{FT,IT}(filter_type, parameter_names, θ, θ0_bar, θθ0_cov,  g_t, obs_cov, g_bar, N_ens, N_g, α_reg, Σ_ω, Σ_ν)
end


"""
construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
Construct the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
    N_θ = length(priors)
    params = zeros(N_ens, N_θ)
    # Ensuring reproducibility of the sampled parameter values
    Random.seed!(rng_seed)
    for i in 1:N_θ
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
function construct_cov(enki::EnKIObj{FT}, x::Array{FT,2}, x_bar::Array{FT}, y::Array{FT,2}, y_bar::Array{FT}) where {FT<:AbstractFloat}
    N_ens, N_x, N_y = enki.N_ens, size(x_bar,1), size(y_bar,1)
    
    xy_cov = zeros(FT, N_x, N_y)
    
    for i = 1: N_ens
        xy_cov .+= (x[i,:] - x_bar)*(y[i,:] - y_bar)'
    end
    
    return xy_cov/(N_ens - 1)
end


function trunc_svd(X, ϵ = 1.0e-6)
    n_row, n_col = size(X)
    svd_X = svd(X)
    rank_X = min(n_row, n_col)
    for i = 1:min(n_row, n_col)
        if svd_X.S[i] <= ϵ*svd_X.S[1]
            rank_X = i
            break
        end
    end
    return svd_X.U[:, 1:rank_X], svd_X.S[1:rank_X], svd_X.Vt[1:rank_X, :]'
end

function update_ensemble!(enki::EnKIObj{FT}, ens_func::Function) where {FT<:AbstractFloat}
    # θ: N_ens x N_θ
    N_ens, N_θ = size(enki.θ[1])
    N_g = enki.N_g
    ############# Prediction 
    θ_p = copy(enki.θ[end])
    θ0_bar = enki.θ0_bar
    α_reg = enki.α_reg
    
    # evolution and observation covariance matrices
    Σ_ω, Σ_ν = enki.Σ_ω, enki.Σ_ν
    
    # generate evolution error
    noise = rand(MvNormal(zeros(N_θ), Σ_ω), N_ens) # N_ens
    
    for j = 1:N_ens
        θ_p[j, :] .= α_reg*θ_p[j, :] + (1-α_reg)*θ0_bar + noise[:, j]
    end
    
    θ_p_bar = dropdims(mean(θ_p, dims=1), dims=1)
    
    ############# Update and Analysis
    
    g = zeros(FT, N_ens, N_g)
    g .= ens_func(θ_p)
    
    
    g_bar = dropdims(mean(g, dims=1), dims=1)
    
    gg_cov = construct_cov(enki, g, g_bar, g, g_bar) + Σ_ν
    θg_cov = construct_cov(enki, θ_p, θ_p_bar, g, g_bar)

    
    
    
    # Kalman gain
    K = θg_cov/gg_cov
    θ_bar = θ_p_bar + K*(enki.g_t - g_bar)

    filter_type = enki.filter_type
    
    if filter_type == "EnKI"
        y = zeros(FT, N_ens, N_g)
        noise = rand(MvNormal(zeros(N_g), Σ_ν), N_ens) # N_ens
        for j = 1:N_ens
            y[j, :] = g[j, :] + noise[:, j]
        end
        
        θ = copy(θ_p) 
        for j = 1:N_ens
            θ[j,:] += K*(enki.g_t - y[j, :]) # N_ens x N_θ
        end
        
    elseif filter_type == "EAKI"
        

        Z_p_t = copy(θ_p)
        for j = 1:N_ens;    Z_p_t[j, :] .-=  θ_p_bar;    end
        Z_p_t ./= sqrt(N_ens - 1)





        
        # update particles by Ensemble Adjustment Kalman Filter
        # Dp = F^T Σ F, Σ = F Dp F^T, Dp is non-singular

        F, sqrt_D_p, V =  trunc_svd(Z_p_t') 

        GZ_p_t = copy(g)  
        for j = 1:N_ens;     GZ_p_t[j, :] .-=  g_bar;    end
        GZ_p_t ./= sqrt(N_ens - 1)


        
        X = V' /(I + GZ_p_t/Σ_ν*GZ_p_t') * V
        svd_X = svd(X)

        U, D = svd_X.U, svd_X.S


        A = (F .* sqrt_D_p' * U .* sqrt.(D)') * (sqrt_D_p .\ F')
        
        
        θ = similar(θ_p) 
        for j = 1:N_ens
            θ[j, :] .= θ_bar + A * (θ_p[j, :] - θ_p_bar) # N_ens x N_θ
        end

        ################# Debug check

        @info "norm(A) ", norm(A)

        θθ_cov = Z_p_t'*(I - GZ_p_t/(GZ_p_t'*GZ_p_t + Σ_ν)*GZ_p_t') *Z_p_t

        @info "norm(Z_p_t) ", norm(Z_p_t), "norm(θθ_cov)", norm(θθ_cov)

        
        # Debug 
        # θθ_p_cov = construct_cov(enki, θ_p, θ_p_bar, θ_p, θ_p_bar)
        # θθ_cov = Z_p_t'*(I - GZ_p_t/(GZ_p_t'*GZ_p_t + Σ_ν)*GZ_p_t') *Z_p_t
        # θ_bar_debug = dropdims(mean(θ, dims=1), dims=1)
        # θθ_cov_debug = construct_cov(enki, θ, θ_bar_debug, θ, θ_bar_debug)
        # @info "mean error is ", norm(θ_bar - θ_bar_debug), " cov error is ", norm(θθ_cov - A*Z_p_t'*Z_p_t*A'), norm(θθ_cov - θθ_cov_debug)
     
        
    elseif filter_type == "ETKI"
        
        # Compute 
        GZ_p_t = copy(g)  
        for j = 1:N_ens;  GZ_p_t[j, :] .-=  g_bar;  end
        GZ_p_t ./= sqrt(N_ens - 1)

        
        # update particles by Ensemble Adjustment Kalman Filter
        # Dp = F^T Σ F, Σ = F Dp F^T, Dp is non-singular
        X = GZ_p_t/Σ_ν*GZ_p_t'
        svd_X = svd(X)

        C, Γ = svd_X.U, svd_X.S
        
        #Original ETKF is  T = C * (Γ .+ 1)^{-1/2}, but it is biased
        T = C ./ sqrt.(Γ .+ 1)' * C'

        # Z_p'
        θ = similar(θ_p) 
        for j = 1:N_ens;  θ[j, :] .=  θ_p[j, :] - θ_p_bar;  end
        # Z' = （Z_p * T)' = T' * Z_p
        θ .= T' * θ 
        for j = 1:N_ens;  θ[j, :] .+=  θ_bar;  end


        ################# Debug check

        # Debug
        # Z_p_t = copy(θ_p)
        # for j = 1:N_ens;    Z_p_t[j, :] .-=  θ_p_bar;    end
        # Z_p_t ./= sqrt(N_ens - 1)

        
        # θθ_p_cov = construct_cov(enki, θ_p, θ_p_bar, θ_p, θ_p_bar)
        # θθ_cov = Z_p_t'*(I - GZ_p_t/(GZ_p_t'*GZ_p_t + Σ_ν)*GZ_p_t') *Z_p_t
        # θ_bar_debug = dropdims(mean(θ, dims=1), dims=1)
        # θθ_cov_debug = construct_cov(enki, θ, θ_bar_debug, θ, θ_bar_debug)
        # @info "mean error is ", norm(θ_bar - θ_bar_debug), " cov error is ", norm(θθ_cov - Z_p_t'*T*T'*Z_p_t), norm(θθ_cov - θθ_cov_debug)
     
        
    else
        error("Filter type :", filter_type, " has not implemented yet!")
    end
    
    
    
    # store new parameters (and observations)
    push!(enki.θ, θ) # N_ens x N_θ
    push!(enki.g_bar, g_bar)
    
end


