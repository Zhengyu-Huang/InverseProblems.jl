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
    θ::Array{FT, 2}
    "Prior mean"
    θ0_bar::Array{FT}
    "Prior convariance"
    θθ0_cov
    "observation"
    g_t::Vector{FT}
    "covariance of the observational noise, which is assumed to be normally distributed"
    obs_cov
    "ensemble size"
    N_ens::IT
    "function size"
    N_g::IT
    "regularization parameter"
    α_reg::Float64
    "Covariance matrix of the evolution error"
    Σ_ω
    "Covariance matrix of the observation error"
    Σ_ν
end

# outer constructors
function EnKIObj(filter_type::String,
    parameter_names::Vector{String},
    N_ens::Int64,
    θ0_bar::Array{FT},
    θθ0_cov,
    g_t,
    obs_cov,
    α_reg::Float64) where {FT<:AbstractFloat}
    

    IT = typeof(N_ens)

    # generate ensemble
    Random.seed!(123)
    θ0 = rand(MvNormal(θ0_bar, θθ0_cov), N_ens)'


    # use svd decomposition
    # svd_θθ0_cov = svd(θθ0_cov)
    # N_θ = size(θ0_bar, 1)
    # θ0 = zeros(Float64, N_ens, N_θ)
    # θ0[1, :] = θ0_bar
    # N_r = div(N_ens, 2)
    # for i = 1: N_r
    #     θ0[i+1,     :] = θ0_bar + sqrt(svd_θθ0_cov.S[i])*svd_θθ0_cov.U[:, i]
    #     θ0[i+1+N_r, :] = θ0_bar - sqrt(svd_θθ0_cov.S[i])*svd_θθ0_cov.U[:, i]
    # end




    θ = copy(θ0) # insert parameters at end of array (in this case just 1st entry)
    
    g_bar = Array{FT}[]  # array of Array{FT, 2}'s
    # observations
    g = Vector{FT}[]
    N_g = size(g_t, 1)

    γ = 2.0
    Σ_ω, Σ_ν =  (γ/(γ-1)-α_reg^2)*θθ0_cov, γ*obs_cov
    # Σ_ω, Σ_ν =  1.0e-15*θθ0_cov, obs_cov
    
    EnKIObj{FT,IT}(filter_type, parameter_names, θ, θ0_bar, θθ0_cov,  g_t, obs_cov, N_ens, N_g, α_reg, Σ_ω, Σ_ν)
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
    N_ens, N_θ = size(enki.θ)
    N_g = enki.N_g
    ############# Prediction 
    θ_p = copy(enki.θ)
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
    
    

    
    
    
    # Kalman gain
    Z_p_t = copy(θ_p)
    for j = 1:N_ens;    Z_p_t[j, :] .-=  θ_p_bar;    end
    Z_p_t ./= sqrt(N_ens - 1)


    V_p_t = copy(g)  
    for j = 1:N_ens;  V_p_t[j, :] .-=  g_bar;  end
    V_p_t ./= sqrt(N_ens - 1)

    #=
    gg_cov = construct_cov(enki, g, g_bar, g, g_bar) + Σ_ν
    θg_cov = construct_cov(enki, θ_p, θ_p_bar, g, g_bar)
    K = θg_cov * gg_cov⁻¹ 
      = Z_p * V_p' * (V_p *V_p' + Σ_ν)
      = Z_p * V_p' * (V_p *V_p' + Σ_ν)⁻¹ 
      = Z_p_t' * X⁻¹ * V_p_t * Σ_ν⁻¹ 
    =#                       
    X = V_p_t/Σ_ν*V_p_t'
    svd_X = svd(X)
    C, Γ = svd_X.U, svd_X.S


    
    θ_bar = θ_p_bar + Z_p_t' * (C *( Γ .\ (C' * (V_p_t * (Σ_ν\(enki.g_t - g_bar))))))

    filter_type = enki.filter_type
    
    if filter_type == "EnKI"
        y = zeros(FT, N_ens, N_g)
        noise = rand(MvNormal(zeros(N_g), Σ_ν), N_ens) # N_ens
        for j = 1:N_ens
            y[j, :] = g[j, :] + noise[:, j]
        end
        
        θ = copy(θ_p) 
        for j = 1:N_ens
            θ[j,:] += Z_p_t' * (C *( Γ .\ (C' * (V_p_t * (Σ_ν\((enki.g_t - y[j, :]))))))) # N_ens x N_θ
        end
        
    elseif filter_type == "EAKI"
        

        


        
        # update particles by Ensemble Adjustment Kalman Filter
        # Dp = F^T Σ F, Σ = F Dp F^T, Dp is non-singular

        F, sqrt_D_p, V =  trunc_svd(Z_p_t') 


        # I + V_p_t/Σ_ν*V_p_t' = C (Γ + I) C'
        # Y = V' /(I + V_p_t/Σ_ν*V_p_t') * V
        Y = V' * C ./ (Γ .+ 1.0)' * C' * V

        svd_Y = svd(Y)

        U, D = svd_Y.U, svd_Y.S


        A = (F .* sqrt_D_p' * U .* sqrt.(D)') * (sqrt_D_p .\ F')
        
        
        θ = similar(θ_p) 
        for j = 1:N_ens
            θ[j, :] .= θ_bar + A * (θ_p[j, :] - θ_p_bar) # N_ens x N_θ
        end

        ################# Debug check

        # θθ_p_cov = construct_cov(enki, θ_p, θ_p_bar, θ_p, θ_p_bar)
        # θθ_cov = Z_p_t'*(I - V_p_t/(V_p_t'*V_p_t + Σ_ν)*V_p_t') *Z_p_t
        # θ_bar_debug = dropdims(mean(θ, dims=1), dims=1)
        # θθ_cov_debug = construct_cov(enki, θ, θ_bar_debug, θ, θ_bar_debug)
        # @info "mean error is ", norm(θ_bar - θ_bar_debug), " cov error is ", norm(θθ_cov - A*Z_p_t'*Z_p_t*A'), norm(θθ_cov - θθ_cov_debug)
     
        
    elseif filter_type == "ETKI"
        
        
        # update particles by Ensemble Adjustment Kalman Filter
        # Dp = F^T Σ F, Σ = F Dp F^T, Dp is non-singular
        X = V_p_t/Σ_ν*V_p_t'
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

        # Z_p_t = copy(θ_p)
        # for j = 1:N_ens;    Z_p_t[j, :] .-=  θ_p_bar;    end
        # Z_p_t ./= sqrt(N_ens - 1)

        
        # θθ_p_cov = construct_cov(enki, θ_p, θ_p_bar, θ_p, θ_p_bar)
        # θθ_cov = Z_p_t'*(I - V_p_t/(V_p_t'*V_p_t + Σ_ν)*V_p_t') *Z_p_t
        # θ_bar_debug = dropdims(mean(θ, dims=1), dims=1)
        # θθ_cov_debug = construct_cov(enki, θ, θ_bar_debug, θ, θ_bar_debug)
        # @info "mean error is ", norm(θ_bar - θ_bar_debug), " cov error is ", norm(θθ_cov - Z_p_t'*T*T'*Z_p_t), norm(θθ_cov - θθ_cov_debug)
     
        
    else
        error("Filter type :", filter_type, " has not implemented yet!")
    end
    
    
    
    # store new parameters (and observations)
    enki.θ .= copy(θ) # N_ens x N_θ
    
    
end




"""
TRUKIObj{FT<:AbstractFloat, IT<:Int}
Struct that is used in Unscented Kalman Inversion (EKI)
"""
mutable struct TRUKIObj{FT<:AbstractFloat, IT<:Int}
    "vector of parameter names (never used)"
     unames::Vector{String}
     "a vector of arrays of size N_ensemble x N_parameters containing the mean of the parameters (in each TRUKI iteration a new array of mean is added)"
     θ0_bar::Array{FT}
     "a vector of arrays of size N_ensemble x (N_parameters x N_r) containing the Square root of the covariance of the parameters (in each TRUKI iteration a new array of cov is added)"
     θθ0_cov_sqr::Array{FT, 2}
    "a vector of arrays of size N_ensemble x N_parameters containing the mean of the parameters (in each TRUKI iteration a new array of mean is added)"
     θ_bar::Array{FT}
     "a vector of arrays of size N_ensemble x (N_parameters x N_r) containing the Square root of the covariance of the parameters (in each TRUKI iteration a new array of cov is added)"
     θθ_cov_sqr::Array{FT, 2}
     "vector of observations (length: N_data); mean of all observation samples"
     g_t::Vector{FT}
     "covariance of the observational noise"
     obs_cov
     "low rank dimension, ensemble size is 2N_r+1 "
     N_r::IT
     "parameter size g = G(θ)"
     N_θ::IT
     "observation size g = G(θ)"
     N_g::IT
     "weights in TRUKI"
     sample_weights::Array{FT}
     μ_weights::Array{FT} 
     cov_weights::Array{FT} 
     "regularization parameter"
     α_reg::Float64
     "Square root of the Covariance matrix of the evolution error"
     Z_ω::Array{FT, 2}
     "Covariance matrix of the observation error"
     Σ_ν
     "For low rank "
     counter::Int64
end



"""
TRUKIObj Constructor 
parameter_names::Vector{String} : parameter name vector
θ0_bar::Array{FT} : prior mean
θθ0_cov::Array{FT, 2} : prior covariance
g_t::Vector{FT} : observation 
obs_cov::Array{FT, 2} : observation covariance
α_reg::Float64 : regularization parameter toward θ0 (0 < α_reg <= 1), default should be 1, without regulariazion
"""
function TRUKIObj(parameter_names::Vector{String},
                N_r::Int64,
                θ0_bar::Array{FT}, 
                θθ0_cov_sqr::Array{FT, 2},
                g_t::Vector{FT}, # observation
                obs_cov,
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

   
    θ_bar = copy(θ0_bar) # insert parameters at end of array (in this case just 1st entry)

    θθ_cov_sqr = copy(θθ0_cov_sqr) # insert parameters at end of array (in this case just 1st entry)

    

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

    Z_ω, Σ_ν =  sqrt(2-α_reg^2)*θθ0_cov_sqr, 2*obs_cov
    
    counter = 0

    TRUKIObj{FT,IT}(parameter_names, θ0_bar, θθ0_cov_sqr,  θ_bar, θθ_cov_sqr, g_t, obs_cov,  N_r, N_θ, N_g, 
                  sample_weights, μ_weights, cov_weights, α_reg, Z_ω, Σ_ν, counter)

end


"""
construct_sigma_ensemble
Construct the sigma ensemble, based on the mean x_bar, and covariance x_cov
"""
function construct_sigma_ensemble(truki::TRUKIObj{FT}, x_bar::Array{FT}, x_cov_sqr::Array{FT,2}) where {FT<:AbstractFloat}
    N_r = truki.N_r
    N_x = size(x_bar, 1)
    sample_weights = truki.sample_weights

    
    # use svd decomposition
    svd_xx_cov_sqr = svd(x_cov_sqr)

    # @info "energy : ", sum(svd_xx_cov.S[1:N_r])/sum(svd_xx_cov.S)
    # @info "svd_xx_cov_sqr.S : ", svd_xx_cov_sqr.S

    # @info "svd_xx_cov.U : ", svd_xx_cov.U
    # @info "x_cov : ", x_cov

    x = zeros(Float64, 2*N_r+1, N_x)
    x[1, :] = x_bar
    for i = 1: N_r
        x[i+1,     :] = x_bar + sample_weights[i]*svd_xx_cov_sqr.S[i]*svd_xx_cov_sqr.U[:, i]
        x[i+1+N_r, :] = x_bar - sample_weights[i]*svd_xx_cov_sqr.S[i]*svd_xx_cov_sqr.U[:, i]
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
function construct_mean(truki::TRUKIObj{FT}, x::Array{FT,2}) where {FT<:AbstractFloat}
    N_ens, N_x = size(x)

    @assert(2*truki.N_r + 1 == N_ens)

    x_bar = zeros(Float64, N_x)

    μ_weights = truki.μ_weights

    
    for i = 1: N_ens
        x_bar += μ_weights[i]*x[i,:]
    end

    return x_bar
end

"""
construct_cov xx_cov from ensemble x and mean x_bar
"""
function construct_cov_sqr(truki::TRUKIObj{FT}, x::Array{FT,2}, x_bar::Array{FT}) where {FT<:AbstractFloat}
    N_r, N_x = truki.N_r, size(x_bar,1)
    
    cov_weights = truki.cov_weights

    xx_cov_sqr = zeros(FT, N_x, 2N_r)

    for i = 2: 2N_r+1
        xx_cov_sqr[:, i-1] .= sqrt(cov_weights[i])*(x[i,:] - x_bar)
        
    end

    return xx_cov_sqr
end





"""
update TRUKI struct
ens_func: The function g = G(θ)
define the function as 
    ens_func(θ_ens) = MyG(phys_params, θ_ens, other_params)
use G(θ_bar) instead of FG(θ)
"""
function update_ensemble!(truki::TRUKIObj{FT}, ens_func::Function) where {FT}
    
    
    θ_bar  = copy(truki.θ_bar)
    θθ_cov_sqr = copy(truki.θθ_cov_sqr)

    # @info "θθ_cov ", diag(θθ_cov)
    ############# Prediction 
    # Generate sigma points, and time step update 
    
    α_reg = truki.α_reg
    Z_ω, Σ_ν = truki.Z_ω, truki.Σ_ν
    

    θ_p_bar  = α_reg*θ_bar + (1-α_reg)*truki.θ0_bar

    nrank , N_r = size(Z_ω, 2), truki.N_r
    col_start = mod1(truki.counter*N_r + 1, nrank)

    if col_start + N_r-1 <= nrank
        θθ_p_cov_sqr = [α_reg^2*θθ_cov_sqr  Z_ω[:, col_start: col_start + N_r-1]]
    else
        θθ_p_cov_sqr = [α_reg^2*θθ_cov_sqr  Z_ω[:, col_start: nrank] Z_ω[:, 1:col_start+N_r-1-nrank]]
    end


    ############# Update
    # Generate sigma points
    N_θ, N_g, N_ens = truki.N_θ, truki.N_g, 2*truki.N_r+1
    θ_p = construct_sigma_ensemble(truki, θ_p_bar, θθ_p_cov_sqr)
    
    #@show  norm(θ_p_bar_ - θ_p_bar), norm(θθ_p_cov - θθ_p_cov_)
    #@info "θθ_p_cov_: ", diag(θθ_p_cov_)
    
    g = zeros(FT, N_ens, N_g)
    g .= ens_func(θ_p)
    g_bar = g[1,:] #construct_mean(truki, g)

    # @info "θ_p: ", θ_p

    Z = construct_cov_sqr(truki, θ_p, θ_p_bar)
    V = construct_cov_sqr(truki, g, g_bar)

    X = V' * (Σ_ν\V)
    svd_X = svd(X)

    P, Γ = svd_X.U, svd_X.S
    # θg_cov = construct_cov(truki, θ_p, θ_p_bar, g, g_bar)

    #Kalman Gain
    # K = θg_cov/gg_cov
    # use G(θ_bar) instead of FG(θ)

    θθ_cov_sqr =  Z * P ./ sqrt.(Γ .+ 1.0)' 

    θ_bar =  θ_p_bar + θθ_cov_sqr./ sqrt.(Γ .+ 1.0)'  * (P' * (V' * (Σ_ν\(truki.g_t - g[1,:]))))

    # K = θθ'G' (Gθθ'G' + Σ_ν)G(θ_ref - θ)
    # @info norm(truki.g_t - g[1,:]), norm(K*(truki.g_t - g[1,:]))
    
    

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
   
    truki.θ_bar .= copy(θ_bar) # N_ens x N_params
  
    truki.θθ_cov_sqr = copy(θθ_cov_sqr) # N_ens x N_data
    truki.counter += 1
    

end
