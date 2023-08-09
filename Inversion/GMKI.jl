using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions
using ForwardDiff

"""
UKIObj{FT<:AbstractFloat, IT<:Int}
Struct that is used in Unscented Kalman Inversion (UKI)
For solving the inverse problem 
    y = G(θ) + η
    
"""
mutable struct GMKIObj{FT<:AbstractFloat, IT<:Int}
    "a vector of arrays of size (N_modes) containing the modal weights of the parameters"
    logθ_w::Vector{Array{FT, 1}}
    "a vector of arrays of size (N_modes x N_parameters) containing the modal means of the parameters"
    θ_mean::Vector{Array{FT, 2}}
    "a vector of arrays of size (N_modes x N_parameters x N_parameters) containing the modal covariances of the parameters"
    θθ_cov::Vector{Array{FT, 3}}
    "a vector of arrays of size N_ensemble x N_y containing the predicted observation (in each gmki iteration a new array of predicted observation is added)"
    y_pred::Vector{Array{FT, 2}}
    "vector of observations (length: N_y)"
    y::Array{FT, 1}
    "covariance of the observational noise"
    Σ_η
    "number of modes"
    N_modes::IT
    "number ensemble size (2N_θ - 1)"
    N_ens::IT
    "size of θ"
    N_θ::IT
    "size of y"
    N_y::IT
    "weights in UKI"
    c_weights::Union{Array{FT, 1}, Array{FT, 2}}
    mean_weights::Array{FT, 1}
    cov_weights::Array{FT, 1}
    "adaptively adjust these weights based cov, otherwise fix it"
    adapt_α::Bool
    "parameters for updating weights"
    trunc_α::FT
    "Covariance matrix of the evolution error"
    Σ_ω::Union{Array{FT, 2}, Nothing}
    "time step"
    Δt::FT
    "update frequency"
    update_freq::IT
    "current iteration number"
    iter::IT
    mixture_power_sampling_method::String
    unscented_transform::String
end


function update_weights!(c_weights, mean_weights, cov_weights; 
                         unscented_transform = "modified-2n+1", adapt_α = false, trunc_α = -1.0, covs = nothing)
    
    N_θ = length(c_weights)
    N_ens = length(mean_weights)
    
    κ = 0.0
    β = 2.0
    α = min(sqrt(4/(N_θ + κ)), 1.0)
    
    if trunc_α > 0
        if adapt_α && covs != nothing 
            for im = 1:size(covs,1)
                _, D, _ = svd(covs[im,:,:])
                α = min(α, trunc_α/sqrt(D[1]))
            end
        else
            α = trunc_α
        end
    end


    λ = α^2*(N_θ + κ) - N_θ

    c_weights[1:N_θ]     .=  sqrt(N_θ + λ)
    mean_weights[1] = λ/(N_θ + λ)
    mean_weights[2:N_ens] .= 1/(2(N_θ + λ))
    cov_weights[1] = λ/(N_θ + λ) + 1 - α^2 + β
    cov_weights[2:N_ens] .= 1/(2(N_θ + λ))

    @assert(unscented_transform == "modified-2n+1" || unscented_transform == "original-2n+1")
    if unscented_transform == "modified-2n+1"
        mean_weights[1] = 1.0
        mean_weights[2:N_ens] .= 0.0
    end
    
end



"""
UKIObj Constructor 
θ0_mean::Array{FT} : initial mean
θθ0_cov::Array{FT, 2} : initial covariance
g_t::Array{FT,1} : observation 
obs_cov::Array{FT, 2} : observation covariance
α_reg::FT : regularization parameter toward θ0 (0 < α_reg <= 1), default should be 1, without regulariazion

unscented_transform : "original-2n+1", "modified-2n+1", "original-n+2", "modified-n+2" 
"""
function GMKIObj(θ0_w::Array{FT, 1},
                θ0_mean::Array{FT, 2}, 
                θθ0_cov::Array{FT, 3},
                y::Array{FT,1},
                Σ_η,
                Δt::FT,
                update_freq::IT;
                unscented_transform::String = "modified-2n+1",
                adapt_α = false,
                trunc_α = -1.0,
                mixture_power_sampling_method = "random-sampling") where {FT<:AbstractFloat, IT<:Int}

    ## check UKI hyperparameters
    @assert(update_freq > 0)
    if update_freq > 0 
        @assert(Δt > 0.0 && Δt < 1)
        @info "Start UKI on the mean-field stochastic dynamical system for Bayesian inference "
        Σ_ω = nothing
    end


    N_θ = size(θ0_mean,2)
    N_y = size(y, 1)

    @assert(unscented_transform == "original-2n+1" ||  unscented_transform == "modified-2n+1")

    # ensemble size
    N_ens = 2*N_θ+1

    c_weights = zeros(FT, N_θ)
    mean_weights = zeros(FT, N_ens)
    cov_weights = zeros(FT, N_ens)
    
    update_weights!(c_weights, mean_weights, cov_weights; unscented_transform = unscented_transform, trunc_α = trunc_α, adapt_α = adapt_α, covs=θθ0_cov)
    

    N_modes = length(θ0_w)

    logθ_w = Array{FT,1}[]      # array of Array{FT, 1}'s
    push!(logθ_w, log.(θ0_w))   # insert parameters at end of array (in this case just 1st entry)
    θ_mean = Array{FT,2}[]      # array of Array{FT, 2}'s
    push!(θ_mean, θ0_mean)      # insert parameters at end of array (in this case just 1st entry)
    θθ_cov = Array{FT,3}[]      # array of Array{FT, 2}'s
    push!(θθ_cov, θθ0_cov)      # insert parameters at end of array (in this case just 1st entry)

    y_pred = Array{FT, 2}[]     # array of Array{FT, 2}'s
   
    iter = 0

    GMKIObj{FT,IT}(logθ_w, θ_mean, θθ_cov, y_pred, 
                  y,   Σ_η, 
                  N_modes, N_ens, N_θ, N_y, 
                  c_weights, mean_weights, cov_weights, adapt_α, trunc_α,
                  Σ_ω, Δt,
                  update_freq, iter, mixture_power_sampling_method, unscented_transform)

end


"""
construct_sigma_ensemble
Construct the sigma ensemble, based on the mean x_mean, and covariance x_cov
"""
function construct_sigma_ensemble(gmki::GMKIObj{FT, IT}, x_means::Array{FT,2}, x_covs::Array{FT,3}) where {FT<:AbstractFloat, IT<:Int}
    N_modes, N_ens = gmki.N_modes, gmki.N_ens

    N_x = size(x_means[1, :],1)

    @assert(N_ens == 2*N_x+1 || N_ens == N_x+2)

    c_weights = gmki.c_weights

    xs = zeros(FT, N_modes, N_ens, N_x)

    for im = 1:N_modes
        chol_xx_cov = cholesky(Hermitian(x_covs[im,:,:])).L
        x_mean = x_means[im, :]
        x = zeros(FT, N_ens, N_x)

        if ndims(c_weights) == 1
            
            x[1, :] = x_mean
            for i = 1: N_x
                x[i+1,     :] = x_mean + c_weights[i]*chol_xx_cov[:,i]
                x[i+1+N_x, :] = x_mean - c_weights[i]*chol_xx_cov[:,i]
            end
        elseif ndims(c_weights) == 2
            x = zeros(FT, N_ens, N_x)
            x[1, :] = x_mean
            for i = 2: N_x + 2
                x[i,     :] = x_mean + chol_xx_cov * c_weights[:, i]
            end
        else
            error("c_weights dimensionality error")
        end

        xs[im, :, :] .= x 

    end

    return xs
end


"""
construct_mean x_mean from ensemble x
"""
function construct_mean(gmki::GMKIObj{FT, IT}, xs::Array{FT,3}) where {FT<:AbstractFloat, IT<:Int}
    N_modes, N_ens = gmki.N_modes, gmki.N_ens
    N_x = size(xs[1, :, :], 2)

    @assert(gmki.N_ens == N_ens)

    x_means = zeros(FT, N_modes, N_x)

    mean_weights = gmki.mean_weights

    for im = 1:N_modes
        for i = 1: N_ens
            x_means[im, :] += mean_weights[i]*xs[im, i, :]
        end
    end

    return x_means
end

"""
construct_cov xx_cov from ensemble x and mean x_mean
"""
function construct_cov(gmki::GMKIObj{FT, IT}, xs::Array{FT,3}, x_means::Array{FT , 2}) where {FT<:AbstractFloat, IT<:Int}
    N_modes, N_ens = gmki.N_modes, gmki.N_ens
    N_ens, N_x = gmki.N_ens, size(x_means[1, :],1)
    
    cov_weights = gmki.cov_weights

    xx_covs = zeros(FT, N_modes, N_x, N_x)

    for im = 1:N_modes
        for i = 1: N_ens
            xx_covs[im, :, :] .+= cov_weights[i]*(xs[im, i, :] - x_means[im, :])*(xs[im, i, :] - x_means[im, :])'
        end
    end

    return xx_covs
end

"""
construct_cov xy_cov from ensemble x and mean x_mean, ensemble y and mean y_mean
"""
function construct_cov(gmki::GMKIObj{FT, IT}, xs::Array{FT,3}, x_means::Array{FT, 2}, ys::Array{FT,3}, y_means::Array{FT, 2}) where {FT<:AbstractFloat, IT<:Int}
    N_modes, N_ens = gmki.N_modes, gmki.N_ens
    N_ens, N_x, N_y = gmki.N_ens, size(x_means[1, :],1), size(y_means[1, :],1)
    
    cov_weights = gmki.cov_weights

    xy_covs = zeros(FT, N_modes, N_x, N_y)

    for im = 1:N_modes
        for i = 1: N_ens
            xy_covs[im, :, :] .+= cov_weights[i]*(xs[im, i,:] - x_means[im, :])*(ys[im, i, :] - y_means[im, :])'
        end
    end

    return xy_covs
end

function Gaussian_density_helper(θ_mean::Array{FT,1}, θθ_cov::Array{FT,2}, θ::Vector) where {FT<:AbstractFloat}    
    return exp( -1/2*((θ - θ_mean)'* (θθ_cov\(θ - θ_mean)) )) / ( sqrt(det(θθ_cov)) )

end


function Gaussian_density(θ_mean::Array{FT,1}, θθ_cov::Array{FT,2}, θ::Array{FT,1}) where {FT<:AbstractFloat}
    N_θ = size(θ_mean,1)
    
    return exp( -1/2*((θ - θ_mean)'* (θθ_cov\(θ - θ_mean)) )) / ( (2π)^(N_θ/2)*sqrt(det(θθ_cov)) )

end
function Gaussian_mixture_density_helper(θ_w::Array{FT,1}, θ_mean::Array{FT,2}, θθ_cov::Array{FT,3}, θ::Vector) where {FT<:AbstractFloat}
    ρ = 0.0
    N_modes, N_θ = size(θ_mean)
    
    for i = 1:N_modes
        ρ += θ_w[i]*Gaussian_density_helper(θ_mean[i,:], θθ_cov[i,:,:], θ)
    end
    return ρ
end


function Gaussian_mixture_density(θ_w::Array{FT,1}, θ_mean::Array{FT,2}, θθ_cov::Array{FT,3}, θ::Array{FT,1}) where {FT<:AbstractFloat}
    ρ = 0.0
    N_modes, N_θ = size(θ_mean)
    
    for i = 1:N_modes
        ρ += θ_w[i]*Gaussian_density(θ_mean[i,:], θθ_cov[i,:,:], θ)
    end
    return ρ
end

function Gaussian_mixture_density_derivatives(θ_w::Array{FT,1}, θ_mean::Array{FT,2}, θθ_cov::Array{FT,3}, θ::Array{FT,1}) where {FT<:AbstractFloat}
    N_modes, N_θ = size(θ_mean)

    ρ = 0.0
    ∇ρ = zeros(N_θ)
    ∇²ρ = zeros(N_θ, N_θ)
   
    
    for i = 1:N_modes
        ρᵢ   = Gaussian_density_helper(θ_mean[i,:], θθ_cov[i,:,:], θ)
        ρ   += θ_w[i]*ρᵢ
        temp = θθ_cov[i,:,:]\(θ_mean[i,:] - θ)
        ∇ρ  += θ_w[i]*ρᵢ*temp
        ∇²ρ += θ_w[i]*ρᵢ*( temp * temp' - inv(θθ_cov[i,:,:]) )
    end
    return ρ, ∇ρ, ∇²ρ
end

"""
Compute f function without the constant in the front!
"""
function f_func_derivatives(θ_w::Array{FT,1}, θ_mean::Array{FT,2}, θθ_cov::Array{FT,3}, i::IT,  θ::Array{FT,1}, Δt::FT) where {FT<:AbstractFloat, IT<:Int}

    f_func = (θ)->(θ_w[i]*Gaussian_density_helper(θ_mean[i,:], θθ_cov[i,:,:], θ)/Gaussian_mixture_density_helper(θ_w, θ_mean, θθ_cov, θ))^Δt
    
    f, ∇f, ∇²f = f_func(θ), ForwardDiff.gradient(f_func, θ), ForwardDiff.hessian(f_func, θ) 
    return f, ∇f, ∇²f
end

# θ_w : N_modes array
# θ_mean: N_modes by N_θ array
# θθ_cov: N_modes by N_θ by N_θ array
# method = disjoint, sampling, approximation
function Gaussian_mixture_power(θ_w::Array{FT,1}, θ_mean::Array{FT,2}, θθ_cov::Array{FT,3}, αpower::FT; method="disjoint") where {FT<:AbstractFloat}
    N_modes, N_θ = size(θ_mean)
    
    
    # default assuming all components have disjoint supports
    θθ_cov_p = θθ_cov/αpower
    θ_w_p = copy(θ_w) 
    for i = 1:N_modes
        θ_w_p[i] = θ_w[i]^αpower * det(θθ_cov[i,:,:])^((1-αpower)/2)
    end
    θ_mean_p = copy(θ_mean)
    
    if method == "UKF-sampling"
        
        N_ens = 2N_θ+1
        α = sqrt((N_ens-1)/2.0)
        xs = zeros(N_ens, N_θ)
        ws = zeros(N_ens)
        for i = 1:N_modes
            
            # construct sigma points
            chol_xx_cov = cholesky(Hermitian(θθ_cov[i,:,:])).L
            
            xs[1, :] = θ_mean[i, :]
            ws[1] = θ_w[i]*Gaussian_mixture_density(θ_w, θ_mean, θθ_cov, xs[1, :])^(αpower - 1)
            for j = 1:N_θ
                xs[j+1,     :] = θ_mean[i, :] + α*chol_xx_cov[:,j]
                xs[j+1+N_θ, :] = θ_mean[i, :] - α*chol_xx_cov[:,j]
                
                ws[j+1]     = θ_w[i]*Gaussian_mixture_density(θ_w, θ_mean, θθ_cov, xs[j+1, :])^(αpower - 1)
                ws[j+1+N_θ] = θ_w[i]*Gaussian_mixture_density(θ_w, θ_mean, θθ_cov, xs[j+1+N_θ, :])^(αpower - 1)
            end
             
            
            θ_mean_p[i,:] = ws' * xs / sum(ws)
            θ_w_p[i] = sum(ws)/N_ens
            
        end
        
        
    elseif method == "random-sampling"
        
        N_ens = 10000
        α = sqrt((N_ens-1)/2.0)
        xs = zeros(N_ens, N_θ)
        ws = zeros(N_ens)
        for i = 1:N_modes
            
            Random.seed!(123);
            chol_xx_cov = cholesky(Hermitian(θθ_cov[i,:,:]/αpower)).L
            xs .= (θ_mean[i, :] .+ chol_xx_cov*rand(Normal(0, 1), N_θ, N_ens))'
            
            for j = 1:N_ens
                ws[j] = (θ_w[i]*Gaussian_density_helper(θ_mean[i,:], θθ_cov[i,:,:], xs[j, :])/Gaussian_mixture_density_helper(θ_w, θ_mean, θθ_cov, xs[j, :]))^(1 - αpower)
            end
            

            θ_mean_p[i,:] = ws' * xs / sum(ws)

            θθ_cov_p[i,:,:] = (xs - ones(N_ens)*θ_mean_p[i,:]')' * (ws .* (xs - ones(N_ens)*θ_mean_p[i,:]'))  / sum(ws)
            θ_w_p[i] = θ_w[i]^αpower * det(θθ_cov[i,:,:])^((1-αpower)/2) * sum(ws)/N_ens
      
        end
    elseif method == "random-sampling-derivatives"
        
        N_ens = 10000
        α = sqrt((N_ens-1)/2.0)
        xs = zeros(N_ens, N_θ)
        fs, ∇fs, ∇²fs = zeros(N_ens), zeros(N_ens, N_θ), zeros(N_ens, N_θ, N_θ)
        for i = 1:N_modes
            
            Random.seed!(123);
            chol_xx_cov = cholesky(Hermitian(θθ_cov[i,:,:]/αpower)).L
            xs .= (θ_mean[i, :] .+ chol_xx_cov*rand(Normal(0, 1), N_θ, N_ens))'
            
            for j = 1:N_ens
                fs[j], ∇fs[j,:], ∇²fs[j,:,:] = f_func_derivatives(θ_w, θ_mean, θθ_cov, i,  xs[j,:], 1 - αpower) 

            end
            

            # θ_mean_p[i,:] = fs' * xs / sum(fs)
            θ_mean_p[i,:] = θ_mean[i,:] + θθ_cov[i,:,:]/αpower * dropdims(sum(∇fs, dims=1), dims=1) / sum(fs)

            # θθ_cov_p[i,:,:] = (xs - ones(N_ens)*θ_mean_p[i,:]')' * (fs .* (xs - ones(N_ens)*θ_mean_p[i,:]'))  / sum(fs)
            θθ_cov_p[i,:,:] = θθ_cov[i,:,:]/αpower + (θ_mean_p[i,:]-θ_mean[i,:])*(θ_mean_p[i,:]-θ_mean[i,:])' + θθ_cov[i,:,:]/αpower^2 * (dropdims(sum(∇²fs, dims=1), dims=1)/sum(fs)) * θθ_cov[i,:,:]
            
            θ_w_p[i] = θ_w[i]^αpower * det(θθ_cov[i,:,:])^((1-αpower)/2) * sum(fs)/N_ens
      
        end

    elseif method == "continuous-time"

        Δt = 1 - αpower
        logθ_w_p = log.(θ_w) 
        for i = 1:N_modes
            
            ρ, ∇ρ, ∇²ρ = Gaussian_mixture_density_derivatives(θ_w, θ_mean, θθ_cov, θ_mean[i, :])
            logρ, ∇logρ, ∇²logρ  =  log(ρ), ∇ρ/ρ, (∇²ρ*ρ - ∇ρ*∇ρ')/ρ^2
       

            θ_mean_p[i,:] = θ_mean[i, :] - Δt*θθ_cov[i, :, :]*∇logρ
            θθ_cov_p[i,:] = inv( inv(θθ_cov[i, :, :]) + Δt*∇²logρ )  
            
            
            @info "svd = ", svd(inv(θθ_cov[i, :, :]) + Δt*∇²logρ)
            
            
            logθ_w_p[i]  -= Δt*logρ
            

        end

        # Normalization
        logθ_w_p .-= maximum(logθ_w_p)
        logθ_w_p .-= log( sum(exp.(logθ_w_p)) )
        θ_w_p = exp.(logθ_w_p)

    else
        @error("method :", method, " has not implemented")
        
    end
    
    return θ_w_p, θ_mean_p, θθ_cov_p
end

# reweight Gaussian mixture weights
function reweight(θ_w::Array{FT,1}, θ_mean::Array{FT,2}, θθ_cov::Array{FT,3}, θ_s::Array{FT,2}, p_s::Array{FT,1}) where {FT<:AbstractFloat}
    N_modes = size(θ_mean, 1)
    N_s   = length(p_s)
    A = zeros(N_s, N_modes)

    for i = 1:N_modes
        for j =1:N_s
            A[i, j] = Gaussian_density_helper(θ_mean[i,:], θθ_cov[i,:,:], θ_s[j,:])
        end
    end

    θ_w_p = A\p_s
    θ_w_p[θ_w_p .< 0] .= 0
    θ_w_p /= sum(θ_w_p)

    return θ_w_p
end
"""
update gmki struct
ens_func: The function g = G(θ)
define the function as 
    ens_func(θ_ens) = MyG(phys_params, θ_ens, other_params)
use G(θ_mean) instead of FG(θ)
"""
function update_ensemble!(gmki::GMKIObj{FT, IT}, ens_func::Function) where {FT<:AbstractFloat, IT<:Int}
    
    gmki.iter += 1

    N_θ, N_y, N_modes, N_ens = gmki.N_θ, gmki.N_y, gmki.N_modes, gmki.N_ens
    update_freq, Δt = gmki.update_freq, gmki.Δt
    

    Σ_ν = (1/Δt) * gmki.Σ_η
    # update evolution covariance matrix
    if update_freq > 0 && gmki.iter % update_freq == 0
        Σ_ω = Δt/(1 - Δt) * gmki.θθ_cov[end]
    else
        Σ_ω = gmki.Σ_ω 
    end

    θ_mean  = gmki.θ_mean[end]
    θθ_cov  = gmki.θθ_cov[end]
    logθ_w  = gmki.logθ_w[end]
    
    update_weights!(gmki.c_weights, gmki.mean_weights, gmki.cov_weights; unscented_transform = gmki.unscented_transform, trunc_α = gmki.trunc_α, covs = θθ_cov)
    
    y = gmki.y
    ############# Prediction step:
    
    logθ_w_p, θ_p_mean, θθ_p_cov = Gaussian_mixture_power(exp.(logθ_w), θ_mean, θθ_cov, 1-Δt; method=gmki.mixture_power_sampling_method)
    logθ_w_p = log.(logθ_w_p)
   
    ############ Generate sigma points
    θ_p = construct_sigma_ensemble(gmki, θ_p_mean, θθ_p_cov)

    # play the role of symmetrizing the covariance matrix
    θθ_p_cov = construct_cov(gmki, θ_p, θ_p_mean)
    

    ###########  Analysis step
    g = zeros(FT, N_modes, N_ens, N_y)
    
    g .= ens_func(θ_p)
    g_mean = construct_mean(gmki, g)
    gg_cov = construct_cov(gmki, g, g_mean)
    θg_cov = construct_cov(gmki, θ_p, θ_p_mean, g, g_mean)
    
    tmp = copy(θg_cov)
    θ_mean_n =  copy(θ_p_mean)
    θθ_cov_n = copy(θθ_p_cov)
    logθ_w_n = copy(logθ_w)

    for im = 1:N_modes
        tmp[im, :, :] = θg_cov[im, :, :] / (gg_cov[im, :, :] + Σ_ν)
        θ_mean_n[im, :] =  θ_p_mean[im, :] + tmp[im, :, :]*(y - g_mean[im, :])
        θθ_cov_n[im, :, :] =  θθ_p_cov[im, :, :] - tmp[im, :, :]*θg_cov[im, :, :]'         
    end


    # match expectation with UKI
    for im = 1:N_modes
        z = y - g_mean[im, :]
        logθ_w_n[im] = logθ_w_p[im] - 1/2*z'*(Σ_ν\z)
    end
    

    logθ_w_n .-= maximum(logθ_w_n)
    logθ_w_n .-= log( sum(exp.(logθ_w_n)) )

    # Clipping
    logθ_w_min = log(0.01)
    logθ_w_n[logθ_w_n .< logθ_w_min] .= logθ_w_min
    logθ_w_n .-= maximum(logθ_w_n)
    logθ_w_n .-= log( sum(exp.(logθ_w_n)) )
    

    # TODO reweight
    REWEIGHT = false
    if REWEIGHT
        g_mean_n = ens_func(θ_mean_n)
        p_n = zeros(N_modes)
        for i = 1:N_modes
            p_n[i] = exp(-1.0/2.0* ((y - g_mean_n[i,:])'*(gmki.Σ_η\(y - g_mean_n[i,:]))) )
        end
        θ_w_n = reweight(exp.(logθ_w_n), θ_mean_n, θθ_cov_n, θ_mean_n, p_n)
        logθ_w_n .= log.(θ_w_n)
    end

    ########### Save resutls
    push!(gmki.y_pred, g_mean)     # N_ens x N_data
    push!(gmki.θ_mean, θ_mean_n)   # N_ens x N_params
    push!(gmki.θθ_cov, θθ_cov_n)   # N_ens x N_data
    push!(gmki.logθ_w, logθ_w_n)   # N_ens x N_data
    

end



function GMKI_Run(s_param, forward::Function, 
    θ0_w, θ0_mean, θθ0_cov,
    y, Σ_η,
    Δt,
    update_freq,
    N_iter;
    unscented_transform::String = "modified-2n+1",
    mixture_power_sampling_method = "random-sampling",
    θ_basis = nothing,
    adapt_α = false,
    trunc_α = -1.0)
    
    
    gmkiobj = GMKIObj(
    θ0_w, θ0_mean, θθ0_cov,
    y, Σ_η,
    Δt,
    update_freq;
    unscented_transform = unscented_transform, 
    mixture_power_sampling_method = mixture_power_sampling_method,
    adapt_α = adapt_α,
    trunc_α = trunc_α)
    
    
    ens_func(θ_ens) = (θ_basis == nothing) ? 
    ensemble(s_param, θ_ens, forward) : 
    # θ_ens is N_ens × N_θ
    # θ_basis is N_θ × N_θ_high
    ensemble(s_param, θ_ens * θ_basis, forward)
    
    
    for i in 1:N_iter
        update_ensemble!(gmkiobj, ens_func) 
    end
    
    return gmkiobj
    
end


