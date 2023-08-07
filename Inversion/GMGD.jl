using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions


"""
GMGDObj{FT<:AbstractFloat, IT<:Int}
Struct that is used in sampling e^{-V} with Gaussian mixture gradient descent
"""
mutable struct GMGDObj{FT<:AbstractFloat, IT<:Int}
    "a vector of arrays of size (N_modes) containing the modal weights of the parameters"
    logθ_w::Vector{Array{FT, 1}}
    "a vector of arrays of size (N_modes x N_parameters) containing the modal means of the parameters"
    θ_mean::Vector{Array{FT, 2}}
    "a vector of arrays of size (N_modes x N_parameters x N_parameters) containing the modal covariances of the parameters"
    θθ_cov::Union{Vector{Array{FT, 3}}, Nothing}
    "sample points"
    N_ens::IT
    "number of modes"
    N_modes::IT
    "size of θ"
    N_θ::IT
    "current iteration number"
    iter::IT
    "update covariance or not"
    update_covariance::Bool
    "metric"
    metric::String
    "the method to approximate the expectation in the gradient flow"
    expectation_method::String
end



"""
GMGDObj Constructor 
"""
function GMGDObj(metric::String,
                update_covariance::Bool,
                # initial condition
                θ0_w::Array{FT, 1},
                θ0_mean::Array{FT, 2},
                θθ0_cov::Union{Array{FT, 3}, Nothing},
                expectation_method::String = "random-sampling",
                N_ens::IT = 1) where {FT<:AbstractFloat, IT<:Int}


    N_θ = size(θ0_mean, 2)

    N_modes = length(θ0_w)

    logθ_w = Array{FT,1}[]      # array of Array{FT, 1}'s
    push!(logθ_w, log.(θ0_w))         # insert parameters at end of array (in this case just 1st entry)
    θ_mean = Array{FT,2}[]   # array of Array{FT, 2}'s
    push!(θ_mean, θ0_mean)   # insert parameters at end of array (in this case just 1st entry)
    θθ_cov = Array{FT,3}[]   # array of Array{FT, 2}'s
    push!(θθ_cov, θθ0_cov)   # insert parameters at end of array (in this case just 1st entry)
    
    

    iter = 0

    GMGDObj(logθ_w, θ_mean, θθ_cov, N_ens,
                  N_modes, 
                  N_θ,
                  iter,
                  update_covariance,
                  metric,
                  expectation_method)

end


"""
construct_ensemble
Construct the ensemble, based on the mean x_mean, and covariance x_cov
"""
function construct_ensemble(gmgd::GMGDObj{FT, IT}, x_means::Array{FT,2}, x_covs::Array{FT,3}) where {FT<:AbstractFloat, IT<:Int}
    expectation_method = gmgd.expectation_method
    N_modes, N_ens = gmgd.N_modes, gmgd.N_ens
    N_x = size(x_means[1, :],1)
    
    if expectation_method == "random_sampling"
        
        xs = zeros(FT, N_modes, N_ens, N_x)
        for im = 1:N_modes
            chol_xx_cov = cholesky(Hermitian(x_covs[im,:,:])).L
            xs[im, :, :] = ones(N_ens)*x_means[im, :]' + rand(Normal(0, 1), N_ens, N_x) * chol_xx_cov'
        end

    elseif expectation_method == "unscented_transform"
        @assert(N_ens == 1)
        xs = zeros(FT, N_modes, N_ens, N_x)
        
        for im = 1:N_modes
            xs[im, :, :] = x_means[im, :]
        end

    else 
        error("expectation_method ", expectation_method, " has not implemented!")

    end

    return xs
end


"""
construct_mean x_mean from ensemble x
"""
function construct_mean(gmgd::GMGDObj{FT, IT}, xs::Array{FT}) where {FT<:AbstractFloat, IT<:Int}
    N_modes = gmgd.N_modes
    # xs is a N_modes by N_ens by size(x) array 
    ndims_x = ndims(xs) - 2
    sizes_x = size(xs)[3:end]

    x_means = zeros(FT, N_modes, sizes_x...)
    mean_weights = 1/gmgd.N_ens
    for im = 1:N_modes
        x_means[im, repeat([:],ndims_x)...] = sum(mean_weights*xs[im, repeat([:],ndims_x+1)...], dims=1)[1, repeat([:],ndims_x)...] 
    end
    return x_means
end




function Gaussian_density_helper(θ_mean::Array{FT,1}, θθ_cov::Array{FT,2}, θ::Array{FT,1}) where {FT<:AbstractFloat}
    return exp( -1/2*((θ - θ_mean)'* (θθ_cov\(θ - θ_mean)) )) / ( sqrt(det(θθ_cov)) )
end

function Gaussian_mixture_density_all_helper(θ_w::Array{FT,1}, θ_mean::Array{FT,2}, θθ_cov::Array{FT,3}, θ::Array{FT,1}) where {FT<:AbstractFloat}
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



function compute_logρ_gm(θ_p, θ_w, θ_mean, θθ_cov)
    N_modes, N_ens, N_θ = size(θ_p)
    logρ = zeros(N_modes, N_ens)
    ∇logρ = zeros(N_modes, N_ens, N_θ)
    ∇²logρ = zeros(N_modes, N_ens, N_θ, N_θ)
    for im = 1:N_modes
        for i = 1:N_ens
            ρ, ∇ρ, ∇²ρ = Gaussian_mixture_density_all_helper(θ_w, θ_mean, θθ_cov, θ_p[im, i, :])
            logρ[im, i]         =   log(  ρ  )
            ∇logρ[im, i, :]     =   ∇ρ/ρ
            ∇²logρ[im, i, :, :] =  (∇²ρ*ρ - ∇ρ*∇ρ')/ρ^2
        end

    end

    return logρ, ∇logρ, ∇²logρ
end
"""
update gmgd struct
ens_func: The function g = G(θ)
define the function as 
    ens_func(θ_ens) = MyG(phys_params, θ_ens, other_params)
use G(θ_mean) instead of FG(θ)
"""
function update_ensemble!(gmgd::GMGDObj{FT, IT}, func_logρ::Function, dt::FT) where {FT<:AbstractFloat, IT<:Int}
    
    metric = gmgd.metric
    update_covariance = gmgd.update_covariance
    
    gmgd.iter += 1
    N_θ,  N_modes = gmgd.N_θ, gmgd.N_modes

    θ_mean  = gmgd.θ_mean[end]
    logθ_w  = gmgd.logθ_w[end]
    θθ_cov  = gmgd.θθ_cov[end]


    ############ Generate sigma points
    θ_p = construct_ensemble(gmgd, θ_mean, θθ_cov)
    ###########  Entropy term
    logρ, ∇logρ, ∇²logρ = compute_logρ_gm(θ_p, exp.(logθ_w), θ_mean, θθ_cov)
    logρ_mean, ∇logρ_mean, ∇²logρ_mean  = construct_mean(gmgd, logρ), construct_mean(gmgd, ∇logρ), construct_mean(gmgd, ∇²logρ)
    ###########  Potential term
    V, ∇V, ∇²V = func_logρ(θ_p)
    V_mean, ∇V_mean, ∇²V_mean  = construct_mean(gmgd, V), construct_mean(gmgd, ∇V), construct_mean(gmgd, ∇²V)

    θ_mean_n =  copy(θ_mean)
    θθ_cov_n = copy(θθ_cov)
    logθ_w_n = copy(logθ_w)


    if metric == "Fisher-Rao"
        for im = 1:N_modes
            θ_mean_n[im, :]    =  θ_mean[im, :] - dt*θθ_cov[im, :, :]*(∇logρ_mean[im, :] + ∇V_mean[im, :]) 

            
            if update_covariance
                θθ_cov_n[im, :, :] =  inv( inv(θθ_cov[im, :, :]) + dt*(∇²logρ_mean[im, :, :] + ∇²V_mean[im, :, :]) )
            else
                θθ_cov_n[im, :, :] = θθ_cov[im, :, :]
            end
            
            
            ρlogρ_V = 0 
            for im = 1:N_modes
                ρlogρ_V += exp(logθ_w[im])*(logρ_mean[im] + V_mean[im])
            end
            logθ_w_n[im] = logθ_w[im] - dt*(logρ_mean[im] + V_mean[im] - ρlogρ_V)
            
        end
       
    end

    # Normalization
    logθ_w_n .-= maximum(logθ_w_n)
    logθ_w_n .-= log( sum(exp.(logθ_w_n)) )


    ########### Save resutls
    push!(gmgd.θ_mean, θ_mean_n)   # N_ens x N_params
    push!(gmgd.θθ_cov, θθ_cov_n)   # N_ens x N_data
    push!(gmgd.logθ_w, logθ_w_n)   # N_ens x N_data
end

function ensemble(θ_ens, forward)
    N_modes, N_ens, N_θ = size(θ_ens)

    V = zeros(N_modes, N_ens)   
    ∇V = zeros(N_modes, N_ens, N_θ)   
    ∇²V = zeros(N_modes, N_ens, N_θ, N_θ)  

    for im = 1:N_modes
        for i = 1:N_ens
            V[im, i], ∇V[im, i, :], ∇²V[im, i, :, :] = forward(θ_ens[im, i, :])
        end
    end

    return V, ∇V, ∇²V 
end

function GMGD_Run(
    forward::Function, 
    N_iter::IT,
    T::FT,
    metric::String,
    update_covariance::Bool, 
    θ0_w::Array{FT, 1}, θ0_mean::Array{FT, 2}, θθ0_cov::Array{FT, 3},
    expectation_method::String = "unscented_transform",
    N_ens::IT = 1) where {FT<:AbstractFloat, IT<:Int}
    
    
    
    gmgdobj = GMGDObj(
    metric, 
    update_covariance, 
    θ0_w, θ0_mean, θθ0_cov,
    expectation_method, N_ens)
     
    func_logρ(θ_ens) = ensemble(θ_ens, forward)  
    
    dt = T/N_iter
    for i in 1:N_iter
        update_ensemble!(gmgdobj, func_logρ, dt) 
    end
    
    return gmgdobj
    
end







