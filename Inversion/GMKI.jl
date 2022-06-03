using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions


# TODO Delete
using PyPlot
include("Plot.jl")
include("Utility.jl")
include("RWMCMC.jl")
include("SMC.jl")
include("EKS.jl")
# generate ensemble
Random.seed!(123)

"""
UKIObj{FT<:AbstractFloat, IT<:Int}
Struct that is used in Unscented Kalman Inversion (UKI)
For solving the inverse problem 
    y = G(θ) + η
    
"""
mutable struct GMUKIObj{FT<:AbstractFloat, IT<:Int}
"vector of parameter names (never used)"
    θ_names::Array{String,1}
    "a vector of arrays of size (N_modes) containing the modal weights of the parameters"
    logθ_w::Vector{Array{FT, 1}}
    "a vector of arrays of size (N_modes x N_parameters) containing the modal means of the parameters"
    θ_mean::Vector{Array{FT, 2}}
    "a vector of arrays of size (N_modes x N_parameters x N_parameters) containing the modal covariances of the parameters"
    θθ_cov::Vector{Array{FT, 3}}
    "a vector of arrays of size N_ensemble x N_y containing the predicted observation (in each uki iteration a new array of predicted observation is added)"
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
    "Covariance matrix of the evolution error"
    Σ_ω::Union{Array{FT, 2}, Nothing}
    "inflation factor for evolution"
    γ_ω::FT
    "inflation factor for observation"
    γ_ν::FT
    "update frequency"
    update_freq::IT
    "current iteration number"
    iter::IT
end



"""
UKIObj Constructor 
parameter_names::Array{String,1} : parameter name vector
θ0_mean::Array{FT} : prior mean
θθ0_cov::Array{FT, 2} : prior covariance
g_t::Array{FT,1} : observation 
obs_cov::Array{FT, 2} : observation covariance
α_reg::FT : regularization parameter toward θ0 (0 < α_reg <= 1), default should be 1, without regulariazion

unscented_transform : "original-2n+1", "modified-2n+1", "original-n+2", "modified-n+2" 
"""
function GMUKIObj(θ_names::Array{String,1},
                # initial condition
                θ0_w::Array{FT, 1},
                θ0_mean::Array{FT, 2}, 
                θθ0_cov::Array{FT, 3},
                y::Array{FT,1},
                Σ_η,
                γ::FT,
                update_freq::IT;
                unscented_transform::String = "modified-2n+1") where {FT<:AbstractFloat, IT<:Int}

    ## check UKI hyperparameters
    @assert(update_freq > 0)
    if update_freq > 0 
        @info "Start UKI on the mean-field stochastic dynamical system for Bayesian inference "
        @assert(γ > 0.0)
        γ_ω = γ
        γ_ν = (γ  + 1.0)/γ 
        Σ_ω = nothing
    end



    N_θ = size(θ0_mean,2)
    N_y = size(y, 1)
    

 

    if unscented_transform == "original-2n+1" ||  unscented_transform == "modified-2n+1"

        # ensemble size
        N_ens = 2*N_θ+1

        c_weights = zeros(FT, N_θ)
        mean_weights = zeros(FT, N_ens)
        cov_weights = zeros(FT, N_ens)

        κ = 0.0
        β = 2.0
        α = min(sqrt(4/(N_θ + κ)), 1.0)
        λ = α^2*(N_θ + κ) - N_θ


        c_weights[1:N_θ]     .=  sqrt(N_θ + λ)
        mean_weights[1] = λ/(N_θ + λ)
        mean_weights[2:N_ens] .= 1/(2(N_θ + λ))
        cov_weights[1] = λ/(N_θ + λ) + 1 - α^2 + β
        cov_weights[2:N_ens] .= 1/(2(N_θ + λ))

        if unscented_transform == "modified-2n+1"
            mean_weights[1] = 1.0
            mean_weights[2:N_ens] .= 0.0
        end

    elseif unscented_transform == "original-n+2" ||  unscented_transform == "modified-n+2"

        N_ens = N_θ+2
        c_weights = zeros(FT, N_θ, N_ens)
        mean_weights = zeros(FT, N_ens)
        cov_weights = zeros(FT, N_ens)

        # todo cov parameter
        α = N_θ/(4*(N_θ+1))
	
        IM = zeros(FT, N_θ, N_θ+1)
        IM[1,1], IM[1,2] = -1/sqrt(2α), 1/sqrt(2α)
        for i = 2:N_θ
            for j = 1:i
                IM[i,j] = 1/sqrt(α*i*(i+1))
            end
            IM[i,i+1] = -i/sqrt(α*i*(i+1))
        end
        c_weights[:, 2:end] .= IM

        if unscented_transform == "original-n+2"
            mean_weights .= 1/(N_θ+1)
            mean_weights[1] = 0.0

            cov_weights .= α
            cov_weights[1] = 0.0

        else unscented_transform == "modified-n+2"
            mean_weights .= 0.0
            mean_weights[1] = 1.0
            
            cov_weights .= α
            cov_weights[1] = 0.0
        end

    else

        error("unscented_transform: ", unscented_transform, " is not recognized")
    
    end

    N_modes = length(θ0_w)

    logθ_w = Array{FT,1}[]      # array of Array{FT, 1}'s
    push!(logθ_w, log.(θ0_w))         # insert parameters at end of array (in this case just 1st entry)
    θ_mean = Array{FT,2}[]   # array of Array{FT, 2}'s
    push!(θ_mean, θ0_mean)   # insert parameters at end of array (in this case just 1st entry)
    θθ_cov = Array{FT,3}[]   # array of Array{FT, 2}'s
    push!(θθ_cov, θθ0_cov)   # insert parameters at end of array (in this case just 1st entry)

    y_pred = Array{FT, 2}[]  # array of Array{FT, 2}'s
   
    iter = 0

    GMUKIObj{FT,IT}(θ_names, logθ_w, θ_mean, θθ_cov, y_pred, 
                  y,   Σ_η, 
                  N_modes, N_ens, N_θ, N_y, 
                  c_weights, mean_weights, cov_weights, 
                  Σ_ω, γ_ω, γ_ν,
                  update_freq, iter)

end


"""
construct_sigma_ensemble
Construct the sigma ensemble, based on the mean x_mean, and covariance x_cov
"""
function construct_sigma_ensemble(uki::GMUKIObj{FT, IT}, x_means::Array{FT,2}, x_covs::Array{FT,3}) where {FT<:AbstractFloat, IT<:Int}
    N_modes, N_ens = uki.N_modes, uki.N_ens

    N_x = size(x_means[1, :],1)

    @assert(N_ens == 2*N_x+1 || N_ens == N_x+2)

    c_weights = uki.c_weights

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
function construct_mean(uki::GMUKIObj{FT, IT}, xs::Array{FT,3}) where {FT<:AbstractFloat, IT<:Int}
    N_modes, N_ens = uki.N_modes, uki.N_ens
    N_x = size(xs[1, :, :], 2)

    @assert(uki.N_ens == N_ens)

    x_means = zeros(FT, N_modes, N_x)

    mean_weights = uki.mean_weights

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
function construct_cov(uki::GMUKIObj{FT, IT}, xs::Array{FT,3}, x_means::Array{FT , 2}) where {FT<:AbstractFloat, IT<:Int}
    N_modes, N_ens = uki.N_modes, uki.N_ens
    N_ens, N_x = uki.N_ens, size(x_means[1, :],1)
    
    cov_weights = uki.cov_weights

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
function construct_cov(uki::GMUKIObj{FT, IT}, xs::Array{FT,3}, x_means::Array{FT, 2}, ys::Array{FT,3}, y_means::Array{FT, 2}) where {FT<:AbstractFloat, IT<:Int}
    N_modes, N_ens = uki.N_modes, uki.N_ens
    N_ens, N_x, N_y = uki.N_ens, size(x_means[1, :],1), size(y_means[1, :],1)
    
    cov_weights = uki.cov_weights

    xy_covs = zeros(FT, N_modes, N_x, N_y)

    for im = 1:N_modes
        for i = 1: N_ens
            xy_covs[im, :, :] .+= cov_weights[i]*(xs[im, i,:] - x_means[im, :])*(ys[im, i, :] - y_means[im, :])'
        end
    end

    return xy_covs
end




function Gaussian_mixture_density(θ_w::Array{FT,1}, θ_mean::Array{FT,2}, θθ_cov::Array{FT,3}, θ::Array{FT,1}) where {FT<:AbstractFloat}
    ρ = 0.0
    N_modes, N_θ = size(θ_mean)
    
    for i = 1:N_modes
        ρ += θ_w[i]*exp( -1/2*((θ - θ_mean[i,:])'* (θθ_cov[i,:,:]\(θ - θ_mean[i,:])) ))
    end
    return ρ
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
    
    if method == "sampling"
        
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

            θ_mean_p[i,:] = θ_mean[i, :] + 5(θ_mean_p[i,:] - θ_mean[i, :])
            
            @show i, θ_mean[i,:], θ_mean_p[i,:]
        end
        
        
    elseif method == "random-sampling"
        
        N_ens = 1000
        α = sqrt((N_ens-1)/2.0)
        xs = zeros(N_ens, N_θ)
        ws = zeros(N_ens)
        for i = 1:N_modes
            
            # construct sigma points
            
            
            xs .= rand(MvNormal(θ_mean[i, :], θθ_cov[i,:,:]), N_ens)'
            
            for j = 1:N_ens
                ws[j] = θ_w[i]*Gaussian_mixture_density(θ_w, θ_mean, θθ_cov, xs[j, :])^(αpower - 1)
            end
             
            
            θ_mean_p[i,:] = ws' * xs / sum(ws)
            
            @show i, θ_mean[i,:], θ_mean_p[i,:]
        end

    else
        @error("method :", method, " has not implemented")
        
    end
    θ_w_p ./ sum(θ_w_p)
    
    return θ_w_p, θ_mean_p, θθ_cov_p
end


"""
update uki struct
ens_func: The function g = G(θ)
define the function as 
    ens_func(θ_ens) = MyG(phys_params, θ_ens, other_params)
use G(θ_mean) instead of FG(θ)
"""
function update_ensemble!(uki::GMUKIObj{FT, IT}, ens_func::Function) where {FT<:AbstractFloat, IT<:Int}
    
    uki.iter += 1

    N_θ, N_y, N_modes, N_ens = uki.N_θ, uki.N_y, uki.N_modes, uki.N_ens
    update_freq, γ_ν, γ_ω = uki.update_freq, uki.γ_ν, uki.γ_ω
    

    Σ_ν = γ_ν * uki.Σ_η
    # update evolution covariance matrix
    if update_freq > 0 && uki.iter % update_freq == 0
        Σ_ω = γ_ω * uki.θθ_cov[end]
    else
        Σ_ω = uki.Σ_ω 
    end

    θ_mean  = uki.θ_mean[end]
    θθ_cov  = uki.θθ_cov[end]
    logθ_w  = uki.logθ_w[end]

    y = uki.y
    ############# Prediction step:
    θ_p_mean  = θ_mean
    #TODO push mean
    # θ_mean_all = mean(θ_mean, dims=1)
    # θ_p_mean  = ones(N_modes)*θ_mean_all + sqrt(1 +  0.2)*(θ_mean - ones(N_modes)*θ_mean_all)
    

    θθ_p_cov = (Σ_ω === nothing ? θθ_cov : θθ_cov + Σ_ω)
    logθ_w_p = (1/(γ_ω + 1))  * logθ_w
    for im = 1:N_modes
        logθ_w_p[im] += (γ_ω/(2*(γ_ω + 1)))*log(det(θθ_cov[im,:,:]))
    end

    ##TODO
    logθ_w_p, θ_p_mean, θθ_p_cov = Gaussian_mixture_power(exp.(logθ_w), θ_mean, θθ_cov, 1/(γ_ω + 1); method="random-sampling")
    logθ_w_p = log.(logθ_w_p)

    @info "updata : θ_mean = ", θ_mean, " θ_p_mean = ", θ_p_mean
    # logθ_w_p = logθ_w
    
    ############ Generate sigma points
    θ_p = construct_sigma_ensemble(uki, θ_p_mean, θθ_p_cov)

    # play the role of symmetrizing the covariance matrix
    θθ_p_cov = construct_cov(uki, θ_p, θ_p_mean)
    

    ###########  Analysis step
    g = zeros(FT, N_modes, N_ens, N_y)
    

    g .= ens_func(θ_p)
    
    
    g_mean = construct_mean(uki, g)
    # gg_cov = construct_cov(uki, g, g_mean) + Σ_ν
    gg_cov = construct_cov(uki, g, g_mean)
    θg_cov = construct_cov(uki, θ_p, θ_p_mean, g, g_mean)
    
    tmp = copy(θg_cov)
    θ_mean_n =  copy(θ_p_mean)
    θθ_cov_n = copy(θθ_p_cov)
    logθ_w_n = copy(logθ_w)

    for im = 1:N_modes

        tmp[im, :, :] = θg_cov[im, :, :] / (gg_cov[im, :, :] + Σ_ν)

        θ_mean_n[im, :] =  θ_p_mean[im, :] + tmp[im, :, :]*(y - g_mean[im, :])

        θθ_cov_n[im, :, :] =  θθ_p_cov[im, :, :] - tmp[im, :, :]*θg_cov[im, :, :]' 

        z = y - g_mean[im, :]
        temp = θθ_cov[im, :, :]\(θg_cov[im, :, :]*(Σ_ν\z))
        logθ_w_n[im] = 1/2*( temp'*θθ_cov_n[im, :, :]*temp -  z'*(Σ_ν\z))
    end

   
    for im = 1:N_modes
        logθ_w_n[im] += logθ_w_p[im] + log(sqrt(det(θθ_cov_n[im, :, :]) / det(θθ_cov[im, :, :])))
    end

    logθ_w_n .-= maximum(logθ_w_n)
    logθ_w_n .-= log( sum(exp.(logθ_w_n)) )
    

    ########### Save resutls
    push!(uki.y_pred, g_mean)     # N_ens x N_data
    push!(uki.θ_mean, θ_mean_n)   # N_ens x N_params
    push!(uki.θθ_cov, θθ_cov_n)   # N_ens x N_data
    push!(uki.logθ_w, logθ_w_n)   # N_ens x N_data
end



function GMUKI_Run(s_param, forward::Function, 
    θ0_w, θ0_mean, θθ0_cov,
    y, Σ_η,
    γ,
    update_freq,
    N_iter;
    unscented_transform::String = "modified-2n+1",
    θ_basis = nothing)
    
    θ_names = s_param.θ_names
    
    
    ukiobj = GMUKIObj(θ_names ,
    θ0_w, θ0_mean, θθ0_cov,
    y, Σ_η,
    γ,
    update_freq;
    unscented_transform = unscented_transform)
    
    
    ens_func(θ_ens) = (θ_basis == nothing) ? 
    ensemble(s_param, θ_ens, forward) : 
    # θ_ens is N_ens × N_θ
    # θ_basis is N_θ × N_θ_high
    ensemble(s_param, θ_ens * θ_basis, forward)
    
    
    for i in 1:N_iter
        update_ensemble!(ukiobj, ens_func) 
    end
    
    return ukiobj
    
end


function Posterior_Plot(forward::Function, forward_aug::Function;  θ_ref = 2.0, σ_η = 0.1, μ_0 = 3.0,  σ_0 = 2.0)
    Run_EKS = false
    Run_SMC = false
    Run_MCMC = true
    N_y, N_θ = 1, 1
    s_param = Setup_Param(N_θ, N_y)
    y = forward(s_param, [θ_ref;])
    Σ_η = reshape([σ_η^2], (N_y, N_y))
    # prior distribution
    μ0,  Σ0   = [μ_0;], reshape([σ_0^2],  (N_θ, N_θ))
    


    # compute posterior distribution by UKI
    update_freq = 1
    N_iter = 50
    N_modes = 2
    θ0_w  = [0.5; 0.5]
    θ0_mean, θθ0_cov  = zeros(N_modes, N_θ), zeros(N_modes, N_θ, N_θ)
    
    θ0_mean[1, :]    .= -3.0
    θθ0_cov[1, :, :] .=  reshape([0.5^2],  (1, 1))
    θ0_mean[2, :]    .=  3.0
    θθ0_cov[2, :, :] .=  reshape([0.5^2],  (1, 1))
    
    s_param_aug = Setup_Param(1,2)
    y_aug = [y ; μ0]
    Σ_η_aug = [Σ_η zeros(Float64, N_y, N_θ); zeros(Float64, N_θ, N_y) Σ0]
    γ = 1.0
    # Δt = γ/(1+γ)
    ukiobj = GMUKI_Run(s_param_aug, forward_aug, θ0_w, θ0_mean, θθ0_cov, y_aug, Σ_η_aug, γ, update_freq, N_iter; unscented_transform="modified-2n+1")
    


    if Run_MCMC
        # compute posterior distribution by MCMC
        logρ(θ) = log_bayesian_posterior(s_param, θ, forward, y, Σ_η, μ0, Σ0) 
        step_length = 1.0
        N_iter , n_burn_in= 5000000, 1000000
        us = RWMCMC_Run(logρ, μ0, step_length, N_iter)
    end
    
    # compute posterior distribution by SMC
    if Run_SMC
        N_ens = 1000
        M_threshold = Float64(N_ens)
        N_t = 100
        step_length = 1.0
        smcobj = SMC_Run(s_param, forward,
        μ0, Σ0, 
        y, Σ_η,
        N_ens, 
        step_length,
        M_threshold,
        N_t) 
    end

    if Run_EKS
        N_iter = 100
        N_ens  = 1000
        eksobj = EKS_Run(s_param, forward, 
        μ0, Σ0,
        N_ens,
        y, Σ_η,
        N_iter)
        @info "EKS large J t = ", sum(eksobj.Δt)
    end
    
    
    # visualization 

    # Visualize different iterations
    for iter  = 1:length(ukiobj.θ_mean)
        nrows, ncols = 1, 1
        fig, ax = PyPlot.subplots(nrows=nrows, ncols=ncols, sharex=true, sharey=true, figsize=(6,6))
        # plot UKI results 

        Nx = 1000
        xxs, zzs = zeros(N_modes, Nx), zeros(N_modes, Nx)
        θ_min = minimum(ukiobj.θ_mean[iter][:,1] .- 5sqrt.(ukiobj.θθ_cov[iter][:,1,1]))
        θ_max = maximum(ukiobj.θ_mean[iter][:,1] .+ 5sqrt.(ukiobj.θθ_cov[iter][:,1,1]))
        
        for i =1:N_modes
            xxs[i, :], zzs[i, :] = Gaussian_1d(ukiobj.θ_mean[iter][i,1], ukiobj.θθ_cov[iter][i,1,1], Nx, θ_min, θ_max)
            zzs[i, :] *= exp(ukiobj.logθ_w[iter][i])
            ax.plot(xxs[i,:], zzs[i,:], marker= "o", linestyle=":", color="C"*string(i), fillstyle="none", markevery=100, label="UKI Modal "*string(i))
        end
        ax.plot(xxs[1,:], sum(zzs, dims=1)', marker= "*", linestyle="-", color="C0", fillstyle="none", markevery=100, label="UKI")
        


        
        if Run_EKS
            # plot EKS results 
            θ = eksobj.θ[end]
            ax.hist(θ, bins = 40, density = true, histtype = "step", label="EKS", color="C4")
            ax.legend()
        end
        

        if Run_MCMC
            # plot MCMC results 
            ax.hist(us[n_burn_in:end, 1], bins = 100, density = true, histtype = "step", label="MCMC", color="C3")
        end


        if Run_SMC
            # plot SMC results 
            θ = smcobj.θ[end]
            weights = smcobj.weights[end]
            ax.hist(θ, bins = 20, weights = weights, density = true, histtype = "step", label="SMC", color="C0")  
        end

        ax.legend()
    end
    
    
    nrows, ncols = 1, 1
    fig, ax = PyPlot.subplots(nrows=nrows, ncols=ncols, sharex=true, sharey=true, figsize=(6,6))
    θ_w = exp.(hcat(ukiobj.logθ_w...))
    for i =1:N_modes
        ax.plot(θ_w[i, :], "--o", label="mode"*string(i))
    end
    ax.legend()
end



mutable struct Setup_Param{IT<:Int}
    θ_names::Array{String,1}
    N_θ::IT
    N_y::IT
end

function Setup_Param(N_θ::IT, N_y::IT) where {IT<:Int}
    return Setup_Param(["θ"], N_θ, N_y)
end

function p1(s_param, θ::Array{Float64,1})  
    return [θ[1] ;]
end

function p1_aug(s_param, θ::Array{Float64,1})  
    return [θ[1] ; θ[1]]
end

function p1_aug_derivative(s_param, θ::Array{Float64,1})  
    return [θ[1] ; θ[1]], [1.0 ; 1.0]
end



function p2(s_param, θ::Array{Float64,1})  
    return [θ[1]^2 ;]
end

function p2_aug(s_param, θ::Array{Float64,1})  
    return [θ[1]^2 ; θ[1]]
end

function p2_aug_derivative(s_param, θ::Array{Float64,1})  
    return [θ[1]^2 ; θ[1]], [2θ[1] ; 1.0]
end



Posterior_Plot(p2, p2_aug; θ_ref = 1.0, σ_η = 1.0, μ_0 = 3.0,  σ_0 = 2.0) 