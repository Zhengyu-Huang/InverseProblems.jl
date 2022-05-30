using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions


# TODO Delete
include("Utility.jl")


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
    θθ_p_cov = (Σ_ω === nothing ? θθ_cov : θθ_cov + Σ_ω)

    logθ_w_p = (1/(γ_ω + 1))  * logθ_w
    for im = 1:N_modes
        logθ_w_p[im] += (γ_ω/(2*(γ_ω + 1)))*log(det(θθ_cov[im,:,:]))
    end

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

# """
# update uki struct
# ens_func: The function g = G(θ)
# define the function as 
#     ens_func(θ_ens) = MyG(phys_params, θ_ens, other_params)
# use G(θ_mean) instead of FG(θ)
# """
# function update_ensemble!(uki::GMUKIObj{FT, IT}, ens_func::Function) where {FT<:AbstractFloat, IT<:Int}
    
#     uki.iter += 1

#     N_θ, N_y, N_modes, N_ens = uki.N_θ, uki.N_y, uki.N_modes, uki.N_ens
#     update_freq, γ_ν, γ_ω = uki.update_freq, uki.γ_ν, uki.γ_ω
    

#     Σ_ν = γ_ν * uki.Σ_η
#     # update evolution covariance matrix
#     if update_freq > 0 && uki.iter % update_freq == 0
#         Σ_ω = γ_ω * uki.θθ_cov[end]
#     else
#         Σ_ω = uki.Σ_ω 
#     end

#     θ_mean  = uki.θ_mean[end]
#     θθ_cov  = uki.θθ_cov[end]
#     θ_w     = uki.θ_w[end]

#     y = uki.y
#     ############# Prediction step:
#     θ_p_mean  = θ_mean
#     θθ_p_cov = (Σ_ω === nothing ? θθ_cov : θθ_cov + Σ_ω)

#     θ_w_p = θ_w.^(1/(γ_ω + 1)) 
#     for im = 1:N_modes
#         θ_w_p[im] *= det(θθ_cov[im,:,:])^(γ_ω/(2*(γ_ω + 1))) 
#     end
#     θ_w_p /= sum(θ_w_p)

#     @info "θ_w_p = ", θ_w_p
#     ############ Generate sigma points
#     θ_p = construct_sigma_ensemble(uki, θ_p_mean, θθ_p_cov)


#     # @show θ_p
#     # play the role of symmetrizing the covariance matrix
#     θθ_p_cov = construct_cov(uki, θ_p, θ_p_mean)
    

#     ###########  Analysis step
#     g = zeros(FT, N_modes, N_ens, N_y)
    

#     g .= ens_func(θ_p)
    
    
#     g_mean = construct_mean(uki, g)
#     # gg_cov = construct_cov(uki, g, g_mean) + Σ_ν
#     gg_cov = construct_cov(uki, g, g_mean)
#     θg_cov = construct_cov(uki, θ_p, θ_p_mean, g, g_mean)
    
#     tmp = copy(θg_cov)
#     θ_mean_n =  copy(θ_p_mean)
#     θθ_cov_n = copy(θθ_p_cov)
#     θ_w_n = copy(θ_w)

#     for im = 1:N_modes

#         tmp[im, :, :] = θg_cov[im, :, :] / (gg_cov[im, :, :] + Σ_ν)

#         θ_mean_n[im, :] =  θ_p_mean[im, :] + tmp[im, :, :]*(y - g_mean[im, :])

#         θθ_cov_n[im, :, :] =  θθ_p_cov[im, :, :] - tmp[im, :, :]*θg_cov[im, :, :]' 

#         z = y - g_mean[im, :] - θg_cov[im, :, :]' * (θθ_cov[im, :, :] \θ_mean[im, :])

#         @info "z = ", z, " y = ", y, " g_mean[im, :] = ", g_mean[im, :]
#         # θ_w_n[im] = θ_w[im] * sqrt(det(θθ_cov_n[im, :, :]) / det(θθ_cov[im, :, :])) * 
#         #                       exp( 1/2*( θ_mean_n[im, :]'*(θθ_cov_n[im, :, :]\θ_mean_n[im, :]) -
#         #                       θ_mean[im, :]'*(θθ_cov[im, :, :]\θ_mean[im, :]) - 
#         #                       z'*(Σ_ν\z)) )

        
#         θ_w_n[im] = 1/2*( θ_mean_n[im, :]'*(θθ_cov_n[im, :, :]\θ_mean_n[im, :]) -
#                               θ_mean[im, :]'*(θθ_cov[im, :, :]\θ_mean[im, :]) - 
#                               z'*(Σ_ν\z))
#         @info " weight = ", θ_mean_n[im, :]'*(θθ_cov_n[im, :, :]\θ_mean_n[im, :]),  θ_mean[im, :]'*(θθ_cov[im, :, :]\θ_mean[im, :]), z'*(Σ_ν\z)
#         @info "mode: im ", im, " weight = ", θ_w_n[im], " z = ", z
#     end

#     _, maxindx =  findmax(θ_w_n + log.(θ_w_p))
#     θ_w_n .-= θ_w_n[maxindx]
#     for im = 1:N_modes
#         θ_w_n[im] = exp.(θ_w_n[im])
#         @info "before 1 θ_w_n[im]     = ", im,  θ_w_n[im], θ_w_n

#         θ_w_n[im] *= θ_w_p[im] * θ_w_n[im] * sqrt(det(θθ_cov_n[im, :, :]) / det(θθ_cov[im, :, :])) 
#         @info θθ_cov_n[im, :, :], θθ_cov[im, :, :], det(θθ_cov_n[im, :, :]), det(θθ_cov[im, :, :]), sqrt(det(θθ_cov_n[im, :, :]) / det(θθ_cov[im, :, :]))
#         @info "before 2 θ_w_n[im]     = ", im,  θ_w_n[im], θ_w_n
#     end
#     @info "before θ_w      = ", θ_w_n
#     θ_w_n /= sum(θ_w_n)

  
#     @info "θ_w      = ", θ_w_n
#     @info "θ_mean_n = ", θ_mean_n
#     @info "θθ_cov_n = ", θθ_cov_n

#     ########### Save resutls
#     push!(uki.y_pred, g_mean) # N_ens x N_data
#     push!(uki.θ_mean, θ_mean_n) # N_ens x N_params
#     push!(uki.θθ_cov, θθ_cov_n) # N_ens x N_data
#     push!(uki.θ_w, θ_w_n)       # N_ens x N_data
# end



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


# function plot_param_iter(ukiobj::UKIObj{FT, IT}, θ_ref::Array{FT,1}, θ_ref_names::Array{String}) where {FT<:AbstractFloat, IT<:Int}
    
#     θ_mean = ukiobj.θ_mean
#     θθ_cov = ukiobj.θθ_cov
    
#     N_iter = length(θ_mean) - 1
#     ites = Array(LinRange(1, N_iter+1, N_iter+1))
    
#     θ_mean_arr = abs.(hcat(θ_mean...))
    
    
#     N_θ = length(θ_ref)
#     θθ_std_arr = zeros(Float64, (N_θ, N_iter+1))
#     for i = 1:N_iter+1
#         for j = 1:N_θ
#             θθ_std_arr[j, i] = sqrt(θθ_cov[i][j,j])
#         end
#     end
    
#     for i = 1:N_θ
#         errorbar(ites, θ_mean_arr[i,:], yerr=3.0*θθ_std_arr[i,:], fmt="--o",fillstyle="none", label=θ_ref_names[i])   
#         plot(ites, fill(θ_ref[i], N_iter+1), "--", color="gray")
#     end
    
#     xlabel("Iterations")
#     legend()
#     tight_layout()
# end


# function plot_opt_errors(ukiobj::UKIObj{FT, IT}, 
#     θ_ref::Union{Array{FT,1}, Nothing} = nothing, 
#     transform_func::Union{Function, Nothing} = nothing) where {FT<:AbstractFloat, IT<:Int}
    
#     θ_mean = ukiobj.θ_mean
#     θθ_cov = ukiobj.θθ_cov
#     y_pred = ukiobj.y_pred
#     Σ_η = ukiobj.Σ_η
#     y = ukiobj.y

#     N_iter = length(θ_mean) - 1
    
#     ites = Array(LinRange(1, N_iter, N_iter))
#     N_subfigs = (θ_ref === nothing ? 2 : 3)

#     errors = zeros(Float64, N_subfigs, N_iter)
#     fig, ax = PyPlot.subplots(ncols=N_subfigs, figsize=(N_subfigs*6,6))
#     for i = 1:N_iter
#         errors[N_subfigs - 1, i] = 0.5*(y - y_pred[i])'*(Σ_η\(y - y_pred[i]))
#         errors[N_subfigs, i]     = norm(θθ_cov[i])
        
#         if N_subfigs == 3
#             errors[1, i] = norm(θ_ref - (transform_func === nothing ? θ_mean[i] : transform_func(θ_mean[i])))/norm(θ_ref)
#         end
        
#     end

#     markevery = max(div(N_iter, 10), 1)
#     ax[N_subfigs - 1].plot(ites, errors[N_subfigs - 1, :], linestyle="--", marker="o", fillstyle="none", markevery=markevery)
#     ax[N_subfigs - 1].set_xlabel("Iterations")
#     ax[N_subfigs - 1].set_ylabel("Optimization error")
#     ax[N_subfigs - 1].grid()
    
#     ax[N_subfigs].plot(ites, errors[N_subfigs, :], linestyle="--", marker="o", fillstyle="none", markevery=markevery)
#     ax[N_subfigs].set_xlabel("Iterations")
#     ax[N_subfigs].set_ylabel("Frobenius norm of the covariance")
#     ax[N_subfigs].grid()
#     if N_subfigs == 3
#         ax[1].set_xlabel("Iterations")
#         ax[1].plot(ites, errors[1, :], linestyle="--", marker="o", fillstyle="none", markevery=markevery)
#         ax[1].set_ylabel("L₂ norm error")
#         ax[1].grid()
#     end
    
# end







# ##### Linear Test
# mutable struct Setup_Param{MAT, IT<:Int}
#     θ_names::Array{String,1}
#     G::MAT
#     N_θ::IT
#     N_y::IT
# end

# function Setup_Param(G, N_θ::IT, N_y::IT) where {IT<:Int}
#     return Setup_Param(["θ"], G, N_θ, N_y)
# end


# function forward(s_param::Setup_Param, θ::Array{FT, 1}) where {FT<:AbstractFloat}
#     G = s_param.G 
#     return G * θ
# end

# function forward_aug(s_param::Setup_Param, θ::Array{FT, 1}) where {FT<:AbstractFloat}
#     G = s_param.G 
#     return [G * θ; θ]
# end


# function Two_Param_Linear_Test(problem_type::String, θ0_bar, θθ0_cov)
    
#     N_θ = length(θ0_bar)

    
#     if problem_type == "under-determined"
#         # under-determined case
#         θ_ref = [0.6, 1.2]
#         G = [1.0 2.0;]
        
#         y = [3.0;]
#         Σ_η = Array(Diagonal(fill(0.1^2, size(y))))
        
        
#     elseif problem_type == "well-determined"
#         # over-determined case
#         θ_ref = [1.0, 1.0]
#         G = [1.0 2.0; 3.0 4.0]
        
#         y = [3.0;7.0]
#         Σ_η = Array(Diagonal(fill(0.1^2, size(y))))
        
#     elseif problem_type == "over-determined"
#         # over-determined case
#         θ_ref = [1/3, 17/12.0]
#         G = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        
#         y = [3.0;7.0;10.0]
#         Σ_η = Array(Diagonal(fill(0.1^2, size(y))))
        
        
#     else
#         error("Problem type : ", problem_type, " has not implemented!")
#     end
    
#     Σ_post = inv(G'*(Σ_η\G) + inv(θθ0_cov))
#     θ_post = θ0_bar + Σ_post*(G'*(Σ_η\(y - G*θ0_bar)))
    

#     return θ_post, Σ_post, G, y, Σ_η, θ_ref
# end


# function construct_cov(x::Array{FT,2}) where {FT<:AbstractFloat}
    
#     x_mean = dropdims(mean(x, dims=1), dims=1)
#     N_ens, N_x = size(x)
    
#     x_cov = zeros(FT, N_x, N_x)
    
#     for i = 1: N_ens
#         x_cov .+= (x[i,:] - x_mean)*(x[i,:] - x_mean)'
#     end
    
#     return x_cov/(N_ens - 1)
# end


# N_θ = 2
# N_modes = 2
# θ0_mean = zeros(Float64, N_modes, N_θ)
# θθ0_cov = zeros(Float64, N_modes, N_θ, N_θ) 

# θ0_mean[1, :]    = zeros(Float64, N_θ)
# θθ0_cov[1, :, :] = Array(Diagonal(fill(0.1^2, N_θ)))
# θ0_mean[2, :]    = ones(Float64, N_θ)
# θθ0_cov[2, :, :] = Array(Diagonal(fill(1.0^2, N_θ)))
# θ0_w  = [0.5 ; 0.5]

# prior_mean, prior_cov = zeros(Float64, N_θ), Array(Diagonal(fill(1.0^2, N_θ)))

# FT = Float64
# uki_objs = Dict()
# mean_errors = Dict()

# Random.seed!(123)
# α_reg = 1.0
# update_freq = 1
# γ = 1.0
# N_iter = 30
# problem_type  = "under-determined" # "well-determined", "over-determined"
    
# θ_post, Σ_post, G, y, Σ_η, θ_ref = Two_Param_Linear_Test(problem_type, prior_mean, prior_cov)

# N_y = length(y)

# s_param = Setup_Param(G, N_θ, N_y)
# s_param_aug = Setup_Param(G, N_θ, N_y+N_θ)

# y_aug = [y ; prior_mean]
# Σ_η_aug = [Σ_η zeros(Float64, N_y, N_θ); zeros(Float64, N_θ, N_y)  prior_cov]

# # UKI

# uki_obj    = GMUKI_Run(s_param_aug, forward_aug, θ0_w, θ0_mean, θθ0_cov, y_aug, Σ_η_aug, γ, update_freq, N_iter; unscented_transform="modified-2n+1")
# uki_errors = zeros(FT, N_iter+1, 2)

# @info uki_obj.θ_w[end]
# @info uki_obj.θ_mean[end]
# @info uki_obj.θθ_cov[end]
# @info θ_post, Σ_post

# # for i = 1:N_iter+1
# #     uki_errors[i, 1] = norm(uki_obj.θ_mean[i] .- θ_post)/norm(θ_post)
# #     uki_errors[i, 2] = norm(uki_obj.θθ_cov[i] .- Σ_post)/norm(Σ_post)
    
# #     uki_2np1_errors[i, 1] = norm(uki_2np1_obj.θ_mean[i] .- θ_post)/norm(θ_post)
# #     uki_2np1_errors[i, 2] = norm(uki_2np1_obj.θθ_cov[i] .- Σ_post)/norm(Σ_post)
        
# #     eki_errors[i, 1] = norm(dropdims(mean(eki_obj.θ[i], dims=1), dims=1) .- θ_post)/norm(θ_post)
# #     eki_errors[i, 2] = norm(construct_cov(eki_obj.θ[i]) .- Σ_post)/norm(Σ_post)
    
# #     eaki_errors[i, 1] = norm(dropdims(mean(eaki_obj.θ[i], dims=1), dims=1) .- θ_post)/norm(θ_post)
# #     eaki_errors[i, 2] = norm(construct_cov(eaki_obj.θ[i]) .- Σ_post)/norm(Σ_post)
    
# #     etki_errors[i, 1] = norm(dropdims(mean(etki_obj.θ[i], dims=1), dims=1) .- θ_post)/norm(θ_post)
# #     etki_errors[i, 2] = norm(construct_cov(etki_obj.θ[i]) .- Σ_post)/norm(Σ_post)
    
# #     eks_errors[i, 1] = norm(dropdims(mean(eks_obj.θ[i], dims=1), dims=1) .- θ_post)/norm(θ_post)
# #     eks_errors[i, 2] = norm(construct_cov(eks_obj.θ[i]) .- Σ_post)/norm(Σ_post)
    
# #     cbs_errors[i, 1] = norm(dropdims(mean(cbs_obj.θ[i], dims=1), dims=1) .- θ_post)/norm(θ_post)
# #     cbs_errors[i, 2] = norm(construct_cov(cbs_obj.θ[i]) .- Σ_post)/norm(Σ_post)
    
# #     ###################
# #     iukf_errors[i, 1] = norm(iukf_obj.θ_mean[i] .- θ_post)/norm(θ_post)
# #     iukf_errors[i, 2] = norm(iukf_obj.θθ_cov[i] .- Σ_post)/norm(Σ_post)
    
# #     iukf_2np1_errors[i, 1] = norm(iukf_2np1_obj.θ_mean[i] .- θ_post)/norm(θ_post)
# #     iukf_2np1_errors[i, 2] = norm(iukf_2np1_obj.θθ_cov[i] .- Σ_post)/norm(Σ_post)
    
# #     iekf_errors[i, 1] = norm(dropdims(mean(iekf_obj.θ[i], dims=1), dims=1) .- θ_post)/norm(θ_post)
# #     iekf_errors[i, 2] = norm(construct_cov(iekf_obj.θ[i]) .- Σ_post)/norm(Σ_post)
    
# #     ietkf_errors[i, 1] = norm(dropdims(mean(ietkf_obj.θ[i], dims=1), dims=1) .- θ_post)/norm(θ_post)
# #     ietkf_errors[i, 2] = norm(construct_cov(ietkf_obj.θ[i]) .- Σ_post)/norm(Σ_post)
    
# #     ieakf_errors[i, 1] = norm(dropdims(mean(ieakf_obj.θ[i], dims=1), dims=1) .- θ_post)/norm(θ_post)
# #     ieakf_errors[i, 2] = norm(construct_cov(ieakf_obj.θ[i]) .- Σ_post)/norm(Σ_post)
    
# #     ieakf_false_errors[i, 1] = norm(dropdims(mean(ieakf_false_obj.θ[i], dims=1), dims=1) .- θ_post)/norm(θ_post)
# #     ieakf_false_errors[i, 2] = norm(construct_cov(ieakf_false_obj.θ[i]) .- Σ_post)/norm(Σ_post)
    
# # end

# # ites = Array(0:N_iter)


# # markevery = 5

# # fig, ax = PyPlot.subplots(nrows = 2, ncols=2, sharex=false, sharey="row", figsize=(14,9))
# # ax[1,1].semilogy(ites, uki_errors[:, 1],   "-.x", color = "C0", fillstyle="none", label="UKI-1 (J=$(N_θ+2))", markevery = markevery)
# # ax[1,1].semilogy(ites, uki_2np1_errors[:, 1],   "-o", color = "C0", fillstyle="none", label="UKI-2 (J=$(2*N_θ+1))", markevery = markevery)
# # ax[1,1].semilogy(ites, eki_errors[:, 1], "-s", color = "C1", fillstyle="none", label="EKI (J=$N_ens)", markevery = markevery)
# # ax[1,1].semilogy(ites, eaki_errors[:, 1], "-^", color = "C2", fillstyle="none", label="EAKI (J=$N_ens)", markevery = markevery)
# # ax[1,1].semilogy(ites, etki_errors[:, 1], "-d", color = "C3", fillstyle="none", label="ETKI (J=$N_ens)", markevery = markevery)
# # ax[1,1].semilogy(ites, eks_errors[:, 1], "-*", color = "C4", fillstyle="none", label="EKS (J=$N_ens)", markevery = markevery)
# # ax[1,1].semilogy(ites, cbs_errors[:, 1], "-v", color = "C5", fillstyle="none", label="CBS (J=$N_ens)", markevery = markevery)
# # ax[1,1].set_xlabel("Iterations")
# # ax[1,1].set_ylabel("Rel. mean error")
# # ax[1,1].grid("on")
# # ax[1,1].legend(bbox_to_anchor=(1.0, 1.0))

# # ax[1,2].semilogy(ites, iukf_errors[:, 1],   "-.x", color = "C0", fillstyle="none", label="IUKF-1 (J=$(N_θ+2))", markevery = markevery)
# # ax[1,2].semilogy(ites, iukf_2np1_errors[:, 1],   "-o", color = "C0", fillstyle="none", label="IUKF-2 (J=$(2*N_θ+1))", markevery = markevery)
# # ax[1,2].semilogy(ites, iekf_errors[:, 1], "-s", color = "C1", fillstyle="none", label="IEnKF (J=$N_ens)", markevery = markevery)
# # ax[1,2].semilogy(ites, ieakf_errors[:, 1], "-^", color = "C2", fillstyle="none", label="IEAKF (J=$N_ens)", markevery = markevery)
# # ax[1,2].semilogy(ites, ietkf_errors[:, 1], "-d", color = "C3", fillstyle="none", label="IETKF (J=$N_ens)", markevery = markevery)
# # # Initialization
# # ax[1,2].semilogy(ites, ieakf_false_errors[:, 1], "-*", color = "C4", fillstyle="none", label="IEAKF* (J=$N_ens)", markevery = markevery)

# # ax[1,2].set_xlabel("Iterations")
# # #ax[1,2].set_ylabel("Rel. mean error")
# # ax[1,2].grid("on")
# # ax[1,2].legend(bbox_to_anchor=(1.0, 1.0))

# # ax[2,1].semilogy(ites, uki_errors[:, 2],   "-.x", color = "C0", fillstyle="none", label="UKI-1 (J=$(N_θ+2))", markevery = markevery)
# # ax[2,1].semilogy(ites, uki_2np1_errors[:, 2],   "-o", color = "C0", fillstyle="none", label="UKI-2 (J=$(2*N_θ+1))", markevery = markevery)
# # ax[2,1].semilogy(ites, eki_errors[:, 2], "-s", color = "C1", fillstyle="none", label="EKI (J=$N_ens)", markevery = markevery)
# # ax[2,1].semilogy(ites, eaki_errors[:, 2], "-^", color = "C2", fillstyle="none", label="EAKI (J=$N_ens)", markevery = markevery)
# # ax[2,1].semilogy(ites, etki_errors[:, 2], "-d", color = "C3", fillstyle="none", label="ETKI (J=$N_ens)", markevery = markevery)
# # ax[2,1].semilogy(ites, eks_errors[:, 2], "-*", color = "C4", fillstyle="none", label="EKS (J=$N_ens)", markevery = markevery)
# # ax[2,1].semilogy(ites, cbs_errors[:, 2], "-v", color = "C5", fillstyle="none", label="CBS (J=$N_ens)", markevery = markevery)
# # ax[2,1].set_xlabel("Iterations")
# # ax[2,1].set_ylabel("Rel. covariance error")
# # ax[2,1].grid("on")
# # ax[2,1].legend(bbox_to_anchor=(1.0, 1.0))


# # ax[2,2].semilogy(ites, iukf_errors[:, 2],   "-.x", color = "C0", fillstyle="none", label="IUKF-1 (J=$(N_θ+2))", markevery = markevery)
# # ax[2,2].semilogy(ites, iukf_2np1_errors[:, 2],   "-o", color = "C0", fillstyle="none", label="IUKF-2 (J=$(2*N_θ+1))", markevery = markevery)
# # ax[2,2].semilogy(ites, iekf_errors[:, 2], "-s", color = "C1", fillstyle="none", label="IEnKF (J=$N_ens)", markevery = markevery)
# # ax[2,2].semilogy(ites, ieakf_errors[:, 2], "-^", color = "C2", fillstyle="none", label="IEAKF (J=$N_ens)", markevery = markevery)
# # ax[2,2].semilogy(ites, ietkf_errors[:, 2], "-d", color = "C3", fillstyle="none", label="IETKF (J=$N_ens)", markevery = markevery)
# # # Initialization
# # ax[2,2].semilogy(ites, ieakf_false_errors[:, 2], "-*", color = "C4", fillstyle="none", label="IEAKF* (J=$N_ens)", markevery = markevery)
# # ax[2,2].set_xlabel("Iterations")
# # #ax[2,2].set_ylabel("Rel. covariance error")
# # ax[2,2].grid("on")
# # ax[2,2].legend(bbox_to_anchor=(1.0, 1.0))
# # fig.suptitle("Linear 2-Parameter Problem : " * problem_type)

# # fig.tight_layout()
