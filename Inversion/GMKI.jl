using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions


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

function Gaussian_density_helper(θ_mean::Array{FT,1}, θθ_cov::Array{FT,2}, θ::Array{FT,1}) where {FT<:AbstractFloat}
    N_θ = size(θ_mean,1)
    
    return exp( -1/2*((θ - θ_mean)'* (θθ_cov\(θ - θ_mean)) )) / ( sqrt(det(θθ_cov)) )

end


function Gaussian_density(θ_mean::Array{FT,1}, θθ_cov::Array{FT,2}, θ::Array{FT,1}) where {FT<:AbstractFloat}
    N_θ = size(θ_mean,1)
    
    return exp( -1/2*((θ - θ_mean)'* (θθ_cov\(θ - θ_mean)) )) / ( (2π)^(N_θ/2)*sqrt(det(θθ_cov)) )

end
function Gaussian_mixture_density_helper(θ_w::Array{FT,1}, θ_mean::Array{FT,2}, θθ_cov::Array{FT,3}, θ::Array{FT,1}) where {FT<:AbstractFloat}
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
            # TODO 
            # θ_mean_p[i,:] = θ_mean[i, :] + 5(θ_mean_p[i,:] - θ_mean[i, :])
            θ_w_p[i] = sum(ws)/N_ens
            
    
            # @show i, θ_mean[i,:], θ_mean_p[i,:], θ_w[i], θ_w_p[i]
        end
        
        
    elseif method == "random-sampling"
        
        N_ens = 1000
        α = sqrt((N_ens-1)/2.0)
        xs = zeros(N_ens, N_θ)
        ws = zeros(N_ens)
        for i = 1:N_modes
            
            # construct sigma points
            method1 = false

            if method1
                Random.seed!(123);
                xs .= rand(MvNormal(θ_mean[i, :], θθ_cov[i,:,:]), N_ens)'
                
                for j = 1:N_ens
                    ws[j] = θ_w[i]*Gaussian_mixture_density(θ_w, θ_mean, θθ_cov, xs[j, :])^(αpower - 1)
                end
                

                θ_mean_p[i,:] = ws' * xs / sum(ws)
                θ_w_p[i] = sum(ws)/N_ens
            else

                Random.seed!(123);
                chol_xx_cov = cholesky(Hermitian(θθ_cov[i,:,:]/αpower)).L
                xs .= (θ_mean[i, :] .+ chol_xx_cov*rand(Normal(0, 1), N_θ, N_ens))'
                
                for j = 1:N_ens
                    # ws[j] = (Gaussian_density(θ_mean[i,:], θθ_cov[i,:,:], xs[j, :])/Gaussian_mixture_density(ones(N_modes), θ_mean, θθ_cov, xs[j, :]))^(1 - αpower)
                
                    ws[j] = (θ_w[i]*Gaussian_density_helper(θ_mean[i,:], θθ_cov[i,:,:], xs[j, :])/Gaussian_mixture_density_helper(θ_w, θ_mean, θθ_cov, xs[j, :]))^(1 - αpower)
                end
                
                # if any(isnan,ws) || sum(ws) < 1.0
                #     continue
                # end
                

                θ_mean_p[i,:] = ws' * xs / sum(ws)

                # @info "i  ws :", ws
                # @info "i  θ_mean_p :", θ_mean_p[i,:]


                # θ_mean_p[i,:] = θ_mean[i,:] + 0.9*(θ_mean_p[i,:] - θ_mean[i,:])
                # θθ_cov_p[i,:,:] = (xs - ones(N_ens)*θ_mean_p[i,:]')' * (Diagonal(ws)/sum(ws)) * (xs - ones(N_ens)*θ_mean_p[i,:]')  
                θ_w_p[i] = θ_w[i]^αpower * det(θθ_cov[i,:,:])^((1-αpower)/2) * sum(ws)/N_ens
            end

            # @show "mode : ", θ_w
            # @show "mode : ", i, θθ_cov[i,:,:], θ_mean[i,:]
            # @show "mode : ", i, θ_w[i], θ_w_p[i], sum(ws)/N_ens
    
            
        end

    elseif method == "Taylor-expansion"
        Ĉ_ji = zeros(N_modes, N_θ, N_θ)
        m̂_ji = zeros(N_modes, N_θ)
        ŵ_ji = zeros(N_modes)
        w̃_ji = zeros(N_modes)
        for i = 1:N_modes

            if θ_w[i] < 1e-6
                continue
            end

            for j = 1:N_modes
                Ĉ_ji[j,:,:] = inv( inv(θθ_cov[j,:,:]) + αpower*inv(θθ_cov[i,:,:]) )
            end
            for iter = 1:1
                for j = 1:N_modes
                    m̂_ji[j,:] = Ĉ_ji[j,:,:]*(θθ_cov[j,:,:]\θ_mean_p[i, :] + αpower*(θθ_cov[i,:,:]\θ_mean[j, :]))
                    # @info Ĉ_ji[j,:,:], θθ_cov[j,:,:]\θ_mean_p[i, :] , αpower*θθ_cov[i,:,:]\θ_mean[j, :], αpower
                    # @info "i, j = ", i, j
                    # @info "mi, mj, mij",θ_mean_p[i, :],  θ_mean[j, :],  m̂_ji[j,:]
                    # @info "Ci, Cj, Cij",θθ_cov[i,:,:],  θθ_cov[j,:,:],  Ĉ_ji[j,:,:]
                    # @info -αpower/2*(θ_mean_p[i, :]'*(θθ_cov[i,:,:]\θ_mean_p[i, :])), 
                    #       1/2*(θ_mean[j, :]'*(θθ_cov[j,:,:]\θ_mean[j, :])),
                    #       1/2*(m̂_ji[j,:]'*(Ĉ_ji[j,:,:]\m̂_ji[j,:]))
                    w̃_ji[j] = exp(-αpower/2*(θ_mean_p[i, :]'*(θθ_cov[i,:,:]\θ_mean_p[i, :])) - 
                                  1/2*(θ_mean[j, :]'*(θθ_cov[j,:,:]\θ_mean[j, :])) +
                                  1/2*(m̂_ji[j,:]'*(Ĉ_ji[j,:,:]\m̂_ji[j,:]))) *
                                  sqrt(det(Ĉ_ji[j,:,:])/det(θθ_cov[j,:,:])/det(θθ_cov[i,:,:]))
                end

                temp_deno = (αpower*θ_w[i]*w̃_ji[i]  + (1-αpower)*sum(θ_w.*w̃_ji))
                θ_w_p[i] = θ_w[i]^(αpower + 1)/det(θθ_cov[i,:,:])^(αpower/2)/temp_deno
                
                # ÂΔm̂ = b̂
                temp_Â = θ_w[i]*w̃_ji[i]/(1 + αpower)*I
                temp_b̂ = zeros(N_θ) 
                @info "temp_Â, temp_b̂ = ", temp_Â, temp_b̂
                for j = 1:N_modes
                    if j != i
                        temp_Â +=        (1-αpower)*θ_w[j]*w̃_ji[j]* (Ĉ_ji[j,:,:]/θθ_cov[j,:,:])
                        temp_b̂ -= αpower*(1-αpower)*θ_w[j]*w̃_ji[j]* (Ĉ_ji[j,:,:]*(θθ_cov[i,:,:]\(θ_mean[j, :] - θ_mean[i, :])))
                    end
                end
                
                Δθ_mean = temp_Â\temp_b̂
                @info "iter = ", iter, "θ_w[i] = ", θ_w[i], " θ_mean[i, :] = ", θ_mean[i, :], "Δθ_mean = ", Δθ_mean
                θ_mean_p[i, :] = θ_mean[i, :] + Δθ_mean


                

                # ####
                # temp_Â = αpower*θ_w[i]*w̃_ji[i]* (Ĉ_ji[i,:,:]/θθ_cov[i,:,:])
                # temp_b̂ = temp_deno*θ_mean[i, :] - αpower*θ_w[i]*w̃_ji[i]*(Ĉ_ji[i,:,:]*(θθ_cov[i,:,:]\θ_mean[i, :]))

                # @info "temp_Â, temp_b̂ = ", temp_Â, temp_b̂

                # for j = 1:N_modes
                #     temp_Â += (1-αpower)*θ_w[j]*w̃_ji[j]*(Ĉ_ji[j,:,:]/θθ_cov[j,:,:])
                #     temp_b̂ -= αpower*(1-αpower)*θ_w[j]*w̃_ji[j]* (Ĉ_ji[j,:,:]*(θθ_cov[i,:,:]\θ_mean[j,:,:]))
                # end

                # @info "θ_w[i] = ", θ_w[i], " temp_Â, temp_b̂ = ", temp_Â, temp_b̂
                # @info "??? ", θ_mean_p[i, :] , temp_Â\temp_b̂
                # θ_mean_p[i, :] = temp_Â\temp_b̂
                                
        
                # @info w̃_ji
                # @info "Mode ", i, " Iteration: ", iter, θ_w[i], θ_w_p[i], m̂_ji[i,:], θ_mean_p[i, :]
            end
            # @show i, θθ_cov[i,:,:], θ_mean[i,:], θ_mean_p[i,:], θ_w[i], θ_w_p[i], m̂_ji[i,:]
        end

    else
        @error("method :", method, " has not implemented")
        
    end
    θ_w_p ./ sum(θ_w_p)
    
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
    
    #TODO push mean
    # θ_mean_all = mean(θ_mean, dims=1)
    # θ_p_mean  = ones(N_modes)*θ_mean_all + sqrt(1 +  0.2)*(θ_mean - ones(N_modes)*θ_mean_all)
    
    # θ_p_mean  = θ_mean
    # θθ_p_cov = (Σ_ω === nothing ? θθ_cov : θθ_cov + Σ_ω)
    # logθ_w_p = (1/(γ_ω + 1))  * logθ_w
    # for im = 1:N_modes
    #     logθ_w_p[im] += (γ_ω/(2*(γ_ω + 1)))*log(det(θθ_cov[im,:,:]))
    # end

    ##TODO
    logθ_w_p, θ_p_mean, θθ_p_cov = Gaussian_mixture_power(exp.(logθ_w), θ_mean, θθ_cov, 1/(γ_ω + 1); method="random-sampling")
    logθ_w_p = log.(logθ_w_p)
    # @info "prediction step  : θ_mean = ", θ_mean, " θ_p_mean = ", θ_p_mean
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


    # @show "After prediction logθ_w_p = ", logθ_w_p

    for im = 1:N_modes

        tmp[im, :, :] = θg_cov[im, :, :] / (gg_cov[im, :, :] + Σ_ν)

        θ_mean_n[im, :] =  θ_p_mean[im, :] + tmp[im, :, :]*(y - g_mean[im, :])

        θθ_cov_n[im, :, :] =  θθ_p_cov[im, :, :] - tmp[im, :, :]*θg_cov[im, :, :]' 

        z = y - g_mean[im, :]
        # temp = θθ_cov[im, :, :]\(θg_cov[im, :, :]*(Σ_ν\z))
        # logθ_w_n[im] = 1/2*( temp'*θθ_cov_n[im, :, :]*temp -  z'*(Σ_ν\z))

        logθ_w_n[im] = 1/2*( (θ_mean_n[im, :] - θ_p_mean[im, :])'*(θθ_cov_n[im, :, :]\(θ_mean_n[im, :] - θ_p_mean[im, :])) -  z'*(Σ_ν\z))
    end

    # @show logθ_w_n
    for im = 1:N_modes
        logθ_w_n[im] += logθ_w_p[im] + log(sqrt(det(θθ_cov_n[im, :, :]) / det(θθ_cov[im, :, :])))
    end

    logθ_w_n .-= maximum(logθ_w_n)
    logθ_w_n .-= log( sum(exp.(logθ_w_n)) )

    # TODO 
    logθ_w_n[logθ_w_n .< -10] .= -10

    # TODO reweight
    REWEIGHT = false
    if REWEIGHT
        g_mean_n = ens_func(θ_mean_n)
        p_n = zeros(N_modes)
        for i = 1:N_modes
            p_n[i] = exp(-1.0/2.0* ((y - g_mean_n[i,:])'*(uki.Σ_η\(y - g_mean_n[i,:]))) )
        end
        θ_w_n = reweight(exp.(logθ_w_n), θ_mean_n, θθ_cov_n, θ_mean_n, p_n)
        logθ_w_n .= log.(θ_w_n)
    end

    # @show "Iteration logθ_w_n = ", logθ_w_n

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


