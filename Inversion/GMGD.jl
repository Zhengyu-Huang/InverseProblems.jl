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
    updata_covariance::Bool
    "metric"
    metric::String
    "the method to approximate the expectation in the gradient flow"
    expectation_method::String
    "weights"
    c_weights::Union{Array{FT, 1}, Array{FT, 2}}
    mean_weights::Array{FT, 1}
    cov_weights::Array{FT, 1}
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
    
    if expectation_method == "unscented_transform_original_2n+1" ||  expectation_method == "unscented_transform_modified_2n+1"

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

        if expectation_method == "unscented_transform_modified_2n+1"
            mean_weights[1] = 1.0
            mean_weights[2:N_ens] .= 0.0
        end

    elseif expectation_method == "random_sampling" 
        c_weights = zeros(FT, N_θ)
        mean_weights = ones(FT, N_ens)/N_ens
        cov_weights = ones(FT, N_ens)/(N_ens-1)

    else

        error("expectation_method: ", expectation_method, " is not recognized")
    
    end

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
                  expectation_method,
                  c_weights, mean_weights, cov_weights)

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

    elseif expectation_method == "unscented_transform_modified_2n+1"
        @assert(N_ens == 2*N_x+1)
        c_weights = gmgd.c_weights
        xs = zeros(FT, N_modes, N_ens, N_x)
        for im = 1:N_modes
            chol_xx_cov = cholesky(Hermitian(x_covs[im,:,:])).L
            x_mean = x_means[im, :]
            x = zeros(FT, N_ens, N_x)
            x[1, :] = x_mean
            for i = 1: N_x
                x[i+1,     :] = x_mean + c_weights[i]*chol_xx_cov[:,i]
                x[i+1+N_x, :] = x_mean - c_weights[i]*chol_xx_cov[:,i]
            end
            xs[im, :, :] .= x 
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
    mean_weights = gmgd.mean_weights
    for im = 1:N_modes
        x_means[im, repeat([:],ndims_x)...] = sum(mean_weights.*xs[im, repeat([:],ndims_x+1)...], dims=1)[1, repeat([:],ndims_x)...] 
    end
    return x_means
end

# """
# construct_cov xx_cov from ensemble x and mean x_mean
# """
# function construct_cov(gmgd::GMGDObj{FT, IT}, xs::Array{FT,3}, x_means::Array{FT , 2}) where {FT<:AbstractFloat, IT<:Int}
#     N_modes, N_ens = gmgd.N_modes, gmgd.N_ens
#     N_ens, N_x = gmgd.N_ens, size(x_means[1, :],1)
    
#     cov_weights = gmgd.cov_weights

#     xx_covs = zeros(FT, N_modes, N_x, N_x)

#     for im = 1:N_modes
#         for i = 1: N_ens
#             xx_covs[im, :, :] .+= cov_weights[i]*(xs[im, i, :] - x_means[im, :])*(xs[im, i, :] - x_means[im, :])'
#         end
#     end

#     return xx_covs
# end



function Gaussian_density_helper(θ_mean::Array{FT,1}, θθ_cov::Array{FT,2}, θ::Array{FT,1}) where {FT<:AbstractFloat}
    return exp( -1/2*((θ - θ_mean)'* (θθ_cov\(θ - θ_mean)) )) / ( sqrt(det(θθ_cov)) )
end

function Gaussian_mixture_density_helper(θ_w::Array{FT,1}, θ_mean::Array{FT,2}, θθ_cov::Array{FT,3}, θ::Array{FT,1}) where {FT<:AbstractFloat}
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
            ρ, ∇ρ, ∇²ρ = Gaussian_mixture_density_helper(θ_w, θ_mean, θθ_cov, θ_p[im, i, :])
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

    # @info "θθ_cov_n = ", θθ_cov_n

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
    expectation_method::String = "unscented_transform_modified_2n+1",
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







######################### TEST #######################################


# function gaussain_V(θ::Array{FT, 1}) where {FT<:AbstractFloat}
#     # sample ρ ∝ exp(-θ'Aθ/2)
#     N_θ = length(θ)
#     A = Diagonal(ones(N_θ))
#     V   = θ'*(A*θ)/2
#     ∇V  = A*θ
#     ∇²V = A
#     return V, ∇V, ∇²V
# end


# T = 40000.0
# N_iter = 40000 
# N_modes = 2
# N_θ = 1
# θ0_w  = fill(1.0, N_modes)/N_modes
# θ0_mean, θθ0_cov  = zeros(N_modes, N_θ), zeros(N_modes, N_θ, N_θ)
# σ_0 = 0.1
# Random.seed!(111);
# for m = 1:N_modes
#     θ0_mean[m, :]    .= rand(Normal(0, σ_0), N_θ)
#     θθ0_cov[m, :, :] .= Array(Diagonal(fill(σ_0^2, N_θ)))
# end
# @info "Run GMKI with ", N_modes, " θ0_mean = ", θ0_mean
# metric = "Fisher-Rao"
# update_covariance = true
# expectation_method = "unscented_transform_modified_2n+1"
# N_ens = 2N_θ+1
# # expectation_method = "random_sampling"
# # N_ens = 2
# gmgdobj = GMGD_Run(
#     gaussain_V, 
#     N_iter,
#     T,
#     metric,
#     update_covariance, 
#     θ0_w, θ0_mean, θθ0_cov,
#     expectation_method,
#     N_ens)


   

# Nx = 1000
# θ_min, θ_max = -5.0, 5.0

# # Plot reference
# xx_ref = Array(LinRange(θ_min, θ_max, 1000))
# yy_ref = copy(xx_ref)
# for i = 1:length(xx_ref)
#         V, _, _ = gaussain_V(xx_ref[i:i])
#         yy_ref[i] = exp(-V[1])
# end
# yy_ref .= yy_ref / ( sum(yy_ref)*(xx_ref[2] - xx_ref[1]) )

# using PyPlot
# include("Plot.jl")

# fig, ax = PyPlot.subplots(sharex=false, sharey="row", figsize=(6,6))
# ax.plot(xx_ref, yy_ref, "-s", label="Reference", color="grey", linewidth=2, fillstyle="none", markevery=25)
# iter = N_iter
# Nx = 1000
# xxs, zzs = zeros(N_modes, Nx), zeros(N_modes, Nx)
# θ_min = minimum(gmgdobj.θ_mean[iter][:,1] .- 5sqrt.(gmgdobj.θθ_cov[iter][:,1,1]))
# θ_max = maximum(gmgdobj.θ_mean[iter][:,1] .+ 5sqrt.(gmgdobj.θθ_cov[iter][:,1,1]))            

# for i =1:N_modes
#     xxs[i, :], zzs[i, :] = Gaussian_1d(gmgdobj.θ_mean[iter][i,1], gmgdobj.θθ_cov[iter][i,1,1], Nx, θ_min, θ_max)
#     zzs[i, :] *= exp(gmgdobj.logθ_w[iter][i])
#     ax.plot(fill(gmgdobj.θ_mean[iter][i,1], 11), LinRange(0,1,11), color = "C"*string(i), marker="o", fillstyle="none", markevery=5)
#     ax.plot(xxs[1,:], zzs[i, :], linestyle=":", color = "C"*string(i), fillstyle="none", markevery=100, linewidth=2)
# end
# ax.plot(xxs[1,:], sum(zzs, dims=1)', linestyle="-", fillstyle="none", markevery=100, label="GMGD", linewidth=2)
# ax.legend()



# fig, ax = PyPlot.subplots(sharex=false, sharey="row", figsize=(6,6))
# θ_w = exp.(hcat(gmgdobj.logθ_w...))
# θ_mean = hcat(gmgdobj.θ_mean...)
# for i =1:N_modes
#     ax.plot(Array(1:N_iter), θ_w[i, 1:N_iter], "--", color = "C"*string(i), fillstyle="none", markevery=5, label = (i == 1 ? "weight" : nothing) )
#     ax.plot(Array(1:N_iter), θ_mean[i, 1:N_iter], ":", color = "C"*string(i), fillstyle="none", markevery=5, label= (i == 1 ? "mean" : nothing))
# end
# ax.legend()

# θ_w = exp.(hcat(gmgdobj.logθ_w...))
# for i =1:N_modes
# ax[2, N_modes].plot(Array(1:N_iter), θ_w[i, 1:N_iter], marker=linestyles[i], color = "C"*string(i), fillstyle="none", markevery=5, label= "mode "*string(i))
# end
# ax[2, N_modes].legend()
# ax[2, N_modes].set_xlabel("Iterations")
# end
# ax[1, 1].set_ylabel("Densities")
# ax[2, 1].set_ylabel("Weights")

# fig.tight_layout()
# fig.savefig("1D-density-"*string(σ_η)*"-"*string(iter)*".pdf")




# end