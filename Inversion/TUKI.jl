using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions


"""
TUKIObj{FT<:AbstractFloat, IT<:Int}
Struct that is used in Unscented Kalman Inversion (EKI)
"""
mutable struct TUKIObj{FT<:AbstractFloat, IT<:Int}
    "vector of parameter names (never used)"
     θ_names::Array{String,1}
    "a vector of arrays of size N_ensemble x N_parameters containing the mean of the parameters (in each uki iteration a new array of mean is added)"
     θ_mean::Vector{Array{FT, 1}}
     "a vector of arrays of size N_ensemble x (N_parameters x N_parameters) containing the covariance of the parameters (in each uki iteration a new array of cov is added)"
     θθ_cov_sqrt::Vector{Array{FT, 2}}
     "a vector of arrays of size N_ensemble x N_y containing the predicted observation (in each uki iteration a new array of predicted observation is added)"
     y_pred::Vector{Array{FT, 1}}
     "vector of observations (length: N_y)"
     y::Array{FT, 1}
     "covariance of the observational noise"
     Σ_η
     "number ensemble size (2N_θ - 1)"
     N_r::IT
     "size of θ"
     N_θ::IT
     "size of y"
     N_y::IT
     "weights in UKI"
     c_weights::Array{FT, 1}
     mean_weights::Array{FT, 1}
     cov_weights::Array{FT, 1}
     "square root of the covariance of the artificial evolution error"
     Z_ω::Array{FT, 2}
     "covariance of the artificial observation error"
     Σ_ν::Array{FT, 2}
     "regularization parameter"
     α_reg::FT
     "regularization vector"
     r::Array{FT, 1}
     "update frequency"
     update_freq::IT
     "current iteration number"
     iter::IT
end



"""
TUKIObj Constructor 
parameter_names::Vector{String} : parameter name vector
θ0_mean::Array{FT} : prior mean
θθ0_cov::Array{FT, 2} : prior covariance
y::Vector{FT} : observation 
obs_cov::Array{FT, 2} : observation covariance
α_reg::Float64 : regularization parameter toward θ0 (0 < α_reg <= 1), default should be 1, without regulariazion
"""
function TUKIObj(θ_names::Vector{String},
    θ0_mean::Array{FT}, 
    θθ0_cov_sqrt::Array{FT, 2},
    y::Array{FT,1},
    Σ_η,
    α_reg::FT,
    update_freq::IT;
    modified_uscented_transform::Bool = true) where {FT<:AbstractFloat, IT<:Int}

    # ensemble size
    N_θ = size(θ0_mean,1)
    N_y = size(y, 1)
    N_r = size(θθ0_cov_sqrt, 2)
    N_ens = 2*N_r+1


 
    c_weights = zeros(FT, N_θ)
    mean_weights = zeros(FT, N_ens)
    cov_weights = zeros(FT, N_ens)

    # TODO parameters λ, α, β
    # α, β = 1.0e-3, 2.0
    κ = 0.0
    β = 2.0
    α = min(sqrt(4/(N_r + κ)), 1.0)
    λ = α^2*(N_r + κ) - N_r

    c_weights[1:N_r]     .=  sqrt(N_r + λ)
    mean_weights[1] = λ/(N_r + λ)
    mean_weights[2:N_ens] .= 1/(2(N_r + λ))
    cov_weights[1] = λ/(N_r + λ) + 1 - α^2 + β
    cov_weights[2:N_ens] .= 1/(2(N_r + λ))

    Z_ω, Σ_ν =  sqrt(2-α_reg^2)*θθ0_cov_sqrt, 2*Σ_η

    if modified_uscented_transform
        mean_weights[1] = 1.0
        mean_weights[2:N_ens] .= 0.0
    end


    θ_mean = Array{FT}[]  # array of Array{FT, 2}'s
    push!(θ_mean, θ0_mean) # insert parameters at end of array (in this case just 1st entry)
    θθ_cov_sqrt = Array{FT,2}[] # array of Array{FT, 2}'s
    push!(θθ_cov_sqrt, θθ0_cov_sqrt) # insert parameters at end of array (in this case just 1st entry)

    y_pred = Array{FT}[]   # array of Array{FT, 2}'s


    r = θ0_mean
    
    iter = 0

    
    TUKIObj{FT,IT}(θ_names, θ_mean, θθ_cov_sqrt, y_pred, 
    y,   Σ_η, 
    N_r, N_θ, N_y, 
    c_weights, mean_weights, cov_weights, 
    Z_ω, Σ_ν, α_reg, r, 
    update_freq, iter)

end


"""
construct_sigma_ensemble
Construct the sigma ensemble, based on the mean x_mean, and covariance x_cov
"""
function construct_sigma_ensemble(tuki::TUKIObj{FT}, x_mean::Array{FT}, xx_cov_sqrt::Array{FT,2}) where {FT<:AbstractFloat}
    N_r = tuki.N_r
    N_x = size(x_mean, 1)
    c_weights = tuki.c_weights

    
    # use svd decomposition
    svd_xx_cov_sqrt = svd(xx_cov_sqrt)

    # @info "energy : ", sum(svd_xx_cov.S[1:N_r])/sum(svd_xx_cov.S)
    # @info "svd_xx_cov_sqr.S : ", svd_xx_cov_sqr.S

    # @info "svd_xx_cov.U : ", svd_xx_cov.U
    # @info "x_cov : ", x_cov

    x = zeros(Float64, 2*N_r+1, N_x)
    x[1, :] = x_mean
    for i = 1: N_r
        x[i+1,     :] = x_mean + c_weights[i]*svd_xx_cov_sqrt.S[i]*svd_xx_cov_sqrt.U[:, i]
        x[i+1+N_r, :] = x_mean - c_weights[i]*svd_xx_cov_sqrt.S[i]*svd_xx_cov_sqrt.U[:, i]
    end



    # chol_xx_cov = cholesky(Hermitian(x_cov)).L
    # x = zeros(Float64, 2*N_r+1, N_x)
    # x[1, :] = x_mean
    # for i = 1: N_r
    #     x[i+1,     :] = x_mean + c_weights[i]*chol_xx_cov[:,i]
    #     x[i+1+N_r, :] = x_mean - c_weights[i]*chol_xx_cov[:,i]
    # end

    return x
end


"""
construct_mean x_mean from ensemble x
"""
function construct_mean(tuki::TUKIObj{FT}, x::Array{FT,2}) where {FT<:AbstractFloat}
    N_ens, N_x = size(x)

    @assert(2*tuki.N_r + 1 == N_ens)

    x_mean = zeros(Float64, N_x)

    mean_weights = tuki.mean_weights

    
    for i = 1: N_ens
        x_mean += mean_weights[i]*x[i,:]
    end

    return x_mean
end

"""
construct_cov xx_cov from ensemble x and mean x_mean
"""
function construct_cov_sqrt(tuki::TUKIObj{FT}, x::Array{FT,2}, x_mean::Array{FT}) where {FT<:AbstractFloat}
    N_r, N_x = tuki.N_r, size(x_mean,1)
    
    cov_weights = tuki.cov_weights

    xx_cov_sqrt = zeros(FT, N_x, 2N_r)

    for i = 2: 2N_r+1
        xx_cov_sqrt[:, i-1] .= sqrt(cov_weights[i])*(x[i,:] - x_mean)
        
    end

    return xx_cov_sqrt
end





"""
update TRUKI struct
ens_func: The function g = G(θ)
define the function as 
    ens_func(θ_ens) = MyG(phys_params, θ_ens, other_params)
use G(θ_mean) instead of FG(θ)
"""
function update_ensemble!(tuki::TUKIObj{FT}, ens_func::Function) where {FT}
    
    tuki.iter += 1
    # update evolution covariance matrix
    if tuki.update_freq > 0 && tuki.iter%tuki.update_freq == 0
        tuki.Z_ω = sqrt(2 - uki.α_reg^2)*tuki.θθ_cov_sqrt[end]
    end


    θ_mean  =  tuki.θ_mean[end]
    θθ_cov_sqrt =  tuki.θθ_cov_sqrt[end]


    α_reg = tuki.α_reg
    Z_ω, Σ_ν = tuki.Z_ω, tuki.Σ_ν
    r = tuki.r
    N_θ, N_y, N_ens = tuki.N_θ, tuki.N_y, 2*tuki.N_r+1
    ############# Prediction step
    # Generate sigma points, and time step update 
    
    
    

    θ_p_mean  = α_reg*θ_mean + (1-α_reg)*r
    θθ_p_cov_sqrt = [α_reg*θθ_cov_sqrt  Z_ω]
    


    
    ############# Generate sigma points
    
    θ_p = construct_sigma_ensemble(tuki, θ_p_mean, θθ_p_cov_sqrt)
    
    #@show  norm(θ_p_mean_ - θ_p_mean), norm(θθ_p_cov - θθ_p_cov_)
    #@info "θθ_p_cov_: ", diag(θθ_p_cov_)
    
    g = zeros(FT, N_ens, N_y)
    g .= ens_func(θ_p)
    g_mean = construct_mean(tuki, g)

    # @info "θ_p: ", θ_p

    Z = construct_cov_sqrt(tuki, θ_p, θ_p_mean)
    Y = construct_cov_sqrt(tuki, g, g_mean)

    X = Y' * (Σ_ν\Y)
    svd_X = svd(X)

    P, Γ = svd_X.U, svd_X.S
    # θg_cov = construct_cov(tuki, θ_p, θ_p_mean, g, g_mean)

    #Kalman Gain
    # K = θg_cov/gg_cov
    # use G(θ_mean) instead of FG(θ)

    θθ_cov_sqrt =  Z * P ./ sqrt.(Γ .+ 1.0)' 

    θ_mean =  θ_p_mean + θθ_cov_sqrt./ sqrt.(Γ .+ 1.0)'  * (P' * (Y' * (Σ_ν\(tuki.y - g_mean))))

    # K = θθ'G' (Gθθ'G' + Σ_ν)G(θ_ref - θ)
    # @info norm(tuki.y - g[1,:]), norm(K*(tuki.y - g[1,:]))
    
    

    # @info " gg_cov ", diag(gg_cov)
    # @info " θg_cov ", diag(θg_cov)
    # @info "θ_mean : ", θ_mean
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


    # store new parameters (and observations  G(θ_mean) instead of FG(θ))
    push!(tuki.y_pred, g_mean) # N_ens x N_data
    push!(tuki.θ_mean, θ_mean) # N_ens x N_params
    push!(tuki.θθ_cov_sqrt, θθ_cov_sqrt) # N_ens x N_data

    

end


function TUKI_Run(s_param, forward::Function, 
    θ0_mean, θθ0_cov,
    y, Σ_η,
    α_reg,
    update_freq,
    N_iter;
    modified_uscented_transform::Bool = true,
    θ_basis = nothing)
    
    θ_names = s_param.θ_names
    
    ens_func(θ_ens) = run_linear_ensemble(θ_ens, G)
    
    tukiobj = TUKIObj(θ_names,
    θ0_mean, 
    θθ0_cov_sqrt,
    y, # observation
    Σ_η,
    α_reg,
    update_freq;
    modified_uscented_transform = modified_uscented_transform)
    
    
    ens_func(θ_ens) = ensemble(s_param, θ_ens, forward)
    
    for i in 1:N_iter
        
        update_ensemble!(tukiobj, ens_func) 
        
    end
    
    return tukiobj
    
end