using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions

# generate ensemble
Random.seed!(123)

function MvNormal_sqrt(N_ens, θ_mean::Array{FT,1}, θθ_cov_sqrt) where {FT<:AbstractFloat}
    
    
    N_θ, N_r = size(θθ_cov_sqrt)
    θ = zeros(FT, N_ens, N_θ)
    for i = 1: N_ens
        θ[i,     :] = θ_mean + θθ_cov_sqrt * rand(Normal(0, 1), N_r)
    end
    
    return θ
end
"""
EKIObj{FT<:AbstractFloat, IT<:Int}
Structure that is used in Ensemble Kalman Inversion (EKI)
#Fields
$(DocStringExtensions.FIELDS)
"""
mutable struct EKIObj{FT<:AbstractFloat, IT<:Int}
    "filter_type type"
    filter_type::String
    "vector of parameter names"
    θ_names::Array{String, 1}
    "a vector of arrays of size N_ensemble x N_parameters containing the parameters (in each EKI iteration a new array of parameters is added)"
    θ::Vector{Array{FT, 2}}
    "a vector of arrays of size N_ensemble x N_y containing the predicted observation (in each uki iteration a new array of predicted observation is added)"
    y_pred::Vector{Array{FT, 1}}
    "vector of observations (length: N_y)"
    y::Array{FT, 1}
    "covariance of the observational noise"
    Σ_η
    "number ensemble size (2N_θ - 1)"
    N_ens::IT
    "size of θ"
    N_θ::IT
    "size of y"
    N_y::IT
    "Covariance matrix square root of the evolution error"
    Z_ω::Union{Array{FT, 2}, Nothing}
    "time step"
    Δt::FT
    "regularization parameter"
    α_reg::FT
    "regularization vector"
    r::Array{FT, 1}
    "update frequency"
    update_freq::IT
    "current iteration number"
    iter::IT
end

# outer constructors
function EKIObj(
    filter_type::String,
    θ_names::Array{String, 1},
    N_ens::IT,
    # initial condition
    θ0_mean::Array{FT},
    θθ0_cov_sqrt::Array{FT,2},
    # prior information
    prior_mean::Array{FT},
    prior_cov_sqrt::Array{FT,2},
    y::Array{FT, 1},
    Σ_η,
    Δt::FT,
    α_reg::FT = 1.0,
    update_freq::IT = 0) where {FT<:AbstractFloat, IT<:Int}
    
    ## check EKI hyperparameters
    Z_ω = sqrt(Δt/(1 - Δt) + 1 -  α_reg^2) * prior_cov_sqrt

    if update_freq > 0
        @assert(Δt > 0.0 && Δt < 1)
        @info "Start ", filter_type, " on the mean-field stochastic dynamical system for Bayesian inference "
        @assert(α_reg ≈ 1.0)

    elseif Δt ≈ 1.0 
        @info "Start original ", filter_type, " for optimization "
        @assert(α_reg ≈ 1.0 && update_freq == 0)
        Z_ω = nothing

    else
        @assert(Δt > 0.0 && Δt < 1)
        @info "Start ", filter_type, " on the regularized stochastic dynamical system for optimization "
        @assert(α_reg >= 0.0 && α_reg <= 1.0)
    end
    
    N_θ = size(θ0_mean,1)
    N_y = size(y, 1)
    
    
    # generate initial assemble
    θ = Array{FT, 2}[] # array of Array{FT, 2}'s
    θ0 = MvNormal_sqrt(N_ens, θ0_mean, θθ0_cov_sqrt)
    push!(θ, θ0) # insert parameters at end of array (in this case just 1st entry)
    
    # prediction
    y_pred = Array{FT, 1}[]  # array of Array{FT, 2}'s
    
    r = prior_mean
    iter = 0
    


    EKIObj{FT,IT}(filter_type, 
    θ_names, θ, 
    y_pred, 
    y, Σ_η, 
    N_ens, N_θ, N_y, 
    Z_ω, Δt, α_reg, r, update_freq, iter)
end

"""
construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
Construct the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_cov(eki::EKIObj{FT}, x::Array{FT,2}, x_mean::Array{FT, 1}, y::Array{FT,2}, y_mean::Array{FT, 1}) where {FT<:AbstractFloat}
    N_ens, N_x, N_y = eki.N_ens, size(x_mean,1), size(y_mean,1)
    
    xy_cov = zeros(FT, N_x, N_y)
    
    for i = 1: N_ens
        xy_cov .+= (x[i,:] - x_mean)*(y[i,:] - y_mean)'
    end
    
    return xy_cov/(N_ens - 1)
end


"""
construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
Construct the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_cov_sqrt(eki::EKIObj{FT}, x::Array{FT,2}) where {FT<:AbstractFloat}
    N_ens, N_x = eki.N_ens, size(x,2)
    x_mean = dropdims(mean(x, dims=1), dims=1)
    
    x_cov_sqrt = zeros(FT, N_x, N_ens)
    
    for i = 1: N_ens
        x_cov_sqrt[:, i] .+= (x[i,:] - x_mean)
    end
    
    return x_cov_sqrt/sqrt(N_ens - 1)
end


function trunc_svd(X,  ϵ = 1.0e-6)
    n_row, n_col = size(X)
    svd_X = svd(X)
    rank_X = min(n_row, n_col)
    for i = 1:min(n_row, n_col)
        if svd_X.S[i] <= ϵ*svd_X.S[1]
            rank_X = i - 1
            break
        end
    end

    return svd_X.U[:, 1:rank_X], svd_X.S[1:rank_X], svd_X.Vt[1:rank_X, :]'
end


function update_ensemble!(eki::EKIObj{FT}, ens_func::Function) where {FT<:AbstractFloat}

    eki.iter += 1

    filter_type = eki.filter_type
    N_ens, N_θ, N_y = eki.N_ens, eki.N_θ, eki.N_y
    r, α_reg, update_freq, Δt = eki.r, eki.α_reg, eki.update_freq, eki.Δt
    
    # θ: N_ens x N_θ
    θ = eki.θ[end]
    # compute the observation covariance matrices
    Σ_ν = (1/Δt) * eki.Σ_η
    

    ############# Prediction step
    θ_p = similar(θ)
    # θ mean at previous time step 
    θ_mean = dropdims(mean(θ, dims=1), dims=1)
    if update_freq > 0 && eki.iter % update_freq == 0 # deterministic update for mean-field evoluation
        θ_p_mean =  r + α_reg*(θ_mean  - r) 
        for j = 1:N_ens
            θ_p[j, :] .= θ_p_mean + sqrt(Δt/(1 - Δt) + 1 - α_reg^2 + 1)*(θ[j, :] - θ_mean)
        end

    else # stochastic update 

        # generate evolution error
        noise = (eki.Z_ω === nothing ? zeros(N_ens, N_θ) : MvNormal_sqrt(N_ens, zeros(N_θ), eki.Z_ω))

        for j = 1:N_ens
            θ_p[j, :] .= α_reg*θ[j, :] + (1-α_reg)*r + noise[j, :]
        end
  
    end

    θ_p_mean = dropdims(mean(θ_p, dims=1), dims=1)

    ############# Analysis step
    
    # evaluation G(θ)
    g = zeros(FT, N_ens, N_y)
    g .= ens_func(θ_p)
    g_mean = dropdims(mean(g, dims=1), dims=1)
    
    
    
    # construct square root matrix for  θ̂ - m̂
    Z_p_t = copy(θ_p)
    for j = 1:N_ens;    Z_p_t[j, :] .-=  θ_p_mean;    end
    Z_p_t ./= sqrt(N_ens - 1)
    
    # construct square root matrix for  g - g_mean
    Y_p_t = copy(g)  
    for j = 1:N_ens;  Y_p_t[j, :] .-=  g_mean;  end
    Y_p_t ./= sqrt(N_ens - 1)
    
    #=
    gg_cov = construct_cov(eki, g, g_mean, g, g_mean) + Σ_ν
    θg_cov = construct_cov(eki, θ_p, θ_p_mean, g, g_mean)
    K   = θg_cov * gg_cov⁻¹ 
        = Z_p * Y_p' * (Y_p *Y_p' + Σ_ν)⁻¹ 
        = Z_p * (I + Y_p' * Σ_ν⁻¹ * Y_p)⁻¹ * Y_p' + Σ_ν⁻¹ 
        = Z_p_t' * (I + P * Γ * P')⁻¹ * Y_p_t * Σ_ν⁻¹ 
        = Z_p_t' * P *(I + Γ)⁻¹*P' * Y_p_t * Σ_ν⁻¹ 
    =#                       
    X = Y_p_t/Σ_ν*Y_p_t'

    
    svd_X = svd(X)
    P, Γ = svd_X.U, svd_X.S
    
    # compute the mean for EAKI and ETKI at next time step 
    θ_mean = θ_p_mean + Z_p_t' * (P *( (Γ .+ 1.0) .\ (P' * (Y_p_t * (Σ_ν\(eki.y - g_mean))))))
    
    
    
    if filter_type == "EKI"
        noise = rand(MvNormal(zeros(N_y), Σ_ν), N_ens) 

        θ = copy(θ_p) 
        for j = 1:N_ens
            θ[j,:] += Z_p_t' * (P *( (Γ .+ 1.0) .\ (P' * (Y_p_t * (Σ_ν\((eki.y - g[j, :] - noise[:, j]))))))) # N_ens x N_θ
        end

        
    elseif filter_type == "EAKI"
        
        # update particles by Ensemble Adjustment Kalman Filter
        # Dp = F^T Σ F, Σ = F Dp F^T, Dp is non-singular

       
        
        F, sqrt_D_p, V =  trunc_svd(Z_p_t') 
        
        
        # I + Y_p_t/Σ_ν*Y_p_t' = P (Γ + I) P'
        # Y = V' /(I + Y_p_t/Σ_ν*Y_p_t') * V
        Y = V' * P ./ (Γ .+ 1.0)' * P' * V
        
        svd_Y = svd(Y)
        
        U, D = svd_Y.U, svd_Y.S
        
        
        A = (F .* sqrt_D_p' * U .* sqrt.(D)') * (sqrt_D_p .\ F')
        
        
        θ = similar(θ_p) 
        for j = 1:N_ens
            θ[j, :] .= θ_mean + A * (θ_p[j, :] - θ_p_mean) # N_ens x N_θ
        end
        
        ################# Debug check
        
        # θθ_p_cov = construct_cov(eki, θ_p, θ_p_mean, θ_p, θ_p_mean)
        # θθ_cov = Z_p_t'*(I - Y_p_t/(Y_p_t'*Y_p_t + Σ_ν)*Y_p_t') *Z_p_t
        # θ_mean_debug = dropdims(mean(θ, dims=1), dims=1)
        # θθ_cov_debug = construct_cov(eki, θ, θ_mean_debug, θ, θ_mean_debug)
        # @info "mean error is ", norm(θ_mean - θ_mean_debug), " cov error is ", norm(θθ_cov - A*Z_p_t'*Z_p_t*A'), norm(θθ_cov - θθ_cov_debug)
        
        
    elseif filter_type == "ETKI"
        
        
        # update particles by Ensemble Adjustment Kalman Filter
        # Dp = F^T Σ F, Σ = F Dp F^T, Dp is non-singular
        # X = Y_p_t/Σ_ν*Y_p_t'
        # svd_X = svd(X)
        
        # P, Γ = svd_X.U, svd_X.S
        
        #Original ETKF is  T = P * (Γ .+ 1)^{-1/2}, but it is biased
        T = P ./ sqrt.(Γ .+ 1)' * P'
        
        # Z_p'
        θ = similar(θ_p) 
        for j = 1:N_ens;  θ[j, :] .=  θ_p[j, :] - θ_p_mean;  end
        # Z' = （Z_p * T)' = T' * Z_p
        θ .= T' * θ 
        for j = 1:N_ens;  θ[j, :] .+=  θ_mean;  end
        
        
        ################# Debug check
        
        # Z_p_t = copy(θ_p)
        # for j = 1:N_ens;    Z_p_t[j, :] .-=  θ_p_mean;    end
        # Z_p_t ./= sqrt(N_ens - 1)
        
        
        # θθ_p_cov = construct_cov(eki, θ_p, θ_p_mean, θ_p, θ_p_mean)
        # θθ_cov = Z_p_t'*(I - Y_p_t/(Y_p_t'*Y_p_t + Σ_ν)*Y_p_t') *Z_p_t
        # θ_mean_debug = dropdims(mean(θ, dims=1), dims=1)
        # θθ_cov_debug = construct_cov(eki, θ, θ_mean_debug, θ, θ_mean_debug)
        # @info "mean error is ", norm(θ_mean - θ_mean_debug), " cov error is ", norm(θθ_cov - Z_p_t'*T*T'*Z_p_t), norm(θθ_cov - θθ_cov_debug)
        
        
    else
        error("Filter type :", filter_type, " has not implemented yet!")
    end
    

    
    # Save results
    push!(eki.θ, θ) # N_ens x N_θ
    push!(eki.y_pred, g_mean)
    
end

# the evolution error covariance is (Δt/(1-Δt) + 1 - α^2) * C_n or  (Δt/(1-Δt) + 1 - α^2) * C_0
# the observation error covariance is Σ_ν = (1/Δt) * Σ_η
function EKI_Run(s_param, forward::Function, 
    filter_type,
    θ0_mean, θθ0_cov_sqrt,
    prior_mean, prior_cov_sqrt,
    N_ens,
    y, Σ_η,
    Δt,
    α_reg,
    update_freq,
    N_iter)
    

    θ_names = s_param.θ_names
    

    ekiobj = EKIObj(filter_type ,
    θ_names,
    N_ens,
    θ0_mean, θθ0_cov_sqrt,
    prior_mean, prior_cov_sqrt,
    y, Σ_η,
    Δt,
    α_reg, update_freq)
    
    
    ens_func(θ_ens) = ensemble(s_param, θ_ens, forward) 
    
    
    for i in 1:N_iter
        update_ensemble!(ekiobj, ens_func) 
    end
    
    return ekiobj
    
end





"""
UKIObj{FT<:AbstractFloat, IT<:Int}
Struct that is used in Unscented Kalman Inversion (UKI)
For solving the inverse problem 
    y = G(θ) + η
    
"""
mutable struct UKIObj{FT<:AbstractFloat, IT<:Int}
"vector of parameter names (never used)"
    θ_names::Array{String,1}
"a vector of arrays of size N_ensemble x N_parameters containing the mean of the parameters (in each uki iteration a new array of mean is added)"
    θ_mean::Vector{Array{FT, 1}}
    "a vector of arrays of size N_ensemble x (N_parameters x N_parameters) containing the covariance of the parameters (in each uki iteration a new array of cov is added)"
    θθ_cov::Vector{Array{FT, 2}}
    "a vector of arrays of size N_ensemble x N_y containing the predicted observation (in each uki iteration a new array of predicted observation is added)"
    y_pred::Vector{Array{FT, 1}}
    "vector of observations (length: N_y)"
    y::Array{FT, 1}
    "covariance of the observational noise"
    Σ_η
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
    "time step"
    Δt::FT
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
UKIObj Constructor 
parameter_names::Array{String,1} : parameter name vector
θ0_mean::Array{FT} : prior mean
θθ0_cov::Array{FT, 2} : prior covariance
g_t::Array{FT,1} : observation 
obs_cov::Array{FT, 2} : observation covariance
α_reg::FT : regularization parameter toward θ0 (0 < α_reg <= 1), default should be 1, without regulariazion

unscented_transform : "original-2n+1", "modified-2n+1", "original-n+2", "modified-n+2" 
"""
function UKIObj(θ_names::Array{String,1},
                # initial condition
                θ0_mean::Array{FT}, 
                θθ0_cov::Array{FT, 2},
                # prior information
                prior_mean::Array{FT},
                prior_cov::Array{FT,2},
                y::Array{FT,1},
                Σ_η,
                Δt::FT,
                α_reg::FT,
                update_freq::IT;
                unscented_transform::String = "modified-2n+1") where {FT<:AbstractFloat, IT<:Int}

    ## check UKI hyperparameters
    Σ_ω = (Δt/(1 - Δt) + 1 -  α_reg^2) * prior_cov
    
    if update_freq > 0 
        @assert(Δt > 0.0 && Δt < 1)
        @info "Start UKI on the mean-field stochastic dynamical system for Bayesian inference "
        @assert(α_reg ≈ 1.0)

    elseif Δt ≈ 1.0 
        @info "Start original UKI for optimization "
        @assert(α_reg ≈ 1.0 && update_freq == 0)
        Σ_ω = nothing

    else
        @assert(Δt > 0.0 && Δt < 1)
        @info "Start UKI on the (regularized) stochastic dynamical system for optimization "
        @assert(α_reg >= 0.0 && α_reg <= 1.0)
        
    end



    N_θ = size(θ0_mean,1)
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

    

    θ_mean = Array{FT,1}[]  # array of Array{FT, 2}'s
    push!(θ_mean, θ0_mean) # insert parameters at end of array (in this case just 1st entry)
    θθ_cov = Array{FT,2}[] # array of Array{FT, 2}'s
    push!(θθ_cov, θθ0_cov) # insert parameters at end of array (in this case just 1st entry)

    y_pred = Array{FT, 1}[]  # array of Array{FT, 2}'s
   
    r = prior_mean
    iter = 0

    UKIObj{FT,IT}(θ_names, θ_mean, θθ_cov, y_pred, 
                  y,   Σ_η, 
                  N_ens, N_θ, N_y, 
                  c_weights, mean_weights, cov_weights, 
                  Σ_ω, Δt, α_reg, r, 
                  update_freq, iter)

end


"""
construct_sigma_ensemble
Construct the sigma ensemble, based on the mean x_mean, and covariance x_cov
"""
function construct_sigma_ensemble(uki::UKIObj{FT, IT}, x_mean::Array{FT}, x_cov::Array{FT,2}) where {FT<:AbstractFloat, IT<:Int}
    N_ens = uki.N_ens
    N_x = size(x_mean,1)
    @assert(N_ens == 2*N_x+1 || N_ens == N_x+2)

    c_weights = uki.c_weights


    chol_xx_cov = cholesky(Hermitian(x_cov)).L

    if ndims(c_weights) == 1
        x = zeros(FT, N_ens, N_x)
        x[1, :] = x_mean
        for i = 1: N_x
            x[i+1,     :] = x_mean + c_weights[i]*chol_xx_cov[:,i]
            x[i+1+N_x, :] = x_mean - c_weights[i]*chol_xx_cov[:,i]
        end
    elseif ndims(c_weights) == 2
        x = zeros(FT, N_ens, N_x)
        x[1, :] = x_mean
        for i = 2: N_x + 2
	    # @info chol_xx_cov,  c_weights[:, i],  chol_xx_cov * c_weights[:, i]
            x[i,     :] = x_mean + chol_xx_cov * c_weights[:, i]
        end
    else
        error("c_weights dimensionality error")
    end
    return x
end


"""
construct_mean x_mean from ensemble x
"""
function construct_mean(uki::UKIObj{FT, IT}, x::Array{FT,2}) where {FT<:AbstractFloat, IT<:Int}
    N_ens, N_x = size(x)

    @assert(uki.N_ens == N_ens)

    x_mean = zeros(FT, N_x)

    mean_weights = uki.mean_weights

    
    for i = 1: N_ens
        x_mean += mean_weights[i]*x[i,:]
    end

    return x_mean
end

"""
construct_cov xx_cov from ensemble x and mean x_mean
"""
function construct_cov(uki::UKIObj{FT, IT}, x::Array{FT,2}, x_mean::Array{FT}) where {FT<:AbstractFloat, IT<:Int}
    N_ens, N_x = uki.N_ens, size(x_mean,1)
    
    cov_weights = uki.cov_weights

    xx_cov = zeros(FT, N_x, N_x)

    for i = 1: N_ens
        xx_cov .+= cov_weights[i]*(x[i,:] - x_mean)*(x[i,:] - x_mean)'
    end

    return xx_cov
end

"""
construct_cov xy_cov from ensemble x and mean x_mean, ensemble y and mean y_mean
"""
function construct_cov(uki::UKIObj{FT, IT}, x::Array{FT,2}, x_mean::Array{FT}, y::Array{FT,2}, y_mean::Array{FT}) where {FT<:AbstractFloat, IT<:Int}
    N_ens, N_x, N_y = uki.N_ens, size(x_mean,1), size(y_mean,1)
    
    cov_weights = uki.cov_weights

    xy_cov = zeros(FT, N_x, N_y)

    for i = 1: N_ens
        xy_cov .+= cov_weights[i]*(x[i,:] - x_mean)*(y[i,:] - y_mean)'
    end

    return xy_cov
end



"""
update uki struct
ens_func: The function g = G(θ)
define the function as 
    ens_func(θ_ens) = MyG(phys_params, θ_ens, other_params)
use G(θ_mean) instead of FG(θ)
"""
function update_ensemble!(uki::UKIObj{FT, IT}, ens_func::Function) where {FT<:AbstractFloat, IT<:Int}
    
    uki.iter += 1

    N_θ, N_y, N_ens = uki.N_θ, uki.N_y, uki.N_ens
    r, α_reg, update_freq, Δt = uki.r, uki.α_reg, uki.update_freq, uki.Δt
    

    Σ_ν = (1/Δt) * uki.Σ_η
    # update evolution covariance matrix
    if update_freq > 0 && uki.iter % update_freq == 0
        Σ_ω = (Δt/(1 - Δt) + 1 - α_reg^2) * uki.θθ_cov[end]
    else
        Σ_ω = uki.Σ_ω 
    end

    θ_mean  = uki.θ_mean[end]
    θθ_cov = uki.θθ_cov[end]
    y = uki.y

    
    
    ############# Prediction step:
    θ_p_mean  = α_reg*θ_mean + (1-α_reg)*r
    θθ_p_cov = (Σ_ω === nothing ? α_reg^2*θθ_cov : α_reg^2*θθ_cov + Σ_ω)
    

    # @show θθ_cov, θθ_p_cov
    ############ Generate sigma points
    θ_p = construct_sigma_ensemble(uki, θ_p_mean, θθ_p_cov)


    # @show θ_p
    # play the role of symmetrizing the covariance matrix
    θθ_p_cov = construct_cov(uki, θ_p, θ_p_mean)
    

    ###########  Analysis step
    g = zeros(FT, N_ens, N_y)
    
    
    # @info "θ_p = ", θ_p
    g .= ens_func(θ_p)
    
    g_mean = construct_mean(uki, g)
    gg_cov = construct_cov(uki, g, g_mean) + Σ_ν
    θg_cov = construct_cov(uki, θ_p, θ_p_mean, g, g_mean)

    tmp = θg_cov / gg_cov

    θ_mean =  θ_p_mean + tmp*(y - g_mean)

    θθ_cov =  θθ_p_cov - tmp*θg_cov' 
    

    ########### Save resutls
    push!(uki.y_pred, g_mean) # N_ens x N_data
    push!(uki.θ_mean, θ_mean) # N_ens x N_params
    push!(uki.θθ_cov, θθ_cov) # N_ens x N_data
end


"""
update uki struct
ens_func: The function g = G(θ)
define the function as 
    ens_func(θ_ens) = MyG(phys_params, θ_ens, other_params)
use G(θ_mean) instead of FG(θ)
"""
function prediction_ensemble(uki::UKIObj{FT, IT}) where {FT<:AbstractFloat, IT<:Int}
    
    uki.iter += 1

    N_θ, N_y, N_ens = uki.N_θ, uki.N_y, uki.N_ens
    r, α_reg, update_freq, Δt = uki.r, uki.α_reg, uki.update_freq, uki.Δt
    

    Σ_ν = (1/Δt) * uki.Σ_η
    # update evolution covariance matrix
    if update_freq > 0 && uki.iter % update_freq == 0
        Σ_ω = (Δt/(1 - Δt) + 1 - α_reg^2) * uki.θθ_cov[end]
    else
        Σ_ω = uki.Σ_ω 
    end

    θ_mean  = uki.θ_mean[end]
    θθ_cov = uki.θθ_cov[end]
    y = uki.y

    
    
    ############# Prediction step:
    θ_p_mean  = α_reg*θ_mean + (1-α_reg)*r
    θθ_p_cov = (Σ_ω === nothing ? α_reg^2*θθ_cov : α_reg^2*θθ_cov + Σ_ω)
    

    # @show θθ_cov, θθ_p_cov
    ############ Generate sigma points
    θ_p = construct_sigma_ensemble(uki, θ_p_mean, θθ_p_cov)


    # @show θ_p
    # play the role of symmetrizing the covariance matrix
    θθ_p_cov = construct_cov(uki, θ_p, θ_p_mean)
    
    
    return θ_p, θθ_p_cov
    
end





"""
update uki struct
ens_func: The function g = G(θ)
define the function as 
    ens_func(θ_ens) = MyG(phys_params, θ_ens, other_params)
use G(θ_mean) instead of FG(θ)
"""
function update_ensemble!(uki::UKIObj{FT, IT}, g::Array{FT,2}) where {FT<:AbstractFloat, IT<:Int}
    
    uki.iter += 1

    N_θ, N_y, N_ens = uki.N_θ, uki.N_y, uki.N_ens
    r, α_reg, update_freq, Δt = uki.r, uki.α_reg, uki.update_freq, uki.Δt
    

    Σ_ν = (1/Δt) * uki.Σ_η
    # update evolution covariance matrix
    if update_freq > 0 && uki.iter % update_freq == 0
        Σ_ω = (Δt/(1 - Δt) + 1 - α_reg^2) * uki.θθ_cov[end]
    else
        Σ_ω = uki.Σ_ω 
    end

    θ_mean  = uki.θ_mean[end]
    θθ_cov = uki.θθ_cov[end]
    y = uki.y

    
    
    ############# Prediction step:
    θ_p_mean  = α_reg*θ_mean + (1-α_reg)*r
    θθ_p_cov = (Σ_ω === nothing ? α_reg^2*θθ_cov : α_reg^2*θθ_cov + Σ_ω)
    

    # @show θθ_cov, θθ_p_cov
    ############ Generate sigma points
    θ_p = construct_sigma_ensemble(uki, θ_p_mean, θθ_p_cov)


    # @show θ_p
    # play the role of symmetrizing the covariance matrix
    θθ_p_cov = construct_cov(uki, θ_p, θ_p_mean)
    

    
    g_mean = construct_mean(uki, g)
    gg_cov = construct_cov(uki, g, g_mean) + Σ_ν
    θg_cov = construct_cov(uki, θ_p, θ_p_mean, g, g_mean)

    tmp = θg_cov / gg_cov

    θ_mean =  θ_p_mean + tmp*(y - g_mean)

    θθ_cov =  θθ_p_cov - tmp*θg_cov' 
    

    ########### Save resutls
    push!(uki.y_pred, g_mean) # N_ens x N_data
    push!(uki.θ_mean, θ_mean) # N_ens x N_params
    push!(uki.θθ_cov, θθ_cov) # N_ens x N_data
end






function UKI_Run(s_param, forward::Function, 
    θ0_mean, θθ0_cov,
    prior_mean, prior_cov,
    y, Σ_η,
    Δt,
    α_reg,
    update_freq,
    N_iter;
    unscented_transform::String = "modified-2n+1",
    θ_basis = nothing)
    
    θ_names = s_param.θ_names
    
    
    ukiobj = UKIObj(θ_names ,
    θ0_mean, θθ0_cov,
    prior_mean, prior_cov,
    y, Σ_η,
    Δt,
    α_reg,
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


function plot_param_iter(ukiobj::UKIObj{FT, IT}, θ_ref::Array{FT,1}, θ_ref_names::Array{String}) where {FT<:AbstractFloat, IT<:Int}
    
    θ_mean = ukiobj.θ_mean
    θθ_cov = ukiobj.θθ_cov
    
    N_iter = length(θ_mean) - 1
    ites = Array(LinRange(1, N_iter+1, N_iter+1))
    
    θ_mean_arr = abs.(hcat(θ_mean...))
    
    
    N_θ = length(θ_ref)
    θθ_std_arr = zeros(Float64, (N_θ, N_iter+1))
    for i = 1:N_iter+1
        for j = 1:N_θ
            θθ_std_arr[j, i] = sqrt(θθ_cov[i][j,j])
        end
    end
    
    for i = 1:N_θ
        errorbar(ites, θ_mean_arr[i,:], yerr=3.0*θθ_std_arr[i,:], fmt="--o",fillstyle="none", label=θ_ref_names[i])   
        plot(ites, fill(θ_ref[i], N_iter+1), "--", color="gray")
    end
    
    xlabel("Iterations")
    legend()
    tight_layout()
end


function plot_opt_errors(ukiobj::UKIObj{FT, IT}, 
    θ_ref::Union{Array{FT,1}, Nothing} = nothing, 
    transform_func::Union{Function, Nothing} = nothing) where {FT<:AbstractFloat, IT<:Int}
    
    θ_mean = ukiobj.θ_mean
    θθ_cov = ukiobj.θθ_cov
    y_pred = ukiobj.y_pred
    Σ_η = ukiobj.Σ_η
    y = ukiobj.y

    N_iter = length(θ_mean) - 1
    
    ites = Array(LinRange(1, N_iter, N_iter))
    N_subfigs = (θ_ref === nothing ? 2 : 3)

    errors = zeros(Float64, N_subfigs, N_iter)
    fig, ax = PyPlot.subplots(ncols=N_subfigs, figsize=(N_subfigs*6,6))
    for i = 1:N_iter
        errors[N_subfigs - 1, i] = 0.5*(y - y_pred[i])'*(Σ_η\(y - y_pred[i]))
        errors[N_subfigs, i]     = norm(θθ_cov[i])
        
        if N_subfigs == 3
            errors[1, i] = norm(θ_ref - (transform_func === nothing ? θ_mean[i] : transform_func(θ_mean[i])))/norm(θ_ref)
        end
        
    end

    markevery = max(div(N_iter, 10), 1)
    ax[N_subfigs - 1].plot(ites, errors[N_subfigs - 1, :], linestyle="--", marker="o", fillstyle="none", markevery=markevery)
    ax[N_subfigs - 1].set_xlabel("Iterations")
    ax[N_subfigs - 1].set_ylabel("Optimization error")
    ax[N_subfigs - 1].grid()
    
    ax[N_subfigs].plot(ites, errors[N_subfigs, :], linestyle="--", marker="o", fillstyle="none", markevery=markevery)
    ax[N_subfigs].set_xlabel("Iterations")
    ax[N_subfigs].set_ylabel("Frobenius norm of the covariance")
    ax[N_subfigs].grid()
    if N_subfigs == 3
        ax[1].set_xlabel("Iterations")
        ax[1].plot(ites, errors[1, :], linestyle="--", marker="o", fillstyle="none", markevery=markevery)
        ax[1].set_ylabel("L₂ norm error")
        ax[1].grid()
    end
    
end







