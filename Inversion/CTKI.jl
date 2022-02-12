"""
GAUKIObj{FT<:AbstractFloat, IT<:Int}
Struct that is used in Unscented Kalman Inversion (EKI)
"""
mutable struct CTKIObj{FT<:AbstractFloat, IT<:Int}
    "vector of parameter names"
    θ_names::Array{String, 1}
    "a vector of arrays of size N_ensemble x N_parameters containing the parameters (in each EKI iteration a new array of paramet\
ers is added)"
    θ::Vector{Array{FT, 2}}
    "a vector of arrays of size N_ensemble x N_y containing the predicted observation (in each uki iteration a new array of predicted observation is added)"
    y_pred::Vector{Array{FT, 2}}
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
    "current iteration number"
    iter::IT
    "number of time steps"
    Nt::IT
    "end time"
    T::FT
end



"""
CTKIObj Constructor
parameter_names::Vector{String} : parameter name vector
θ0::Array{FT} : prior particles
y::Vector{FT} : observation
Σ_η::Array{FT, 2} : observation covariance
α_reg::Float64 : regularization parameter toward θ0 (0 < α_reg <= 1), default should be 1, without regulariazion
"""
function CTKIObj(
    θ_names::Array{String, 1},
    θ0::Array{FT, 2},
    y::Vector{FT}, # observation
    Σ_η::Array{FT, 2},
    Nt::IT,
    T::FT) where {FT<:AbstractFloat, IT<:Int}

    # ensemble size
    N_ens, N_θ = size(θ0)
    # parameters
    θ = Array{FT, 2}[] # array of Array{FT, 2}'s
    push!(θ, θ0) # insert parameters at end of array (in this case just 1st entry)
    # observations

    N_y = size(y, 1)
    y_pred = Array{FT, 1}[]  # array of Array{FT, 2}'s

    CTKIObj{FT,IT}(θ_names, θ, y_pred, y, Σ_η, N_ens, N_θ, N_y, 0, Nt, T)
end



"""
construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
Construct the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_cov(ctki::CTKIObj{FT}, x::Array{FT,2}, x_bar::Array{FT}, y::Array{FT,2}, y_bar::Array{FT}) where {FT<:AbstractFloat}
    N_ens, N_x, N_y = ctki.N_ens, size(x_bar,1), size(y_bar,1)

    xy_cov = zeros(FT, N_x, N_y)

    for i = 1: N_ens
        xy_cov .+= (x[i,:] - x_bar)*(y[i,:] - y_bar)'
    end

    return xy_cov/(N_ens - 1)
 end
    
function update_ensemble!(ctki::CTKIObj{FT}, ens_func::Function) where {FT<:AbstractFloat}
    # θ: N_ens x N_params
    N_ens, N_θ = size(ctki.θ[1])
    N_y = ctki.N_y
    Δt = ctki.T/ctki.Nt
    ############# Prediction
    θ_p = copy(ctki.θ[end])
    θ_p_bar = dropdims(mean(θ_p, dims=1), dims=1)

    ############# Update and Analysis
    
    g = zeros(FT, N_ens, N_y)
    g .= ens_func(θ_p)

    g_bar = dropdims(mean(g, dims=1), dims=1)

    Σ_η = ctki.Σ_η/Δt

    θ = copy(θ_p)
    θg_cov = construct_cov(ctki, θ_p, θ_p_bar, g, g_bar)

    gg_cov = construct_cov(ctki, g, g_bar, g, g_bar) + Σ_η
    tmp = θg_cov/gg_cov
    y = zeros(FT, N_ens, N_y)
    noise = rand(MvNormal(zeros(N_y), Σ_η), N_ens)
    for j = 1:N_ens
        y[j, :] = g[j, :] + noise[:, j]
    end

    for j = 1:N_ens
        θ[j,:] += tmp*(ctki.y - y[j, :]) # N_ens x N_params
    end
        
    # store new parameters (and observations)
    push!(ctki.θ, θ) # N_ens x N_params

end



function CTKI_Run(s_param, 
    forward::Function, 
    θ0_mean, θθ0_cov,
    N_ens,
    y, Σ_η,
    Nt,
    T = 1.0)
    
    θ_names = s_param.θ_names
    
    θ0 = Array(rand(MvNormal(θ0_mean, θθ0_cov), N_ens)')

    ctkiobj = CTKIObj(θ_names,
    θ0, 
    y,
    Σ_η,
    Nt,
    T)
    
    
    ens_func(θ_ens) = ensemble(s_param, θ_ens, forward) 

    for i = 1:Nt
        update_ensemble!(ctkiobj, ens_func) 
    end
    
    return ctkiobj
    
end




