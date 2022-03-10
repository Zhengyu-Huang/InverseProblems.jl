using Roots
"""
GACTUKIObj{FT<:AbstractFloat, IT<:Int}
Struct that is used in Unscented Kalman Inversion (EKI)
"""
mutable struct CBSObj{FT<:AbstractFloat, IT<:Int}
    "vector of parameter names"
    θ_names::Array{String, 1}
    "a vector of arrays of size N_ensemble x N_parameters containing the parameters (in each EKI iteration a new array of paramet\
ers is added)"
    θ::Vector{Array{FT, 2}}
    "a vector of arrays of size N_ensemble x N_y containing the predicted observation (in each CTUKI iteration a new array of predicted observation is added)"
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
    "current iteration number"
    iter::IT
    "prior mean covariance"
    r_0::Array{FT, 1}
    Σ_0::Array{FT, 2}
    "method parameters"
    α::FT
end



"""
CBSObj Constructor
parameter_names::Vector{String} : parameter name vector
θ0::Array{FT} : prior particles
y::Vector{FT} : observation
Σ_η::Array{FT, 2} : observation covariance
α_reg::Float64 : regularization parameter toward θ0 (0 < α_reg <= 1), default should be 1, without regulariazion
"""
function CBSObj(
    θ_names::Array{String, 1},
    N_ens::IT,
    θ0::Array{FT, 2},
    y::Vector{FT}, # observation
    Σ_η::Array{FT, 2},
    r_0::Array{FT, 1},
    Σ_0::Array{FT, 2},
    α::FT) where {FT<:AbstractFloat, IT<:Int}

    # ensemble size
    N_ens, N_θ = size(θ0)
    # parameters
    θ = Array{FT, 2}[] # array of Array{FT, 2}'s
    push!(θ, θ0) # insert parameters at end of array (in this case just 1st entry)
    # observations

    N_y = size(y, 1)
    y_pred = Array{FT, 1}[]  # array of Array{FT, 2}'s

    CBSObj{FT,IT}(θ_names, θ, y_pred, y, Σ_η, N_ens, N_θ, N_y, 0, r_0, Σ_0,  α)
end



"""
construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
Construct the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_cov(cbs::CBSObj{FT}, x::Array{FT,2}, x_bar::Array{FT}, y::Array{FT,2}, y_bar::Array{FT}) where {FT<:AbstractFloat}
    N_ens, N_x, N_y = cbs.N_ens, size(x_bar,1), size(y_bar,1)

    xy_cov = zeros(FT, N_x, N_y)

    for i = 1: N_ens
        xy_cov .+= (x[i,:] - x_bar)*(y[i,:] - y_bar)'
    end

    return xy_cov/(N_ens - 1)
 end
    

function update_ensemble!(cbs::CBSObj{FT}, ens_func::Function) where {FT<:AbstractFloat}


    # θ: N_ens x N_θ
    N_ens, N_θ, N_y = cbs.N_ens, cbs.N_θ, cbs.N_y
    
    θ = copy(cbs.θ[end])

    
    Σ_η = cbs.Σ_η
    Σ_0 = cbs.Σ_0
    r_0 = cbs.r_0
    α = cbs.α
    y = cbs.y
    
    
    ############# Analysis step
    
    # evaluation G(θ)
    g = zeros(FT, N_ens, N_y)
    g .= ens_func(θ)

    f = zeros(FT, N_ens)

    for i = 1:N_ens
        f[i] = 0.5 * (y - g[i, :])' * (Σ_η\(y - g[i, :])) + 0.5 * (θ[i, :] - r_0)' * (Σ_0\(θ[i, :] - r_0))
    end

    # adaptive + sampling
    β = solve_β(f)
    λ = 1.0/(1.0 + β)

    Zβ = FT(0)
    Mβ = zeros(FT, N_θ)
    for i = 1:N_ens
        Zβ += exp(-β*f[i])
        Mβ += exp(-β*f[i]) * θ[i,:]
    end
    Mβ /= Zβ
    Cβ = zeros(FT, N_θ, N_θ)


    for i = 1:N_ens
        Cβ += exp(-β*f[i]) / Zβ  * ( θ[i,:] - Mβ ) * ( θ[i,:] - Mβ )'
    end
    
    noise = rand(MvNormal(Matrix(Hermitian(Cβ))), N_ens)'
    for i = 1:N_ens
        θ[i,:] = Mβ + α*(θ[i,:] - Mβ) + sqrt((1 - α^2)/λ) * noise[i, :]
    end
    
    
    # Save results
    push!(cbs.θ, θ) # N_ens x N_θ
    
end


function CBS_Run(
    s_param, 
    forward::Function, 
    θ0_mean, θθ0_cov,
    N_ens,
    y, Σ_η,
    Nt,
    α;
    exact_init = false)
    
    θ_names = s_param.θ_names
    
    


    θ0 = Array(rand(MvNormal(θ0_mean, θθ0_cov), N_ens)')
    N_θ = length(θ0_mean)
    if exact_init
        # shift mean to 0
        θ0 -= (ones(N_ens) * dropdims(mean(θ0, dims=1), dims=1)')

        # #θ0_new = θ0 * X with covariance θθ0_cov
        U1, S1, V1t = svd(θ0)
        U2, S2, U2t = svd((N_ens - 1.0)*θθ0_cov)
        
        θ0 = θ0 * (V1t * (S1 .\ sqrt.(S2) .* U2t'))
   
        # #add θ0_mean
        θ0 += (ones(N_ens) * θ0_mean')

        
    end


    cbsobj = CBSObj(
        θ_names,
        N_ens,
        θ0, 
        y,
        Σ_η,
        θ0_mean, 
        θθ0_cov,
        α
    )
    
    ens_func(θ_ens) = ensemble(s_param, θ_ens, forward) 

    for i = 1:Nt
        update_ensemble!(cbsobj, ens_func) 
    end
    
    return cbsobj
    
end


function solve_β(f, η = 0.5)

    f = f .- minimum(f)

    function g(β)
        ω = exp.(-β*f)
        J = sum(ω)^2 / sum(ω.^2) - η*length(f)
        return J
    end
    low, up = 0.0, 1.0

    while g(up) > 0.0
        low = up
        up *= 2.0
    end

    return find_zero(g, (low, up), Roots.Bisection())
end