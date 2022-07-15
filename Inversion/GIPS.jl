using LinearAlgebra
using Random
using Distributions
using EllipsisNotation
using PyPlot
"""
GIPSObj{FT<:AbstractFloat, IT<:Int}
Structure that is used in Gaussian Interaction Particle System (GIPS)
"""
mutable struct GIPSObj{FT<:AbstractFloat, IT<:Int}
    "a vector of arrays of size N_ensemble x N_parameters containing the parameters (in each GIPS iteration a new array of parameters is added)"
    θ::Vector{Array{FT, 2}}
    "a vector of arrays of size N_ensemble x N_y containing the predicted observation (in each gips iteration a new array of predicted observation is added)"
    y_pred::Vector{Array{FT, 1}}
    "vector of observations (length: N_y)"
    y::Array{FT, 1}
    "size of y"
    N_y::IT
    "covariance of the observational noise"
    Σ_η
    "number ensemble size (2N_θ - 1)"
    N_ens::IT
    "size of θ"
    N_θ::IT
    "time step"
    Δt::FT
    "current iteration number"
    iter::IT
    "method, EKS, Stein"
    method::String
end


# outer constructors
function GIPSObj(
    N_ens::IT,
    θ0_mean::Array{FT,1},
    θθ0_cov::Array{FT,2},
    y::Array{FT, 1},
    Σ_η::Array{FT,2},
    Δt::FT,
    method::String) where {FT<:AbstractFloat, IT<:Int}
    
    
    N_θ = size(θ0_mean,1)
    N_y = size(y, 1)
    
    
    # generate initial assemble
    θ = Array{FT, 2}[] # array of Array{FT, 2}'s
    θ0 = Array(rand(MvNormal(θ0_mean, θθ0_cov), N_ens)')
    push!(θ, θ0) # insert parameters at end of array (in this case just 1st entry)
    
    # prediction
    y_pred = Array{FT, 1}[]  # array of Array{FT, 1}'s
    
    iter = 0

    GIPSObj{FT,IT}(
    θ, 
    y_pred, y, N_y, Σ_η, 
    N_ens, N_θ, 
    Δt, iter, method)
end


function ensemble(s_param, θ_ens::Array{FT,2}, forward::Function)  where {FT<:AbstractFloat}
    
    N_ens,  N_θ = size(θ_ens)
    N_y = s_param.N_y
    g_ens = zeros(FT, N_ens,  N_y)
    
    Threads.@threads for i = 1:N_ens
        θ = θ_ens[i, :]
        g_ens[i, :] .= forward(s_param, θ)
    end
    
    return g_ens
end

function GIPS_Run(s_param, forward::Function, 
    θ0_mean, θθ0_cov,
    y, Σ_η,
    N_ens,
    Δt,
    N_iter, method)
    
    
    gipsobj = GIPSObj(
    N_ens,
    θ0_mean,
    θθ0_cov,
    y,
    Σ_η,
    Δt, 
    method)
    
    
    ens_func(θ_ens) = ensemble(s_param, θ_ens, forward) 
    
    
    for i in 1:N_iter
        update_ensemble!(gipsobj, ens_func) 
    end
    
    return gipsobj
    
end

"""
construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
Construct the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_cov(gips::GIPSObj{FT}, x::Array{FT,2}, x_mean::Array{FT, 1}, y::Array{FT,2}, y_mean::Array{FT, 1}) where {FT<:AbstractFloat}
    N_ens, N_x, N_y = gips.N_ens, size(x_mean,1), size(y_mean,1)
    
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
function construct_cov(gips::GIPSObj{FT}, x::Array{FT,2}) where {FT<:AbstractFloat}
    
    x_mean =  construct_mean(gips, x)
    return construct_cov(gips, x, x_mean, x, x_mean)
end

"""
construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
Construct the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_mean(gips::GIPSObj{FT}, x::Array{FT,2}) where {FT<:AbstractFloat}
    N_ens, N_x = size(x)
    x_mean = zeros(N_x)
    
    
    for i = 1: N_ens
        x_mean .+= x[i,:] 
    end
    
    return x_mean/N_ens
end








function update_ensemble!(gips::GIPSObj{FT}, ens_func::Function) where {FT<:AbstractFloat}
    
    
    N_ens, N_θ, N_y = gips.N_ens, gips.N_θ, gips.N_y
    y, Σ_η = gips.y, gips.Σ_η

    θ = gips.θ[end]
    g = zeros(FT, N_ens, N_y)

    g .= ens_func(θ)
    
   
    # u_mean: N_par × 1
    θ_mean = construct_mean(gips, θ)
    θθ_cov = construct_cov(gips, θ, θ_mean, θ, θ_mean)
    # g_mean: N_obs × 1
    g_mean =  construct_mean(gips, g)
    # g_cov: N_obs × N_obs
    θg_cov = construct_cov(gips, θ, θ_mean, g, g_mean)
    # u_cov: N_par × N_par
    gg_cov = construct_cov(gips, g, g_mean, g, g_mean)




    # GIPS
    if method == "EKS"
        dθ = (ones(N_ens)*y' - g)/Σ_η*θg_cov' + (θ - ones(N_ens)*θ_mean')
    # Stein 
    elseif method == "Stein"
        β = 1.0
        dθ = (((θ - ones(N_ens)*θ_mean')/θθ_cov) * ((θ' - θ_mean*ones(N_ens)')*(ones(N_ens)*y' - g) /N_ens)  /Σ_η * θg_cov')  + (θ - ones(N_ens)*θ_mean') + β*ones(N_ens)*(θg_cov* (Σ_η\(y - g_mean)))'
    else
        error("method = ", method)
    end
    # # D: N_ens × N_ens
    # D = (1/gips.N_ens) * (E' * (gips.Σ_η \ R))
    # Δt = 1/(norm(D) + 1e-8)
    

    θ = θ + Δt * dθ
    
    
    
    # Save results
    push!(gips.θ, θ) # N_ens x N_θ
    push!(gips.y_pred, g_mean)
    

    gips.iter += 1
    
    
end





######################### TEST #######################################

mutable struct Setup_Param{MAT, IT<:Int}
    θ_names::Array{String,1}
    G::MAT
    N_θ::IT
    N_y::IT
end

function Setup_Param(G, N_θ::IT, N_y::IT) where {IT<:Int}
    return Setup_Param(["θ"], G, N_θ, N_y)
end


function forward(s_param::Setup_Param, θ::Array{FT, 1}) where {FT<:AbstractFloat}
    G = s_param.G 
    return G * θ
end

function forward_aug(s_param::Setup_Param, θ::Array{FT, 1}) where {FT<:AbstractFloat}
    G = s_param.G 
    return [G * θ; θ]
end



function Two_Param_Linear_Test(problem_type::String, θ0_bar, θθ0_cov)
    
    N_θ = length(θ0_bar)

    
    if problem_type == "under-determined"
        # under-determined case
        θ_ref = [0.6, 1.2]
        G = [1.0 2.0;]
        
        y = [3.0;]
        Σ_η = Array(Diagonal(fill(0.1^2, size(y))))
        
        
    elseif problem_type == "well-determined"
        # over-determined case
        θ_ref = [1.0, 1.0]
        G = [1.0 2.0; 3.0 4.0]
        
        y = [3.0;7.0]
        Σ_η = Array(Diagonal(fill(0.1^2, size(y))))
        
    elseif problem_type == "over-determined"
        # over-determined case
        θ_ref = [1/3, 17/12.0]
        G = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        
        y = [3.0;7.0;10.0]
        Σ_η = Array(Diagonal(fill(0.1^2, size(y))))
        
        
    else
        error("Problem type : ", problem_type, " has not implemented!")
    end
    
    Σ_post = inv(G'*(Σ_η\G) + inv(θθ0_cov))
    θ_post = θ0_bar + Σ_post*(G'*(Σ_η\(y - G*θ0_bar)))
    

    return θ_post, Σ_post, G, y, Σ_η, θ_ref
end

N_θ = 2
θ0_mean = zeros(Float64, N_θ)
θθ0_cov = Array(Diagonal(fill(1.0^2, N_θ)))
θθ0_cov_sqrt = θθ0_cov

prior_mean     = θ0_mean
prior_cov      = θθ0_cov
prior_cov_sqrt = θθ0_cov_sqrt

problem_type = "well-determined"
θ_post, Σ_post, G, y, Σ_η, θ_ref = Two_Param_Linear_Test(problem_type, θ0_mean, θθ0_cov)
N_y = length(y)

s_param_aug = Setup_Param(G, N_θ, N_y+N_θ)
y_aug = [y ;θ0_mean]
Σ_η_aug = [Σ_η zeros(Float64, N_y, N_θ); zeros(Float64, N_θ, N_y)  θθ0_cov]
 

N_ens = 100
Δt = 0.0001
N_iter = 20000
method = "Stein"
gips_obj = GIPS_Run(s_param_aug, forward_aug, θ0_mean, θθ0_cov, y_aug, Σ_η_aug, N_ens, Δt, N_iter, method)



gips_errors    = zeros(N_iter+1, 2)
for i = 1:N_iter+1
    gips_errors[i, 1] = norm(dropdims(mean(gips_obj.θ[i], dims=1), dims=1) .- θ_post)/norm(θ_post)
    gips_errors[i, 2] = norm(construct_cov(gips_obj, gips_obj.θ[i]) .- Σ_post)/norm(Σ_post)
end

ites = Array(0:N_iter)

markevery = 5

fig, ax = PyPlot.subplots(nrows = 1, ncols=2, sharex=false, sharey="row", figsize=(14,9))
ax[1].semilogy(ites, gips_errors[:, 1],   "-.x", color = "C0", fillstyle="none", label="GIPS", markevery = markevery)
ax[1].set_xlabel("Iterations")
ax[1].set_ylabel("Rel. mean error")
ax[1].grid("on")
ax[1].legend(bbox_to_anchor=(1.0, 1.0))


ax[2].semilogy(ites, gips_errors[:, 2],   "-.x", color = "C0", fillstyle="none", label="GIPS", markevery = markevery)
ax[2].set_xlabel("Iterations")
ax[2].set_ylabel("Rel. covariance error")
ax[2].grid("on")
ax[2].legend(bbox_to_anchor=(1.0, 1.0))


