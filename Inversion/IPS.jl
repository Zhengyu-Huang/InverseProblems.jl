using LinearAlgebra
using Random
using Distributions
using EllipsisNotation
using PyPlot
"""
IPSObj{FT<:AbstractFloat, IT<:Int}
Structure that is used in Gaussian Interaction Particle System (IPS)
"""
mutable struct IPSObj{FT<:AbstractFloat, IT<:Int}
    "a vector of arrays of size N_ensemble x N_parameters containing the parameters (in each IPS iteration a new array of parameters is added)"
    θ::Vector{Array{FT, 2}}
    "number ensemble size (2N_θ - 1)"
    N_ens::IT
    "size of θ"
    N_θ::IT
    "time step"
    Δt::FT
    "current iteration number"
    iter::IT
    "method, Wasserstein, Stein"
    method::String
    "Preconditioner, true or false"
    preconditioner::Bool
    "Kernel param"
    kernel_param
end


# outer constructors
function IPSObj(
    N_ens::IT,
    θ0::Array{FT,2},
    Δt::FT,
    method::String,
    preconditioner::Bool,
    kernel_param=nothing) where {FT<:AbstractFloat, IT<:Int}
    
    # generate initial assemble
    θ = Array{FT, 2}[] # array of Array{FT, 2}'s

    push!(θ, θ0) # insert parameters at end of array (in this case just 1st entry)
    
    iter = 0
    
    _,  N_θ = size(θ0)
    
    IPSObj{FT,IT}(
    θ, N_ens, N_θ, 
    Δt, iter, method, preconditioner, kernel_param)
end


function ensemble_∇logρ(s_param, θ_ens::Array{FT,2}, forward::Function)  where {FT<:AbstractFloat}
    
    N_ens,  N_θ = size(θ_ens)
    g_ens = zeros(FT, N_ens)
    ∇g_ens = zeros(FT, N_ens,  N_θ)
    
    Threads.@threads for i = 1:N_ens
        θ = θ_ens[i, :]
        g_ens[i], ∇g_ens[i, :] = forward(s_param, θ)
    end
    
    return g_ens, ∇g_ens
end

function IPS_Run(s_param, ∇logρ_func::Function, 
    θ0,
    N_ens,
    Δt,
    N_iter, method, preconditioner, kernel_param=nothing)
    
    ipsobj = IPSObj(
    N_ens,
    θ0,
    Δt, 
    method, 
    preconditioner, kernel_param)
    
    
    ens_func(θ_ens) = ensemble_∇logρ(s_param, θ_ens, ∇logρ_func) 
    
    for i in 1:N_iter
        update_ensemble!(ipsobj, ens_func) 
    end
    
    return ipsobj
    
end

"""
construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
Construct the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_cov(x::Array{FT,2}, x_mean::Array{FT, 1}, y::Array{FT,2}, y_mean::Array{FT, 1}) where {FT<:AbstractFloat}
    N_ens, N_x, N_y = size(x, 1), size(x_mean,1), size(y_mean,1)
    
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
function construct_cov(x::Array{FT,2}) where {FT<:AbstractFloat}
    
    x_mean =  construct_mean(x)
    return construct_cov(x, x_mean, x, x_mean)
end

"""
construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
Construct the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_mean(x::Array{FT,2}) where {FT<:AbstractFloat}
    N_ens, N_x = size(x)
    x_mean = zeros(N_x)
    
    
    for i = 1: N_ens
        x_mean .+= x[i,:] 
    end
    
    return x_mean/N_ens
end


function compute_h(θ)
    J, nd = size(θ)

    XY = θ*θ';
    x2= sum(θ.^2, dims=2);
    X2e = repeat(x2, 1, J);
    pairwise_dists = X2e + X2e' - 2*XY

    
    h = median(pairwise_dists)  
    h = sqrt(0.5 * h / log(J+1))
    
    return h
end

#compute κ(xi, xj) and ∇_xj κ(xi, xj)
function kernel(xs; C = nothing)
    N_ens, d = size(xs)
    κ = zeros(N_ens, N_ens)
    dκ = zeros(N_ens, N_ens, d)
    
    if C === nothing
        h = compute_h(xs)
        C = h^2
        scale = sqrt( (1 + 4*log(N_ens + 1)/d)^d )
    else
        C = C
        scale = sqrt( (1 + 2)^d )
    end

     
    
    for i = 1:N_ens
        for j = 1:N_ens
            dpower = C\(xs[i,:] - xs[j,:])
            power =  -0.5* ( (xs[i,:] - xs[j,:])' * dpower )  
            κ[i, j] = exp(power)
            dκ[i, j, :] = exp(power) * dpower
        end
    end
    
    return scale*κ, scale*dκ
end



#compute κ(xi, xj) and ∇_xj κ(xi, xj)
function kernel(xs, xs_; C = nothing)
    N_ens,  d = size(xs)
    N_ens_, d = size(xs_)
    κ = zeros(N_ens, N_ens_)
    
    if C === nothing
        h = compute_h(xs)
        C = h^2
        scale = sqrt( (1 + 4*log(N_ens + 1)/d)^d )
    else
        C = C
        scale = sqrt( (1 + 2)^d )
    end

     
    
    for i = 1:N_ens
        for j = 1:N_ens_
            dpower = C\(xs[i,:] - xs_[j,:])
            power =  -0.5* ( (xs[i,:] - xs_[j,:])' * dpower )  
            κ[i, j] = exp(power)
        end
    end
    
    return scale*κ
end


function update_ensemble!(ips::IPSObj{FT}, ens_func::Function) where {FT<:AbstractFloat}
    
    N_ens, N_θ = ips.N_ens, ips.N_θ
    method = ips.method
    Δt = ips.Δt
    θ = ips.θ[end]
    logρ = zeros(FT, N_ens)
    ∇logρ = zeros(FT, N_ens, N_θ)
    logρ, ∇logρ = ens_func(θ)
    
   
    # u_mean: N_par × 1
    θ_mean = construct_mean(θ)
    θθ_cov = construct_cov(θ, θ_mean, θ, θ_mean)
    Prec = (ips.preconditioner ? θθ_cov : I)

    if method == "Wasserstein" || method == "Wasserstein-Fisher-Rao"
        noise = rand(Normal(0, 1), (N_θ, N_ens))
        if ips.preconditioner & !isposdef(θθ_cov)
            @info θθ_cov
            @info eigen(θθ_cov)
        end
        σ = (ips.preconditioner ? cholesky(Hermitian(θθ_cov)).L : I)
        dθ = ∇logρ * Prec' + sqrt(2/Δt)*(σ*noise)'
    # Stein 
    elseif method == "Stein" || method == "Stein-Fisher-Rao"
        κ, dκ = kernel(θ; C = (ips.preconditioner ? θθ_cov : nothing) )
        dθ = 1/N_ens * (Prec *  ∇logρ'  * κ)'
        for i = 1:N_ens
            dθ[i, :] += 1/N_ens * (Prec *  sum(dκ[i,:,:], dims = 1)' )
        end
    else
        error("method = ", method)
    end

    θ = θ + Δt * dθ
    
    if method == "Wasserstein-Fisher-Rao" || method == "Stein-Fisher-Rao"
        # Always use the kernel without prconditioner
        # κ, _ = kernel(θ; C = (ips.preconditioner ? θθ_cov : nothing) )
        κ, _ = kernel(θ; C = ips.kernel_param)
        # equation 57
        Λ = log.(sum(κ, dims=2)/N_ens)[:] + sum(κ./sum(κ, dims=2) , dims=2)[:] - logρ .- sum(log.(sum(κ, dims=2)/N_ens))/N_ens .- 1 .+ sum(logρ)/N_ens

        
        Λ .-= sum(Λ)/N_ens
        p_Λ = 1 .- exp.(-abs.(Λ)*Δt)
        
        p_rand = rand(Uniform(0, 1), N_ens)
        θ_copy = copy(θ)
        
        # generate random number exclude itself
        p_ind = rand(1:N_ens-1, N_ens)
        for i = 1:N_ens
            if p_ind[i] >= i
                p_ind[i] += 1
            end
        end
        
        
        for i = 1:N_ens
            if p_rand[i] <= p_Λ[i]
                if Λ[i] > 0 #kill θi with probability 1 − exp(−Λ[i]∆t),
                    θ[i, :] = θ_copy[p_ind[i], :]
                elseif Λ[i] < 0 #duplicate θi with probability 1 − exp(Λ[i]∆t),
                    θ[p_ind[i], :] = θ_copy[i, :]
                end
            end
        end
    end
    
    
 
    # Save results
    push!(ips.θ, θ) # N_ens x N_θ    

    ips.iter += 1
    
    
end




