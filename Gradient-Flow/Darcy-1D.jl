using LinearAlgebra
using Distributions
using Random
using SparseArrays
using JLD2
using ForwardDiff
using NPZ
include("../Inversion/RWMCMC.jl")
include("../Inversion/Plot.jl")


mutable struct Darcy{FT<:AbstractFloat, IT<:Int}
    # physics
    N_x::IT            # number of grid points (including both ends)
    L::FT              # computational domain [0, L]
    Δx::FT
    xx::Array{FT, 1}   # uniform grid [0, Δx, 2Δx ... L]

    # source term
    f::Array{FT, 1}
    
    # for parameterization
    d::FT
    τ::FT
    
    # truth 
    N_KL::IT  # number of Karhunen–Loève expansion modes 
    logk::Array{FT, 1}
    φ::Array{FT, 2}
    λ::Array{FT, 1}
    θ_ref::Array{FT, 1}
    h_ref::Array{FT, 1}
    
    # inverse parameters
    N_θ::IT
    # observation locations and number of observations
    obs_locs::Array{IT, 1}
    
    N_obs::IT
    y_obs::Array{FT, 1}
    σ_obs::FT  # observation error
    
    σ_0::FT  # prior standard deviation
    
end


function Darcy(N_x::IT, L::FT, 
                     N_KL::IT, obs_ΔN::IT, 
                     Nθ::IT, σ_obs::FT, σ_0::FT, d::FT=2.0, τ::FT=3.0) where {FT<:AbstractFloat, IT<:Int}
    xx = Array(LinRange(0, L, N_x))
    Δx = xx[2] - xx[1]
    
    φ, λ = generate_KL_basis(xx, N_KL, d, τ)
    θ_ref = σ_0*rand(Normal(0, 1), N_KL)
    logk = compute_logk(θ_ref, φ, λ)
    
    f = compute_f(xx)
    k = exp.(logk)
    h = solve_Darcy_1D(k, Δx, N_x, f) 

    obs_locs = Array(obs_ΔN:obs_ΔN:N_x-obs_ΔN)
    N_obs = length(obs_locs)
    y_obs_noiseless = compute_obs(h, obs_locs)
    
    
    @assert(Nθ ≤ N_KL)
    noise = σ_obs*rand(Normal(0, 1), N_obs)
    y_obs = y_obs_noiseless + noise

    
    Darcy(N_x, L, Δx, xx, f, d, τ, N_KL, logk, φ, λ, θ_ref, h, Nθ, obs_locs, N_obs, y_obs, σ_obs, σ_0)
end


#=
Initialize the source term term
xx is the 1d x coordinate
=#
function compute_f(xx::Array{FT, 1}) where {FT<:AbstractFloat}

    N_x = length(xx)
    f = zeros(FT, N_x)
    for i = 1:N_x
        if (xx[i] <= 1.0/3.0)
            f[i] = 2000.0
        elseif (xx[i] <= 2.0/3.0)
            f[i] = 1000.0
        else
            f[i] = 0.0
        end
    end
    return f
end


#=
Generate parameters for logk field, including eigenfunctions φ, eigenvalues λ
and the reference parameters θ_ref, and reference field logk field

logk(x) = ∑ θ_{(l)} √λ_{l} φ_{l}(x)
where λ_{l} = (π^2l^2 + τ^2)^{-d}  and φ_{l}(x) = √2 cos(πlx)

generate_θ_KL function generates the summation of the first N_KL terms 
=#
function generate_KL_basis(xx::Array{FT,1}, N_KL::IT, d::FT=2.0, τ::FT=3.0) where {FT<:AbstractFloat, IT<:Int}
    
    N_x = length(xx) 
    φ = zeros(FT, N_KL, N_x)
    λ = zeros(FT, N_KL)
    
    for l = 1:N_KL
        λ[l] = (pi^2*l^2  + τ^2)^(-d)
        φ[l, :] = sqrt(2)*cos.(pi * l * xx)
    end
    
    return φ, λ
end



#=
Compute logk field from θ, as 
logk = ∑ θ[l] * sqrt(λ[l]) * φ[l, :]
=#
function compute_logk(θ, φ, λ) 
    N_KL, Nx = size(φ)
    
    logk = zeros(eltype(θ), Nx)
    
    N_θ = length(θ)
    
    for l = 1:N_θ
        logk .+= θ[l] * sqrt(λ[l]) * φ[l, :]
    end
    
    return logk
end


#=
    solve Darcy equation:
    -∇(k∇h) = f
    with Dirichlet boundary condition, h=0 on ∂Ω

    f_i = -∇(k∇h) (x_i)
        = -(k_{i+0.5}∇h_{i+0.5} - k_{i-0.5}∇h_{i-0.5}) / Δx
        = -(k_{i+0.5}(h_{i+1} - h_{i})/ Δx - k_{i-0.5}(h_{i} - h_{i-1})/ Δx) / Δx
        = -k_{i+0.5}/Δx^2 h_{i+1} + (k_{i+0.5}/Δx^2 + k_{i-0.5}/Δx^2)h_{i}) - k_{i-0.5}/Δx^2  h_{i-1}
    =#
function solve_Darcy_1D(k, Δx, N_x, f)
    𝓒 = Δx^2

    # This is a tridiagonal matrix
    d  = zeros(eltype(k),  N_x-2)
    dl = zeros(eltype(k),  N_x-3)
    dr = zeros(eltype(k),  N_x-3)
    for ix = 2:N_x-1
        d[ix-1] = (k[ix+1] + 2*k[ix] + k[ix-1])/2.0/𝓒
        if ix > 2
            dl[ix-2] = -(k[ix] + k[ix-1])/2.0/𝓒
        end

        if ix < N_x-1
            dr[ix-1] = -(k[ix+1] + k[ix])/2.0/𝓒
        end
    end
    df = Tridiagonal(dl, d, dr)  


    # Multithread does not support sparse matrix solver
    h = df\(f[2:N_x-1])[:]
    
    # include the Dirichlet boundary points
    h_sol = zeros(eltype(k), N_x)
    h_sol[2:N_x-1] .= h
    
    return h_sol
end


function compute_obs(h, obs_locs)  
    return h[obs_locs]
end





# plot any 1D field, with/without highligh the observations by scatter
function plot_field(darcy::Darcy, u::Array{FT, 1}, plot_obs::Bool,  filename::String = "None"; y_obs = u[darcy.obs_locs]) where {FT<:AbstractFloat}
    N_x = darcy.N_x
    xx = darcy.xx

    
    PyPlot.plot(xx, u)

    if plot_obs
        obs_locs = darcy.obs_locs
        x_obs = xx[obs_locs]
        PyPlot.scatter(x_obs, y_obs, color="black")
    end

    tight_layout()
    if filename != "None"
        savefig(filename)
    end
end


function logρ_likelihood(θ, darcy)
    φ, λ = darcy.φ, darcy.λ
    
    logk = compute_logk(θ, φ, λ)
    
    k = exp.(logk)
    
    @assert( minimum(k) > 0.0 )
    
    h = solve_Darcy_1D(k, darcy.Δx, darcy.N_x, darcy.f) 
   
    y = compute_obs(h, darcy.obs_locs)
    
    return  -0.5*(y - darcy.y_obs)'*(y - darcy.y_obs)/darcy.σ_obs^2 
end


function logρ_posterior(θ, darcy)
    return logρ_likelihood(θ, darcy) - 0.5*θ'*θ/darcy.σ_0^2
end
