using Distributions
using Random
using LinearAlgebra
using SparseArrays
include("../Inversion/Plot.jl")



mutable struct Setup_Param{FT<:AbstractFloat, IT<:Int}
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
    logκ::Array{FT, 1}
    φ::Array{FT, 2}
    λ::Array{FT, 1}
    θ_ref::Array{FT, 1}
    
    
    # inverse parameters
    θ_names::Array{String, 1}
    N_θ::IT
    # observation locations and number of observations
    y_locs::Array{IT, 1}
    N_y::IT
    
end


function Setup_Param(N_x::IT, L::FT, 
                     N_KL::IT, obs_ΔN::IT, 
                     N_θ::IT, d::FT=2.0, τ::FT=3.0; seed::IT=123) where {FT<:AbstractFloat, IT<:Int}
    xx = Array(LinRange(0, L, N_x))
    Δx = xx[2] - xx[1]
    
    logκ, φ, λ, θ_ref = generate_θ_KL(xx, N_KL, d, τ, seed=seed)
    f = compute_f(xx)

    y_locs = Array(obs_ΔN:obs_ΔN:N_x-obs_ΔN)
    N_y = length(y_locs)

    θ_names = ["θ"]
    Setup_Param(N_x, L, Δx, xx, f, d, τ, N_KL, logκ, φ, λ, θ_ref, θ_names, N_θ, y_locs, N_y)
end


#=
Initialize the source term term
xx is the 1d x coordinate
=#
function compute_f(xx::Array{FT, 1}) where {FT<:AbstractFloat}

    N_x = length(xx)
    f = zeros(FT, N_x)
    for i = 1:N_x
        if (xx[i] <= 0.5)
            f[i] = 1000.0
        else
            f[i] = 2000.0
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
function generate_θ_KL(xx::Array{FT,1}, N_KL::IT, d::FT=2.0, τ::FT=3.0; seed::IT=123) where {FT<:AbstractFloat, IT<:Int}
    
    N_x = length(xx) 
    φ = zeros(FT, N_KL, N_x)
    λ = zeros(FT, N_KL)
    
    for l = 1:N_KL
        φ[l, :] = sqrt(2)*cos.(pi * l * xx)
    end
    
    Random.seed!(seed);
    θ_ref = rand(Normal(0, 1), N_KL)

    logκ = zeros(FT, N_x)
    for l = 1:N_KL
        λ[l] = (pi^2*l^2  + τ^2)^(-d)

        logκ .+= θ_ref[l]*sqrt(λ[l])*φ[l, :]
    end
    
    return logκ, φ, λ, θ_ref
end

#=
Compute logk field from θ, as 
logk = ∑ θ[l] * sqrt(λ[l]) * φ[l, :]
=#
function compute_logκ(darcy::Setup_Param, θ::Array{FT, 1}) where {FT<:AbstractFloat}
    N_x, N_KL = darcy.N_x, darcy.N_KL
    λ, φ = darcy.λ, darcy.φ
    N_θ = length(θ)
    
    @assert(N_θ <= N_KL) 
    logκ = zeros(FT, N_x)
    for l = 1:N_θ
        logκ .+= θ[l] * sqrt(λ[l]) * φ[l, :]
    end
    
    return logκ
end


#=
    solve Darcy equation:
    -∇(κ∇h) = f
    with Dirichlet boundary condition, h=0 on ∂Ω

    f_i = -∇(κ∇h) (x_i)
        = -(κ_{i+0.5}∇h_{i+0.5} - κ_{i-0.5}∇h_{i-0.5}) / Δx
        = -(κ_{i+0.5}(h_{i+1} - h_{i})/ Δx - κ_{i-0.5}(h_{i} - h_{i-1})/ Δx) / Δx
        = -κ_{i+0.5}/Δx^2 h_{i+1} + (κ_{i+0.5}/Δx^2 + κ_{i-0.5}/Δx^2)h_{i}) - κ_{i-0.5}/Δx^2  h_{i-1}
    =#
function solve_Darcy_1D(darcy::Setup_Param, κ::Array{FT,1}) where {FT<:AbstractFloat}
    Δx, N_x = darcy.Δx, darcy.N_x
    𝓒 = Δx^2
    f = darcy.f

    # This is a tridiagonal matrix
    d  = zeros(FT,  N_x-2)
    dl = zeros(FT,  N_x-3)
    dr = zeros(FT,  N_x-3)
    for ix = 2:N_x-1
        d[ix-1] = (κ[ix+1] + 2*κ[ix] + κ[ix-1])/2.0/𝓒
        if ix > 2
            dl[ix-2] = -(κ[ix] + κ[ix-1])/2.0/𝓒
        end

        if ix < N_x-1
            dr[ix-1] = -(κ[ix+1] + κ[ix])/2.0/𝓒
        end
    end
    df = Tridiagonal(dl, d, dr)  


    # Multithread does not support sparse matrix solver
    h = df\(f[2:N_x-1])[:]
    
    # include the Dirichlet boundary points
    h_sol = zeros(FT, N_x)
    h_sol[2:N_x-1] .= h
    
    return h_sol
end


function compute_obs(darcy::Setup_Param, h::Array{FT, 1})  where {FT<:AbstractFloat}
    return h[darcy.y_locs]
end

function forward(darcy::Setup_Param, θ::Array{FT, 1}) where {FT<:AbstractFloat}
    
    logκ = compute_logκ(darcy, θ)
    κ = exp.(logκ)
    h = solve_Darcy_1D(darcy, κ)  
    y = compute_obs(darcy, h)  
    return y
end

function aug_forward(darcy::Setup_Param, θ::Array{FT, 1}) where {FT<:AbstractFloat}
    y = forward(darcy, θ)  
    return [y ; θ]
end

# plot any 1D field, with/without highligh the observations by scatter
function plot_field(darcy::Setup_Param, u::Array{FT, 1}, plot_obs::Bool,  filename::String = "None"; y_obs = u[darcy.y_locs]) where {FT<:AbstractFloat}
    N_x = darcy.N_x
    xx = darcy.xx

    
    PyPlot.plot(xx, u)

    if plot_obs
        y_locs = darcy.y_locs
        x_obs, y_obs = xx[y_locs], u[y_locs]
        PyPlot.scatter(x_obs, y_obs, color="black")
    end

    tight_layout()
    if filename != "None"
        savefig(filename)
    end
end







    
    






