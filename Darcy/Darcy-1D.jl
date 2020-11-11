using JLD2
using Statistics
using LinearAlgebra
using Distributions
using Random
using SparseArrays
include("../Plot.jl")



mutable struct Param_Darcy
    N::Int64
    L::Float64
    Δx::Float64
    xx::Array{Float64, 1}
    
    #for observation
    obs_ΔN::Int64
    n_data::Int64
    
    #for parameterization
    trunc_KL::Int64  # this is for generating the truth
    α::Float64
    τ::Float64
    
    logκ::Array{Float64, 1}
    φ::Array{Float64, 2}
    λ::Array{Float64, 1}
    u_ref::Array{Float64, 1}
    
    #for source term
    f::Array{Float64, 1}
    
end


function Param_Darcy(N::Int64, obs_ΔN::Int64, L::Float64, trunc_KL::Int64, α::Float64=2.0, τ::Float64=3.0)
    Δx =  L/(N-1)
    xx = Array(LinRange(0, L, N))
    @assert(xx[2] - xx[1] ≈ Δx)
    
    logκ,φ,λ,u = generate_θ_KL(N, xx, trunc_KL, α, τ)
    f = compute_f(N, xx)
    n_data = length(obs_ΔN:obs_ΔN:N-obs_ΔN)
    Param_Darcy(N, L, Δx, xx, obs_ΔN, n_data, trunc_KL, α, τ, logκ, φ, λ, u, f)
end



function plot_field(darcy::Param_Darcy, u::Array{Float64, 1}, plot_obs::Bool,  filename::String = "None")
    N = darcy.N
    xx = darcy.xx
    figure(123)
    
    plot(xx, u)

    if plot_obs
        obs_ΔN = darcy.obs_ΔN
        x_obs, y_obs = xx[obs_ΔN:obs_ΔN:N-obs_ΔN][:], u[obs_ΔN:obs_ΔN:N-obs_ΔN][:] 
        scatter(x_obs, y_obs, color="black")
    end

    tight_layout()
    if filename != "None"
        savefig(filename)
        close(123)
    end
end

function plot_field(darcy::Param_Darcy, u::Array{Float64, 2},  clim, ax)
    N = darcy.N
    xx = darcy.xx
    figure(123)
    return ax.plot(xx,  u)
end


#=
Compute reference logk field, as 
∑ u[i] * sqrt(λ[i]) * φ[i, :, :]
=#
function compute_logκ(darcy::Param_Darcy, u::Array{Float64, 1})
    N, trunc_KL = darcy.N, darcy.trunc_KL
    λ, φ = darcy.λ, darcy.φ
    N_θ = length(u)
    
    @assert(N_θ <= trunc_KL) 
    logκ = zeros(Float64, N)
    for i = 1:N_θ
        logκ .+= u[i] * sqrt(λ[i]) * φ[i, :]
    end
    
    return logκ
end



#=
Initialize forcing term
=#
function compute_f(N::Int64, xx::Array{Float64, 1})
    #f_2d = ones(Float64, N, N)
    
    f = zeros(Float64, N)
    
    for i = 1:N
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
and the reference parameters u, and reference field logk_2d field
=#
function generate_θ_KL(N::Int64, xx::Array{Float64,1}, trunc_KL::Int64, α::Float64=2.0, τ::Float64=3.0)

    φ = zeros(Float64, trunc_KL, N)
    λ = zeros(Float64, trunc_KL)
    for i = 1:trunc_KL
        φ[i, :] = sqrt(2)*cos.(pi * i * xx)
    end
    
    Random.seed!(123);
    u = rand(Normal(0, 1), trunc_KL)

    logκ = zeros(Float64, N)
    for i = 1:trunc_KL
        λ[i] = (pi^2*i^2  + τ^2)^(-α)

        logκ .+= u[i]*sqrt(λ[i])*φ[i, :]
    end
    
    return logκ, φ, λ, u
end



#=
    solve Darcy equation:
    -∇(κ∇h) = f
    with Dirichlet boundary condition, h=0 on ∂Ω
=#
function solve_GWF(darcy::Param_Darcy, κ::Array{Float64,1})
    Δx, N = darcy.Δx, darcy.N
    𝓒 = Δx^2
    f = darcy.f

    # This is a tridiagonal matrix
    d  = zeros(Float64,  N-2)
    dl = zeros(Float64,  N-3)
    dr = zeros(Float64,  N-3)
    for ix = 2:N-1
        d[ix-1] = (κ[ix+1] + 2*κ[ix] + κ[ix-1])/2.0/𝓒
        if ix > 2
            dl[ix-2] = -(κ[ix] + κ[ix-1])/2.0/𝓒
        end

        if ix < N-1
            dr[ix-1] = -(κ[ix+1] + κ[ix])/2.0/𝓒
        end
    end
    df = Tridiagonal(dl, d, dr)  


    # Multithread does not support sparse matrix solver
    h = df\(f[2:N-1])[:]
    
    h_sol = zeros(Float64, N)
    h_sol[2:N-1] .= h
    
    return h_sol
end



#=
Compute observation values
=#
function compute_obs(darcy::Param_Darcy, h::Array{Float64, 1})
    N = darcy.N
    obs_ΔN = darcy.obs_ΔN
    
    # X---X(1)---X(2) ... X(obs_N)---X
    obs_2d = h[obs_ΔN:obs_ΔN:N-obs_ΔN] 
    
    return obs_2d[:]
end



function run_Darcy_ensemble(darcy::Param_Darcy, params_i::Array{Float64, 2})
    N_ens,  N_θ = size(params_i)
    
    g_ens = zeros(Float64, N_ens, darcy.n_data)
    
    for i = 1:N_ens
        
        logκ = compute_logκ(darcy, params_i[i, :])
        κ = exp.(logκ)
        
        h = solve_GWF(darcy, κ)
        
        obs = compute_obs(darcy, h)
        
        # g: N_ens x N_data
        g_ens[i,:] .= obs 
    end
    
    return g_ens
end







N, L = 256, 1.0
obs_ΔN = 10
α = 2.0
τ = 3.0
KL_trunc = 64
darcy = Param_Darcy(N, obs_ΔN, L, KL_trunc, α, τ)
N_ite = 20
    
κ = exp.(darcy.logκ)
h = solve_GWF(darcy, κ)
plot_field(darcy, h, true, "Figs/Darcy-1D-obs-ref.pdf")
plot_field(darcy, darcy.logκ, false, "Figs/Darcy-1D-logk-ref.pdf")
    
    






