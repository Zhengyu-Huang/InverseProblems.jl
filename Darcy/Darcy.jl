using JLD2
using Statistics
using LinearAlgebra
using Distributions
using Random
using SparseArrays




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
    
    logκ_2d::Array{Float64, 2}
    φ::Array{Float64, 3}
    λ::Array{Float64, 1}
    u_ref::Array{Float64, 1}
    
    #for source term
    f_2d::Array{Float64, 2}
    
end


function Param_Darcy(N::Int64, obs_ΔN::Int64, L::Float64, trunc_KL::Int64, α::Float64=2.0, τ::Float64=3.0)
    Δx =  L/(N-1)
    xx = Array(LinRange(0, L, N))
    @assert(xx[2] - xx[1] ≈ Δx)
    
    logκ_2d,φ,λ,u = generate_θ_KL(N, xx, trunc_KL, α, τ)
    f_2d = compute_f_2d(N, xx)
    n_data = length(obs_ΔN:obs_ΔN:N-obs_ΔN)^2
    Param_Darcy(N, L, Δx, xx, obs_ΔN, n_data, trunc_KL, α, τ, logκ_2d, φ, λ, u, f_2d)
end

function point(darcy::Param_Darcy, ix::Int64, iy::Int64)
    return darcy.xx[ix], darcy.xx[iy]
end

function ind(darcy::Param_Darcy, ix::Int64, iy::Int64)
    return (ix-1) + (iy-2)*(darcy.N - 2)
end

function plot_field(darcy::Param_Darcy, u_2d::Array{Float64, 2}, plot_obs::Bool,  filename::String = "None")
    @info "start to plot and save to ", filename
    N = darcy.N
    xx = darcy.xx
    figure(123)
    X,Y = repeat(xx, 1, N), repeat(xx, 1, N)'
    pcolormesh(X, Y, u_2d, cmap="jet")
    colorbar()

    if plot_obs
        obs_ΔN = darcy.obs_ΔN
        x_obs, y_obs = X[obs_ΔN:obs_ΔN:N-obs_ΔN,obs_ΔN:obs_ΔN:N-obs_ΔN][:], Y[obs_ΔN:obs_ΔN:N-obs_ΔN,obs_ΔN:obs_ΔN:N-obs_ΔN][:] 
        scatter(x_obs, y_obs, color="black")
    end

    tight_layout()
    if filename != "None"
        savefig(filename)
        close("all")
    end
end

function plot_field(darcy::Param_Darcy, u_2d::Array{Float64, 2},  clim, ax)
    N = darcy.N
    xx = darcy.xx
    figure(123)
    X,Y = repeat(xx, 1, N), repeat(xx, 1, N)'
    return ax.pcolormesh(X, Y, u_2d, cmap="jet", clim=clim)
end


#=
Compute reference logk field, as 
∑ u[i] * sqrt(λ[i]) * φ[i, :, :]
=#
function compute_logκ_2d(darcy::Param_Darcy, u::Array{Float64, 1})
    N, trunc_KL = darcy.N, darcy.trunc_KL
    λ, φ = darcy.λ, darcy.φ
    N_θ = length(u)
    
    @assert(N_θ <= trunc_KL) 
    logκ_2d = zeros(Float64, N, N)
    for i = 1:N_θ
        logκ_2d .+= u[i] * sqrt(λ[i]) * φ[i, :, :]
    end
    
    return logκ_2d
end



#=
Initialize forcing term
=#
function compute_f_2d(N::Int64, yy::Array{Float64, 1})
    #f_2d = ones(Float64, N, N)
    
    f_2d = zeros(Float64, N, N)
    
    for i = 1:N
        if (yy[i] <= 4/6)
            f_2d[:,i] .= 1000.0
        elseif (yy[i] >= 4/6 && yy[i] <= 5/6)
            f_2d[:,i] .= 2000.0
        elseif (yy[i] >= 5/6)
            f_2d[:,i] .= 3000.0
        end
    end
    
    return f_2d
end

#=
Compute sorted pair (a1, a2), sorted by a1^2 + a2^2
with a1≥0 and a2≥0 anad a1+a2>0
=#
function compute_seq_pairs(trunc_KL::Int64)
    seq_pairs = zeros(Int64, trunc_KL, 2)
    trunc_Nx = trunc(Int64, sqrt(2*trunc_KL)) + 1
    
    seq_pairs = zeros(Int64, (trunc_Nx+1)^2 - 1, 2)
    seq_pairs_mag = zeros(Int64, (trunc_Nx+1)^2 - 1)
    
    seq_pairs_i = 0
    for i = 0:trunc_Nx
        for j = 0:trunc_Nx
            if (i == 0 && j ==0)
                continue
            end
            seq_pairs_i += 1
            seq_pairs[seq_pairs_i, :] .= i, j
            seq_pairs_mag[seq_pairs_i] = i^2 + j^2
        end
    end
    
    seq_pairs = seq_pairs[sortperm(seq_pairs_mag), :]
    return seq_pairs[1:trunc_KL, :]
end

#=
Generate parameters for logk field, including eigenfunctions φ, eigenvalues λ
and the reference parameters u, and reference field logk_2d field
=#
function generate_θ_KL(N::Int64, xx::Array{Float64,1}, trunc_KL::Int64, α::Float64=2.0, τ::Float64=3.0)
    #logκ = ∑ u_l √λ_l φ_l(x)      l ∈ Z^{+}
    #                                  (0, 1), (1, 0) 
    #                                  (0, 2), (1,  1), (2, 0)  ...
    
    
    X,Y = repeat(xx, 1, N), repeat(xx, 1, N)'
    
    seq_pairs = compute_seq_pairs(trunc_KL)
    
    φ = zeros(Float64, trunc_KL, N, N)
    λ = zeros(Float64, trunc_KL)
    
    for i = 1:trunc_KL
        if (seq_pairs[i, 1] == 0 && seq_pairs[i, 2] == 0)
            φ[i, :, :] .= 1.0
        elseif (seq_pairs[i, 1] == 0)
            φ[i, :, :] = sqrt(2)*cos.(pi * (seq_pairs[i, 2]*Y))
        elseif (seq_pairs[i, 2] == 0)
            φ[i, :, :] = sqrt(2)*cos.(pi * (seq_pairs[i, 1]*X))
        else
            φ[i, :, :] = 2*cos.(pi * (seq_pairs[i, 1]*X)) .*  cos.(pi * (seq_pairs[i, 2]*Y))
        end

        λ[i] = (pi^2*(seq_pairs[i, 1]^2 + seq_pairs[i, 2]^2) + τ^2)^(-α)
    end
    
    Random.seed!(123);
    u = rand(Normal(0, 1), trunc_KL)

    logκ_2d = zeros(Float64, N, N)
    for i = 1:trunc_KL
        logκ_2d .+= u[i]*sqrt(λ[i])*φ[i, :, :]
    end
    
    return logκ_2d, φ, λ, u
end



#=
    solve Darcy equation:
    -∇(κ∇h) = f
    with Dirichlet boundary condition, h=0 on ∂Ω
=#
function solve_GWF(darcy::Param_Darcy, κ_2d::Array{Float64,2})
    Δx, N = darcy.Δx, darcy.N
    
    indx = Int64[]
    indy = Int64[]
    vals = Float64[]
    
    f_2d = darcy.f_2d
    
    𝓒 = Δx^2
    for iy = 2:N-1
        for ix = 2:N-1
            
            ixy = ind(darcy, ix, iy) 
            
            #top
            if iy == N-1
                #ft = -(κ_2d[ix, iy] + κ_2d[ix, iy+1])/2.0 * (0 - h_2d[ix,iy])
                push!(indx, ixy)
                push!(indy, ixy)
                push!(vals, (κ_2d[ix, iy] + κ_2d[ix, iy+1])/2.0/𝓒)
            else
                #ft = -(κ_2d[ix, iy] + κ_2d[ix, iy+1])/2.0 * (h_2d[ix,iy+1] - h_2d[ix,iy])
                append!(indx, [ixy, ixy])
                append!(indy, [ixy, ind(darcy, ix, iy+1)])
                append!(vals, [(κ_2d[ix, iy] + κ_2d[ix, iy+1])/2.0/𝓒, -(κ_2d[ix, iy] + κ_2d[ix, iy+1])/2.0/𝓒])
                
            end
            
            #bottom
            if iy == 2
                #fb = -(κ_2d[ix, iy] + κ_2d[ix, iy-1])/2.0 * (h_2d[ix,iy] - 0)
                push!(indx, ixy)
                push!(indy, ixy)
                push!(vals,  (κ_2d[ix, iy] + κ_2d[ix, iy-1])/2.0/𝓒)
            else
                #fb = -(κ_2d[ix, iy] + κ_2d[ix, iy-1])/2.0 * (h_2d[ix,iy] - h_2d[ix,iy-1])
                append!(indx, [ixy, ixy])
                append!(indy, [ixy, ind(darcy, ix, iy-1)])
                append!(vals, [(κ_2d[ix, iy] + κ_2d[ix, iy-1])/2.0/𝓒, -(κ_2d[ix, iy] + κ_2d[ix, iy-1])/2.0/𝓒])
            end
            
            #right
            if ix == N-1
                #fr = -(κ_2d[ix, iy] + κ_2d[ix+1, iy])/2.0 * (0 - h_2d[ix,iy])
                push!(indx, ixy)
                push!(indy, ixy)
                push!(vals, (κ_2d[ix, iy] + κ_2d[ix+1, iy])/2.0/𝓒)
            else
                #fr = -(κ_2d[ix, iy] + κ_2d[ix+1, iy])/2.0 * (h_2d[ix+1,iy] - h_2d[ix,iy])
                append!(indx, [ixy, ixy])
                append!(indy, [ixy, ind(darcy, ix+1, iy)])
                append!(vals, [(κ_2d[ix, iy] + κ_2d[ix+1, iy])/2.0/𝓒, -(κ_2d[ix, iy] + κ_2d[ix+1, iy])/2.0/𝓒])
            end  
            
            #left
            if ix == 2
                #fl = -(κ_2d[ix, iy] + κ_2d[ix-1, iy])/2.0 * (h_2d[ix,iy] - 0)
                push!(indx, ixy)
                push!(indy, ixy)
                push!(vals, (κ_2d[ix, iy] + κ_2d[ix-1, iy])/2.0/𝓒)
            else
                #fl = -(κ_2d[ix, iy] + κ_2d[ix-1, iy])/2.0 * (h_2d[ix,iy] - h_2d[ix-1,iy])
                append!(indx, [ixy, ixy])
                append!(indy, [ixy, ind(darcy, ix-1, iy)])
                append!(vals, [(κ_2d[ix, iy] + κ_2d[ix-1, iy])/2.0/𝓒, -(κ_2d[ix, iy] + κ_2d[ix-1, iy])/2.0/𝓒])
            end
            
            
            #f[ix,iy] = (ft - fb + fr - fl)/𝓒
            
        end
    end
    
    
    
    df = sparse(indx, indy, vals, (N-2)^2, (N-2)^2)
    # Multithread does not support sparse matrix solver
    h = df\(f_2d[2:N-1,2:N-1])[:]
    
    h_2d = zeros(Float64, N, N)
    h_2d[2:N-1,2:N-1] .= reshape(h, N-2, N-2) 
    
    return h_2d
end



#=
Compute observation values
=#
function compute_obs(darcy::Param_Darcy, h_2d::Array{Float64, 2})
    N = darcy.N
    obs_ΔN = darcy.obs_ΔN
    
    # X---X(1)---X(2) ... X(obs_N)---X
    obs_2d = h_2d[obs_ΔN:obs_ΔN:N-obs_ΔN,obs_ΔN:obs_ΔN:N-obs_ΔN] 
    
    return obs_2d[:]
end



function run_Darcy_ensemble(darcy::Param_Darcy, params_i::Array{Float64, 2})
    N_ens,  N_θ = size(params_i)
    
    g_ens = zeros(Float64, N_ens, darcy.n_data)
    
    for i = 1:N_ens
        
        logκ_2d = compute_logκ_2d(darcy, params_i[i, :])
        κ_2d = exp.(logκ_2d)
        
        h_2d = solve_GWF(darcy, κ_2d)
        
        obs = compute_obs(darcy, h_2d)
        
        # g: N_ens x N_data
        g_ens[i,:] .= obs 
    end
    
    return g_ens
end














