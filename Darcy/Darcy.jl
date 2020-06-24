using JLD2
using Statistics
using LinearAlgebra
using Distributions
using Random
using SparseArrays
include("../Plot.jl")
include("../UKI.jl")
include("../EKI.jl")



mutable struct Param_Darcy
    N::Int64
    L::Float64
    Δx::Float64
    xx::Array{Float64, 1}
    
    #for observation
    obs_ΔN::Int64
    
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
    
    Param_Darcy(N, L, Δx, xx, obs_ΔN, trunc_KL, α, τ, logκ_2d, φ, λ, u, f_2d)
end

function point(darcy::Param_Darcy, ix::Int64, iy::Int64)
    return darcy.xx[ix], darcy.xx[iy]
end

function ind(darcy::Param_Darcy, ix::Int64, iy::Int64)
    return (ix-1) + (iy-2)*(darcy.N - 2)
end

function plot_field(darcy::Param_Darcy, u_2d::Array{Float64, 2}, filename::String = "None")
    N = darcy.N
    xx = darcy.xx
    X,Y = repeat(xx, 1, N), repeat(xx, 1, N)'
    #pcolormesh(X, Y, u_2d, shading= "gouraud", cmap="jet")
    pcolormesh(X, Y, u_2d, cmap="jet")
    colorbar()
    tight_layout()
    if filename != "None"
        savefig(filename)
        close("all")
    end
end


function plot_obs(darcy::Param_Darcy, u_2d::Array{Float64, 2}, filename::String = "None")
    N, obs_ΔN = darcy.N, darcy.obs_ΔN
    
    xx = darcy.xx
    X,Y = repeat(xx, 1, N), repeat(xx, 1, N)'
    
    x_obs, y_obs = X[obs_ΔN:obs_ΔN:N-obs_ΔN,obs_ΔN:obs_ΔN:N-obs_ΔN][:], Y[obs_ΔN:obs_ΔN:N-obs_ΔN,obs_ΔN:obs_ΔN:N-obs_ΔN][:] 
    
    #pcolormesh(X, Y, u_2d, shading= "gouraud", cmap="jet")
    pcolormesh(X, Y, u_2d, cmap="jet")
    scatter(x_obs, y_obs, color="black")
    colorbar()
    tight_layout()
    if filename != "None"
        savefig(filename)
        close("all")
    end
end

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


function compute_seq_pairs(trunc_KL::Int64)
    seq_pairs = zeros(Int64, trunc_KL, 2)
    trunc_Nx = trunc(Int64, sqrt(2*trunc_KL)) + 1
    
    seq_pairs = zeros(Int64, (trunc_Nx+1)^2, 2)
    seq_pairs_mag = zeros(Int64, (trunc_Nx+1)^2)
    
    seq_pairs_i = 0
    for i = 0:trunc_Nx
        for j = 0:trunc_Nx
            seq_pairs_i += 1
            seq_pairs[seq_pairs_i, :] .= i, j
            seq_pairs_mag[seq_pairs_i] = i^2 + j^2
        end
    end
    
    seq_pairs = seq_pairs[sortperm(seq_pairs_mag), :]
    
    return seq_pairs[1:trunc_KL, :]
end

function generate_θ_KL(N::Int64, xx::Array{Float64,1}, trunc_KL::Int64, α::Float64=2.0, τ::Float64=3.0)
    #logκ = ∑ u_l √λ_l φ_l(x)      l ∈ Z^{+}
    #                                  (0, 0)
    #                                  (0, 1), (1, 0) 
    #                                  (0, 2), (1,  1), (2, 0)  ...
    
    
    
    X,Y = repeat(xx, 1, N), repeat(xx, 1, N)'
    
    seq_pairs = compute_seq_pairs(trunc_KL)
    
    φ = zeros(Float64, trunc_KL, N, N)
    λ = zeros(Float64, trunc_KL)
    
    for i = 1:trunc_KL
        φ[i, :, :] = cos.(pi * (seq_pairs[i, 1]*X + seq_pairs[i, 2]*Y))
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




#-∇(κ∇h) = f

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
    h = df\(f_2d[2:N-1,2:N-1])[:]
    
    h_2d = zeros(Float64, N, N)
    h_2d[2:N-1,2:N-1] .= reshape(h, N-2, N-2) 
    
    return h_2d
end



function compute_obs(darcy::Param_Darcy, h_2d::Array{Float64, 2})
    N = darcy.N
    obs_ΔN = darcy.obs_ΔN
    
    # X---X(1)---X(2) ... X(obs_N)---X
    obs_2d = h_2d[obs_ΔN:obs_ΔN:N-obs_ΔN,obs_ΔN:obs_ΔN:N-obs_ΔN] 
    
    return obs_2d[:]
end



function run_Darcy_ensemble(darcy::Param_Darcy, params_i::Array{Float64, 2})
    N_ens,  N_θ = size(params_i)
    
    g_ens = Vector{Float64}[]
    
    for i = 1:N_ens
        
        logκ_2d = compute_logκ_2d(darcy, params_i[i, :])
        κ_2d = exp.(logκ_2d)
        
        h_2d = solve_GWF(darcy, κ_2d)
        
        obs = compute_obs(darcy, h_2d)
        
        # g: N_ens x N_data
        push!(g_ens, obs) 
    end
    
    return hcat(g_ens...)'
end



# function Data_Gen(θ, G, Σ_η)
#     # y = Gθ + η
#     t_mean, t_cov = G*θ, Σ_η


#     @save "t_mean.jld2" t_mean
#     @save "t_cov.jld2" t_cov

#     return t_mean, t_cov

# end


function UKI_Run(t_mean, t_cov, θ_bar, θθ_cov,  darcy::Param_Darcy,  N_iter::Int64 = 100)
    parameter_names = ["logκ_2d"]
    
    ens_func(θ_ens) = run_Darcy_ensemble(darcy, θ_ens)
    
    ukiobj = UKIObj(parameter_names,
    θ_bar, 
    θθ_cov,
    t_mean, # observation
    t_cov)
    
    
    for i in 1:N_iter
        
        params_i = deepcopy(ukiobj.θ_bar[end])
        
        @info "L₂ error of params_i :", norm(darcy.u_ref[1:length(params_i)] - params_i), " / ",  norm(darcy.u_ref[1:length(params_i)])
        
        logκ_2d_i = compute_logκ_2d(darcy, params_i)
        
        @info "F error of logκ :", norm(darcy.logκ_2d - logκ_2d_i), " / ",  norm(darcy.logκ_2d )
        
        
        update_ensemble!(ukiobj, ens_func) 
        
        @info "F error of data_mismatch :", (ukiobj.g_bar[end] - ukiobj.g_t)'*(ukiobj.obs_cov\(ukiobj.g_bar[end] - ukiobj.g_t))
        
        
    end
    
    return ukiobj
    
end


function EKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov,  darcy, f,  N_ens, N_iter::Int64 = 100)
    parameter_names = ["logκ_2d"]
    
    ens_func(θ_ens) = run_Darcy_ensemble(darcy, f, θ_ens)
    
    initial_params = Array(rand(MvNormal(θ0_bar, θθ0_cov), N_ens)')
    
    ekiobj = EKIObj(parameter_names,
    initial_params, 
    θθ0_cov,
    t_mean, # observation
    t_cov)
    
    for i = 1:N_iter
        update_ensemble!(ekiobj, ens_func) 
    end
    
    return ekiobj
    
end




function Darcy_Test(darcy::Param_Darcy, N_θ::Int64= 16, N_ite::Int64 = 100, noise::Bool=false)
    @assert(N_θ <= darcy.trunc_KL)
    
    κ_2d = exp.(darcy.logκ_2d)
    h_2d = solve_GWF(darcy, κ_2d)
    
    t_mean = compute_obs(darcy, h_2d)
    if noise
        Random.seed!(123);
        noise = rand(Uniform(-0.01,0.01), length(t_mean))
        t_mean .*= (1.0 .+ noise)
    end
    
    
    t_cov = Array(Diagonal(fill(1.0, length(t_mean))))
    
    θ0_bar = zeros(Float64, N_θ)  # mean 
    
    θθ0_cov = Array(Diagonal(fill(1.0, N_θ)))
    
    ukiobj = UKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov, darcy, N_ite)
    
    return ukiobj
end


function plot_KI_error(ukiobj::UKIObj, filename::String)
    N_θ = 5 #first 3 components
    θ_bar = ukiobj.θ_bar
    θθ_cov = ukiobj.θθ_cov
    θ_bar_arr = hcat(θ_bar...)[:, 1:N_ite]
    
    θθ_cov_arr = zeros(Float64, (N_θ, N_ite))
    for i = 1:N_ite
        for j = 1:N_θ
            θθ_cov_arr[j, i] = sqrt(θθ_cov[i][j,j])
        end
    end
    ites = Array(LinRange(1, N_ite, N_ite))
    errorbar(ites, θ_bar_arr[1,:], yerr=3.0*θθ_cov_arr[1,:], errorevery = 20, fmt="--o",fillstyle="none", label=L"\theta_0")

    errorbar(ites.+2, θ_bar_arr[2,:], yerr=3.0*θθ_cov_arr[2,:], errorevery = 20,fmt="--o",fillstyle="none", label=L"\theta_1")
 
    errorbar(ites.-2, θ_bar_arr[3,:], yerr=3.0*θθ_cov_arr[3,:], errorevery = 20,fmt="--o",fillstyle="none", label=L"\theta_2")

    errorbar(ites.+4, θ_bar_arr[4,:], yerr=3.0*θθ_cov_arr[4,:], errorevery = 20,fmt="--o",fillstyle="none", label=L"\theta_3")

    errorbar(ites.-4, θ_bar_arr[5,:], yerr=3.0*θθ_cov_arr[5,:], errorevery = 20,fmt="--o",fillstyle="none", label=L"\theta_4")
    
    
    ites = Array(LinRange(1, N_ite+10, N_ite+10))
    for i = 1:N_θ
        plot(ites, fill(darcy.u_ref[i], N_ite+10), "--", color="gray")
    end
    
    xlabel("Iterations")
    legend()
    grid("on")
    tight_layout()
    savefig(filename)
    close("all")
    
end

N, L = 80, 1.0
obs_ΔN = 10
α = 2.0
τ = 3.0
KL_trunc = 256
darcy = Param_Darcy(N, obs_ΔN, L, KL_trunc, α, τ)


N_ite = 200
N_θ1, N_θ2 = 32, 8
ukiobj_1 = Darcy_Test(darcy, N_θ1, N_ite, false) 
ukiobj_2 = Darcy_Test(darcy, N_θ2, N_ite, false) 

# Plot logκ error and Data mismatch

ites = Array(LinRange(1, N_ite, N_ite))
errors = zeros(Float64, (4, N_ite))
for i = 1:N_ite
    
    errors[1, i] = norm(darcy.logκ_2d - compute_logκ_2d(darcy, ukiobj_1.θ_bar[i]))/norm(darcy.logκ_2d)
    errors[2, i] = (ukiobj_1.g_bar[i] - ukiobj_1.g_t)'*(ukiobj_1.obs_cov\(ukiobj_1.g_bar[i] - ukiobj_1.g_t))
    
    errors[3, i] = norm(darcy.logκ_2d - compute_logκ_2d(darcy, ukiobj_2.θ_bar[i]))/norm(darcy.logκ_2d)
    errors[4, i] = (ukiobj_2.g_bar[i] - ukiobj_2.g_t)'*(ukiobj_2.obs_cov\(ukiobj_2.g_bar[i] - ukiobj_2.g_t))
    
end

semilogy(ites, errors[1, :], "--o", fillstyle="none", label= "\$N_{θ}=32\$")
semilogy(ites, errors[3, :], "--o", fillstyle="none", label= "\$N_{θ}=8\$")
xlabel("Iterations")
ylabel("Relative Frobenius norm error")
#ylim((0.1,15))
grid("on")
legend()
tight_layout()
savefig("Darcy-Params-Noise.pdf")
close("all")


semilogy(ites, errors[2, :], "--o", fillstyle="none", label= "\$N_{θ}=32\$")
semilogy(ites, errors[4, :], "--o", fillstyle="none", label= "\$N_{θ}=8\$")
xlabel("Iterations")
ylabel("Data misfit")
#ylim((0.1,15))
grid("on")
legend()
tight_layout()
savefig("Darcy-Data-Mismatch.pdf")
close("all")


κ_2d = exp.(darcy.logκ_2d)
h_2d = solve_GWF(darcy, κ_2d)
plot_obs(darcy, h_2d, "Darcy-obs-ref.pdf")
plot_field(darcy, darcy.logκ_2d, "Darcy-logk-ref.pdf")
plot_field(darcy, compute_logκ_2d(darcy, ukiobj_1.θ_bar[N_ite]), "Darcy-logk-32.pdf")
plot_field(darcy, compute_logκ_2d(darcy, ukiobj_2.θ_bar[N_ite]), "Darcy-logk-8.pdf")


####################################################################################
plot_KI_error(ukiobj_1,  "darcy_error_32.pdf")
plot_KI_error(ukiobj_2,  "darcy_error_8.pdf")


@info "finished"




