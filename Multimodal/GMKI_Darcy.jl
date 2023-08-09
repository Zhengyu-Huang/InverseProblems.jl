using LinearAlgebra
using Distributions
using Random
using SparseArrays
using JLD2

include("../Inversion/Plot.jl")
include("../Inversion/KalmanInversion.jl")
include("../Fluid/Darcy-2D.jl")


#=
A hardcoding source function, 
which assumes the computational domain is
[0 1]×[0 1]
f(x,y) = f(y),
which dependes only on y
=#
function compute_f_2d(yy::Array{FT, 1}) where {FT<:AbstractFloat}
    N = length(yy)
    f_2d = zeros(FT, N, N)

    for i = 1:N
            f_2d[:,i] .= 1000.0 * sin(4pi*yy[i])
    end
    return f_2d
end


function generate_θ_KL(xx::Array{FT,1}, N_KL::IT, d::FT=2.0, τ::FT=3.0; seed::IT=123) where {FT<:AbstractFloat, IT<:Int}
    N = length(xx)
    X,Y = repeat(xx, 1, N), repeat(xx, 1, N)'
    
    seq_pairs = compute_seq_pairs(N_KL)
    
    φ = zeros(FT, N_KL, N, N)
    λ = zeros(FT, N_KL)
    
    for i = 1:N_KL
        if (seq_pairs[i, 1] == 0 && seq_pairs[i, 2] == 0)
            φ[i, :, :] .= 1.0
        elseif (seq_pairs[i, 1] == 0)
            φ[i, :, :] = sqrt(2)*cos.(pi * (seq_pairs[i, 2]*Y))
        elseif (seq_pairs[i, 2] == 0)
            φ[i, :, :] = sqrt(2)*cos.(pi * (seq_pairs[i, 1]*X))
        else
            φ[i, :, :] = 2*cos.(pi * (seq_pairs[i, 1]*X)) .*  cos.(pi * (seq_pairs[i, 2]*Y))
        end

        λ[i] = (pi^2*(seq_pairs[i, 1]^2 + seq_pairs[i, 2]^2) + τ^2)^(-d)
    end
    
    Random.seed!(seed);
#     rng = MersenneTwister(seed)
#     θ_ref = rand(rng, Normal(0, 10), N_KL)
    θ_ref = rand(Normal(0, 10), N_KL)
    
    logκ_2d = zeros(FT, N, N)
    for i = 1:N_KL
        logκ_2d .+= θ_ref[i]*sqrt(λ[i])*φ[i, :, :]
    end
    
    return logκ_2d, φ, λ, θ_ref
end

#=
Compute observation values
=#
function compute_obs(darcy::Setup_Param{FT, IT}, h_2d::Array{FT, 2}) where {FT<:AbstractFloat, IT<:Int}
    # X---X(1)---X(2) ... X(obs_N)---X
    obs_2d = h_2d[darcy.x_locs, darcy.y_locs] 
    
    Nx_o, Ny_o = size(obs_2d)
    
    obs_2d_sym = (obs_2d[1:div(Nx_o+1, 2), :] + obs_2d[end:-1:div(Nx_o, 2)+1, :]) / 2.0
    
    # obs_2d_sym = (obs_2d[:, 1:div(Ny_o+1, 2)] + obs_2d[:, end:-1:div(Ny_o, 2)+1]) / 2.0
    
    return obs_2d_sym[:]
end


function plot_field(darcy::Setup_Param{FT, IT}, u_2d::Array{FT, 2}, plot_obs::Bool,  filename::String = "None") where {FT<:AbstractFloat, IT<:Int}
    N = darcy.N
    xx = darcy.xx

    X,Y = repeat(xx, 1, N), repeat(xx, 1, N)'
    pcolormesh(X, Y, u_2d, cmap="viridis")
    colorbar()

    if plot_obs
        x_obs, y_obs = X[darcy.x_locs[1:div(length(darcy.x_locs)+1,2)], darcy.y_locs][:], Y[darcy.x_locs[1:div(length(darcy.x_locs)+1,2)], darcy.y_locs][:] 
        scatter(x_obs, y_obs, color="black")
        
        x_obs, y_obs = X[darcy.x_locs[div(length(darcy.x_locs)+1,2)+1:end], darcy.y_locs][:], Y[darcy.x_locs[div(length(darcy.x_locs)+1,2)+1:end], darcy.y_locs][:] 
        scatter(x_obs, y_obs, color="black", facecolors="none")
    end

    tight_layout()
    if filename != "None"
        savefig(filename)
    end
end


N, L = 80, 1.0
obs_ΔN = 10
d = 2.0
τ = 3.0
N_KL = 128
N_θ = 128
darcy = Setup_Param(N, L, N_KL, obs_ΔN, N_θ, d, τ)



κ_2d = exp.(darcy.logκ_2d)
h_2d = solve_Darcy_2D(darcy, κ_2d)
y_noiseless = compute_obs(darcy, h_2d)

figure(1)
plot_field(darcy, h_2d, true, "Darcy-2D-obs.pdf")
figure(2)
plot_field(darcy, darcy.logκ_2d, false, "Darcy-2D-logk-ref.pdf")
    


# initial mean and covariance
# GMKI
N_y = length(y_noiseless)
σ_η = 1.0^2
Σ_η = σ_η^2 * Array(Diagonal(fill(1.0, N_y)))
Random.seed!(123);
y = y_noiseless + rand(Normal(0, σ_η), N_y)

update_freq = 1
N_iter = 50
α_reg  = 1.0

N_θ = darcy.N_θ = 128
μ_0 = zeros(Float64, N_θ)  # prior/initial mean 
Σ_0 = Array(Diagonal(fill(10.0^2, N_θ)))  # prior/initial covariance

aug_y     = [y; μ_0] 
aug_Σ_η   = [Σ_η zeros(Float64, N_y, N_θ); zeros(Float64, N_θ, N_y) Σ_0]  
darcy.N_y = (N_y + N_θ)




# compute posterior distribution by GMKI
update_freq = 1
N_modes = 3
θ0_w  = fill(1.0, N_modes)/N_modes
θ0_mean, θθ0_cov  = zeros(N_modes, N_θ), zeros(N_modes, N_θ, N_θ)



Random.seed!(63);
σ_0 = 10
θ0_mean[1, :]    .= rand(Normal(0, σ_0), N_θ) 
θθ0_cov[1, :, :] .= Array(Diagonal(fill(1.0^2, N_θ)))
θ0_mean[2, :]    .= rand(Normal(0, σ_0), N_θ) 
θθ0_cov[2, :, :] .= Array(Diagonal(fill(1.0^2, N_θ)))
θ0_mean[3, :]    .= rand(Normal(0, σ_0), N_θ) 
θθ0_cov[3, :, :] .= Array(Diagonal(fill(1.0^2, N_θ)))




Δt = 0.5
gmkiobj = GMKI_Run(darcy, aug_forward, θ0_w, θ0_mean, θθ0_cov, aug_y, aug_Σ_η, Δt, update_freq, N_iter; unscented_transform="modified-2n+1")
    
@save "gmkiobj-Darcy.jld2" gmkiobj





N_ens = 2N_θ + 1
# fig, (ax1, ax2, ax3) = PyPlot.subplots(ncols=3, figsize=(16,5))
# ites = Array(LinRange(0, N_iter, N_iter+1))
# errors = zeros(Float64, (3, N_iter+1, 4))
# # GMKI-1
# for i = 1:N_iter+1
#     errors[1, i, 1] = norm(darcy.logκ_2d - compute_logκ_2d(darcy, gmkiobj.θ_mean[i]))/norm(darcy.logκ_2d)
#     errors[2, i, 1] = 0.5*(gmkiobj.y_pred[i] - gmkiobj.y)'*(gmkiobj.Σ_η\(gmkiobj.y_pred[i] - gmkiobj.y))
#     errors[3, i, 1] = norm(gmkiobj.θθ_cov[i])
# end

# ax1.plot(ites, errors[1, :, 1], "-.x", color = "C0", fillstyle="none", markevery=1, label= "GMKI-1 (J=$N_ens)")
# ax1.set_xlabel("Iterations")
# ax1.set_ylabel("Rel. error of loga")
# ax1.legend()

# ax2.plot(ites, errors[2, :, 1], "-.x",  color = "C0", fillstyle="none", markevery=1, label= "GMKI-1 (J=$N_ens)")
# ax2.set_xlabel("Iterations")
# ax2.set_ylabel("Optimization error")
# ax2.legend()

# ax3.plot(ites, errors[3, :, 1], "-.x",  color = "C0", fillstyle="none", markevery=1, label= "GMKI-1 (J=$N_ens)")
# ax3.set_xlabel("Iterations")
# ax3.set_ylabel("Frobenius norm of covariance")
# ax3.legend()
# fig.tight_layout()


# visulize the log permeability field
fig_logk, ax_logk = PyPlot.subplots(ncols = 4, sharex=true, sharey=true, figsize=(20,5))
for ax in ax_logk ;  ax.set_xticks([]) ; ax.set_yticks([]) ; end
color_lim = (minimum(darcy.logκ_2d), maximum(darcy.logκ_2d))

plot_field(darcy, darcy.logκ_2d, color_lim, ax_logk[1]) 
ax_logk[1].set_title("Truth")
plot_field(darcy, compute_logκ_2d(darcy, gmkiobj.θ_mean[N_iter][1,:]),  color_lim, ax_logk[2]) 
ax_logk[2].set_title("Mode 1")
plot_field(darcy, compute_logκ_2d(darcy, gmkiobj.θ_mean[N_iter][2,:]),  color_lim, ax_logk[3]) 
ax_logk[3].set_title("Mode 2")
plot_field(darcy, compute_logκ_2d(darcy, gmkiobj.θ_mean[N_iter][3,:]),  color_lim, ax_logk[4]) 
ax_logk[4].set_title("Mode 3")


fig_logk.tight_layout()
fig_logk.savefig("Darcy-2D-logk-LR.pdf")




N_ens = 2N_θ + 1
fig, (ax1, ax2, ax3, ax4) = PyPlot.subplots(ncols=4, figsize=(20,5))
ites = Array(LinRange(0, N_iter-1, N_iter))
errors = zeros(Float64, (3, N_iter, N_modes))

for m = 1:N_modes
    for i = 1:N_iter
        if m in [2,3]
            logκ_2d_truth = darcy.logκ_2d
        else
            logκ_2d_truth = darcy.logκ_2d[end:-1:1, :]
        end
        errors[1, i, m] = norm(logκ_2d_truth - compute_logκ_2d(darcy, gmkiobj.θ_mean[i][m,:]))/norm(darcy.logκ_2d)
        errors[2, i, m] = 0.5*(gmkiobj.y_pred[i][m,:] - gmkiobj.y)'*(gmkiobj.Σ_η\(gmkiobj.y_pred[i][m,:] - gmkiobj.y))
        errors[3, i, m] = norm(gmkiobj.θθ_cov[i][m,:,:])
    end
end

linestyles = ["o"; "x"; "s"]
markevery = 5
for m = 1: N_modes
    ax1.plot(ites, errors[1, :, m], marker=linestyles[m], color = "C"*string(m), fillstyle="none", markevery=markevery, label= "mode "*string(m))
end
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Rel. error of a(x)")
ax1.legend()

for m = 1: N_modes
    ax2.semilogy(ites, errors[2, :, m], marker=linestyles[m], color = "C"*string(m), fillstyle="none", markevery=markevery, label= "mode "*string(m))
end
ax2.set_xlabel("Iterations")
ax2.set_ylabel(L"\Phi_R")
ax2.legend()

for m = 1: N_modes
    ax3.plot(ites, errors[3, :, m], marker=linestyles[m], color = "C"*string(m), fillstyle="none", markevery=markevery, label= "mode "*string(m))
end
ax3.set_xlabel("Iterations")
ax3.set_ylabel("Frobenius norm of covariance")
ax3.legend()


θ_w = exp.(hcat(gmkiobj.logθ_w...))
for m = 1: N_modes
    ax4.plot(ites, θ_w[m, 1:N_iter], marker=linestyles[m], color = "C"*string(m), fillstyle="none", markevery=markevery, label= "mode "*string(m))
end
ax4.set_xlabel("Iterations")
ax4.set_ylabel("Weights")
ax4.legend()
fig.tight_layout()
fig.savefig("Darcy-2D-convergence.pdf")


fig, ax = PyPlot.subplots(ncols=1, figsize=(16,5))
θ_ref = darcy.θ_ref

n_ind = 16
θ_ind = Array(1:n_ind)
ax.scatter(θ_ind, θ_ref[θ_ind], s = 100, marker="x", color="black", label="Truth")
for m = 1:N_modes
    ax.scatter(θ_ind, gmkiobj.θ_mean[N_iter][m,θ_ind], s = 50, marker="o", color="C"*string(m), facecolors="none", label="Mode "*string(m))
end

Nx = 1000
for i in θ_ind
    θ_min = minimum(gmkiobj.θ_mean[N_iter][:,i] .- 3sqrt.(gmkiobj.θθ_cov[N_iter][:,i,i]))
    θ_max = maximum(gmkiobj.θ_mean[N_iter][:,i] .+ 3sqrt.(gmkiobj.θθ_cov[N_iter][:,i,i]))
        
    xxs = zeros(N_modes, Nx)  
    zzs = zeros(N_modes, Nx)  
    for m =1:N_modes
        xxs[m, :], zzs[m, :] = Gaussian_1d(gmkiobj.θ_mean[N_iter][m,i], gmkiobj.θθ_cov[N_iter][m,i,i], Nx, θ_min, θ_max)
        zzs[m, :] *= exp(gmkiobj.logθ_w[N_iter][m]) * 3
    end
    label = nothing
    if i == 1
        label = "GMKI"
    end
    ax.plot(sum(zzs, dims=1)' .+ i, xxs[1,:], linestyle="-", color="C0", fillstyle="none", label=label)
    ax.plot(fill(i, Nx), xxs[1,:], linestyle=":", color="black", fillstyle="none")
        
end
ax.set_xticks(θ_ind)
ax.set_xlabel(L"\theta" * " indices")
ax.legend(loc="center left", bbox_to_anchor=(0.95, 0.5))
fig.tight_layout()
fig.savefig("Darcy-2D-density.pdf")

