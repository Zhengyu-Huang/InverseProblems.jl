using LinearAlgebra
using Distributions
using Random
using SparseArrays
using JLD2

include("../Inversion/Plot.jl")
include("../Inversion/GMGD.jl")
include("../Fluid/Darcy-2D.jl")


#=
A hardcoding source function, 
which assumes the computational domain is
[0 1]×[0 1]
f(x,y) = f(y),
which dependes only on y
=#
# function compute_f_2d(yy::Array{FT, 1}) where {FT<:AbstractFloat}
#     N = length(yy)
#     f_2d = zeros(FT, N, N)

#     for i = 1:N
#             f_2d[:,i] .= 1000.0 * (2.0 .+ cos(4pi*yy[i]) * sin.(pi*yy))
#     end
#     return f_2d
# end

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


N, L = 81, 1.0
obs_ΔNx, obs_ΔNy = 5, 5
d = 2.0
τ = 3.0
N_KL = 16
N_θ = 16
σ_0 = 5.0
darcy = Setup_Param(N, L, N_KL, obs_ΔNx, obs_ΔNy, N_θ, d, τ, σ_0; seed=111)



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
σ_η = 1.0
Σ_η = σ_η^2 * Array(Diagonal(fill(1.0, N_y)))
Random.seed!(123);
y = y_noiseless  #+ rand(Normal(0, σ_η), N_y)

 
N_iter = 50
 
μ_0 = zeros(Float64, N_θ)  # prior/initial mean 

Σ_0 = Array(Diagonal(fill(σ_0^2, N_θ)))  # prior/initial covariance
darcy.N_y = N_f = (N_y + N_θ)




# compute posterior distribution by GMKI
N_modes = 2
θ0_w  = fill(1.0, N_modes)/N_modes
θ0_mean, θθ0_cov  = zeros(N_modes, N_θ), zeros(N_modes, N_θ, N_θ)



Random.seed!(3);
# θ_ref = darcy.θ_ref
# θ0_mean[1, :]    .= θ_ref #rand(Normal(0, σ_0), N_θ) 
# θθ0_cov[1, :, :] .= Array(Diagonal(fill(σ_0^2, N_θ)))
# θ0_mean[2, :]    .= -θ_ref #rand(Normal(0, σ_0), N_θ) 
# θθ0_cov[2, :, :] .= Array(Diagonal(fill(σ_0^2, N_θ)))
# θ0_mean[3, :]    .= 1.01*θ_ref # rand(Normal(0, σ_0), N_θ) 
# neg_ind = [2;3;6;10;11;12;14]
# θ0_mean[3, neg_ind] *= -1.0 
# θθ0_cov[3, :, :] .= Array(Diagonal(fill(σ_0^2, N_θ)))
# θ0_mean[4, :]    .= 1.3*θ_ref # rand(Normal(0, σ_0), N_θ) 
# neg_ind = [2;3;6;10;11;12;14]
# θ0_mean[4, neg_ind] *= -1.0 
# θθ0_cov[4, :, :] .= Array(Diagonal(fill(σ_0^2, N_θ)))

for im = 1:N_modes
    θ0_mean[im, :]    .= rand(Normal(0, σ_0), N_θ) 
    θθ0_cov[im, :, :] .= Array(Diagonal(fill(σ_0^2, N_θ)))
end
for im = 1:div(N_modes,2)
    θ0_mean[N_modes-im+1, :]    .= -θ0_mean[im, :]
end


dt = 0.5
T = dt*N_iter
func_args = (y, μ_0, σ_η, σ_0)
func_F(x) = darcy_F(darcy, func_args, x)
        
gmgdobj = GMGD_Run(
        func_F, 
        T,
        N_iter,
        # Initial condition
        θ0_w, θ0_mean, θθ0_cov;
        sqrt_matrix_type = "Cholesky",
        # setup for Gaussian mixture part
        quadrature_type_GM = "mean_point",
        # setup for potential function part
        Bayesian_inverse_problem = true, 
        N_f = N_f,
        quadrature_type = "unscented_transform",
        c_weight_BIP = 1.0e-3,
        w_min=1e-2)


    
@save "gmgdobj-Darcy.jld2" gmgdobj





N_ens = 2N_θ + 1
# fig, (ax1, ax2, ax3) = PyPlot.subplots(ncols=3, figsize=(16,5))
# ites = Array(LinRange(0, N_iter, N_iter+1))
# errors = zeros(Float64, (3, N_iter+1, 4))
# # GMKI-1
# for i = 1:N_iter+1
#     errors[1, i, 1] = norm(darcy.logκ_2d - compute_logκ_2d(darcy, gmgdobj.x_mean[i]))/norm(darcy.logκ_2d)
#     errors[2, i, 1] = 0.5*(gmgdobj.y_pred[i] - gmgdobj.y)'*(gmgdobj.Σ_η\(gmgdobj.y_pred[i] - gmgdobj.y))
#     errors[3, i, 1] = norm(gmgdobj.θθ_cov[i])
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
fig_logk, ax_logk = PyPlot.subplots(ncols = N_modes+2, sharex=true, sharey=true, figsize=(4*(N_modes+2),4))
for ax in ax_logk ;  ax.set_xticks([]) ; ax.set_yticks([]) ; end
color_lim = (minimum(darcy.logκ_2d), maximum(darcy.logκ_2d))

plot_field(darcy, darcy.logκ_2d, color_lim, ax_logk[1]) 
ax_logk[1].set_title("Truth")
plot_field(darcy, darcy.logκ_2d[end:-1:1, :],  color_lim, ax_logk[2]) 
ax_logk[2].set_title("Truth (mirrored)")
for im = 1:N_modes
    plot_field(darcy, compute_logκ_2d(darcy, gmgdobj.x_mean[N_iter][im,:]),  color_lim, ax_logk[im+2]) 
    ax_logk[im+2].set_title("Mode " * string(im))
end


fig_logk.tight_layout()
fig_logk.savefig("Darcy-2D-logk.pdf")




N_ens = 2N_θ + 1
fig, (ax1, ax2, ax3, ax4) = PyPlot.subplots(ncols=4, figsize=(20,5))
ites = Array(LinRange(0, N_iter-1, N_iter))
errors = zeros(Float64, (3, N_iter, N_modes))

for m = 1:N_modes
    for i = 1:N_iter

        logκ_2d_truth = darcy.logκ_2d
        # if m in [2,3]
            
        # else
        #     logκ_2d_truth = darcy.logκ_2d[end:-1:1, :]
        # end
        errors[1, i, m] = norm(logκ_2d_truth - compute_logκ_2d(darcy, gmgdobj.x_mean[i][m,:]))/norm(darcy.logκ_2d)
        errors[2, i, m] = gmgdobj.phi_r_pred[i][m]
        errors[3, i, m] = norm(gmgdobj.xx_cov[i][m,:,:])
    end
end

linestyles = ["o"; "x"; "s"; "*"]
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


x_w = exp.(hcat(gmgdobj.logx_w...))
for m = 1: N_modes
    ax4.plot(ites, x_w[m, 1:N_iter], marker=linestyles[m], color = "C"*string(m), fillstyle="none", markevery=markevery, label= "mode "*string(m))
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
    ax.scatter(θ_ind, gmgdobj.x_mean[N_iter][m,θ_ind], s = 50, marker="o", color="C"*string(m), facecolors="none", label="Mode "*string(m))
end

Nx = 1000
scale = 1
for i in θ_ind
    x_min = minimum(gmgdobj.x_mean[N_iter][:,i] .- 3sqrt.(gmgdobj.xx_cov[N_iter][:,i,i]))
    x_max = maximum(gmgdobj.x_mean[N_iter][:,i] .+ 3sqrt.(gmgdobj.xx_cov[N_iter][:,i,i]))
        
    xxs = zeros(N_modes, Nx)  
    zzs = zeros(N_modes, Nx)  
    for m =1:N_modes
        xxs[m, :], zzs[m, :] = Gaussian_1d(gmgdobj.x_mean[N_iter][m,i], gmgdobj.xx_cov[N_iter][m,i,i], Nx, x_min, x_max)
        zzs[m, :] *= exp(gmgdobj.logx_w[N_iter][m])

        @info "ind ", i, " mode ", m, " mean ", gmgdobj.x_mean[N_iter][m,i],  " std ", sqrt(gmgdobj.xx_cov[N_iter][m,i,i])
    end
    
    label = nothing
    if i == 1
        label = "DF-GMGD"
    end
    ax.plot(sum(zzs, dims=1)'/scale .+ i, xxs[1,:], linestyle="-", color="C0", fillstyle="none", label=label)
    ax.plot(fill(i, Nx), xxs[1,:], linestyle=":", color="black", fillstyle="none")
        
end
ax.set_xticks(θ_ind)
ax.set_xlabel(L"\theta" * " indices")
ax.legend(loc="center left", bbox_to_anchor=(0.95, 0.5))
fig.tight_layout()
fig.savefig("Darcy-2D-density.pdf")

