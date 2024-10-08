using LinearAlgebra
using Distributions
using Random
using SparseArrays

include("../Inversion/Plot.jl")
include("../Inversion/KalmanInversion.jl")
include("../Inversion/RWMCMC.jl")
include("Darcy-2D.jl")
using JLD


function construct_cov(x::Array{FT,2}) where {FT<:AbstractFloat}
    x_mean = dropdims(mean(x, dims=1), dims=1)
    N_ens, N_x = size(x)
    x_cov = zeros(FT, N_x, N_x)
    for i = 1: N_ens
        x_cov .+= (x[i,:] - x_mean)*(x[i,:] - x_mean)'
    end
    return x_cov/(N_ens - 1)
end

function construct_mean(x::Array{FT,2}) where {FT<:AbstractFloat}    
    x_mean = dropdims(mean(x, dims=1), dims=1)
    return x_mean
end

N, L = 80, 1.0
obs_ΔN = 10
d = 2.0
τ = 3.0
N_KL = 128
N_θ = 0
darcy = Setup_Param(N, L, N_KL, obs_ΔN, N_θ, d, τ)

κ_2d = exp.(darcy.logκ_2d)
h_2d = solve_Darcy_2D(darcy, κ_2d)
y_noiseless = compute_obs(darcy, h_2d)

figure(1)
plot_field(darcy, h_2d, true, "Darcy-2D-obs.pdf")
figure(2)
plot_field(darcy, darcy.logκ_2d, false, "Darcy-2D-logk-ref.pdf")
    

# N_θ = 32 case with 5% Gaussian error

noise_level = 0.05
N_y = length(y_noiseless)
# observation
y = copy(y_noiseless)
Random.seed!(123);
for i = 1:length(y)
    # noise = rand(Normal(0, noise_level*y[i]))
    noise = rand(Normal(0, 1.0))
    y[i] += noise
end
Σ_η = Array(Diagonal(fill(1.0, length(y))))

# initial mean and covariance
# UKI
update_freq = 1
N_iter = 10
α_reg  = 1.0

N_θ = darcy.N_θ = 29
θ0_mean = zeros(Float64, N_θ)  # mean 
θθ0_cov = Array(Diagonal(fill(1.0, N_θ)))
θθ0_cov_sqrt = Array(Diagonal(fill(1.0, N_θ)))
aug_y     = [y; zeros(Float64, N_θ)] 
aug_Σ_η   = [Σ_η zeros(Float64, N_y, N_θ); zeros(Float64, N_θ, N_y) θθ0_cov]  
darcy.N_y = (N_y + N_θ)
ukiobj = UKI_Run(darcy,  aug_forward, θ0_mean, θθ0_cov, aug_y, aug_Σ_η, α_reg, update_freq, N_iter+1; unscented_transform = "modified-n+2" );
@info "Finish modified-n+2"

N_θ = darcy.N_θ = 15
θ0_mean = zeros(Float64, N_θ)  # mean 
θθ0_cov = Array(Diagonal(fill(1.0, N_θ)))
θθ0_cov_sqrt = Array(Diagonal(fill(1.0, N_θ)))
aug_y     = [y; zeros(Float64, N_θ)] 
aug_Σ_η   = [Σ_η zeros(Float64, N_y, N_θ); zeros(Float64, N_θ, N_y) θθ0_cov]  
darcy.N_y = (N_y + N_θ)
uki_2np1_obj = UKI_Run(darcy,  aug_forward, θ0_mean, θθ0_cov, aug_y, aug_Σ_η, α_reg, update_freq, N_iter+1; unscented_transform = "modified-2n+1");
@info "Finish modified-2n+1"

γ_ω = 1.0
γ_η = (γ_ω + 1)/γ_ω

# EKI
N_ens = 31
N_θ = darcy.N_θ = N_KL
θ0_mean = zeros(Float64, N_KL)  # mean 
θθ0_cov = Array(Diagonal(fill(1.0, N_KL)))
θθ0_cov_sqrt = Array(Diagonal(fill(1.0, N_KL)))
aug_y     = [y; zeros(Float64, N_θ)] 
aug_Σ_η   = [Σ_η zeros(Float64, N_y, N_θ); zeros(Float64, N_θ, N_y) θθ0_cov]  
darcy.N_y = (N_y + N_θ)
filter_type = "EAKI"
eakiobj = EKI_Run(darcy,  aug_forward, filter_type, θ0_mean, θθ0_cov_sqrt, N_ens, aug_y, aug_Σ_η, γ_ω, γ_η, N_iter+1);
@info "Finish EAKI"

filter_type = "ETKI"
etkiobj = EKI_Run(darcy,  aug_forward, filter_type, θ0_mean, θθ0_cov_sqrt, N_ens, aug_y, aug_Σ_η, γ_ω, γ_η, N_iter+1);
@info "Finish ETKI"

##################################################

fig, (ax1, ax2, ax3) = PyPlot.subplots(ncols=3, figsize=(16,5))
ites = Array(LinRange(0, N_iter, N_iter+1))
errors = zeros(Float64, (3, N_iter+1, 4))
# UKI-1
for i = 1:N_iter+1
    errors[1, i, 1] = norm(darcy.logκ_2d - compute_logκ_2d(darcy, ukiobj.θ_mean[i]))/norm(darcy.logκ_2d)
    errors[2, i, 1] = 0.5*(ukiobj.y_pred[i] - ukiobj.y)'*(ukiobj.Σ_η\(ukiobj.y_pred[i] - ukiobj.y))
    errors[3, i, 1] = norm(ukiobj.θθ_cov[i])
end

# EAKI
for i = 1:N_iter+1
    errors[1, i, 2] = norm(darcy.logκ_2d - compute_logκ_2d(darcy, construct_mean(eakiobj.θ[i])))/norm(darcy.logκ_2d)
    errors[2, i, 2] = 0.5*(eakiobj.y_pred[i] - eakiobj.y)'*(eakiobj.Σ_η\(eakiobj.y_pred[i] - eakiobj.y))
    errors[3, i, 2] = norm(construct_cov(eakiobj.θ[i])) 
end

# ETKI
for i = 1:N_iter+1
    errors[1, i, 3] = norm(darcy.logκ_2d - compute_logκ_2d(darcy, construct_mean(etkiobj.θ[i])))/norm(darcy.logκ_2d)
    errors[2, i, 3] = 0.5*(etkiobj.y_pred[i] - etkiobj.y)'*(etkiobj.Σ_η\(etkiobj.y_pred[i] - etkiobj.y))
    errors[3, i, 3] = norm(construct_cov(etkiobj.θ[i]))
end

# UKI-2
for i = 1:N_iter+1
    errors[1, i, 4] = norm(darcy.logκ_2d - compute_logκ_2d(darcy, uki_2np1_obj.θ_mean[i]))/norm(darcy.logκ_2d)
    errors[2, i, 4] = 0.5*(uki_2np1_obj.y_pred[i] - uki_2np1_obj.y)'*(uki_2np1_obj.Σ_η\(uki_2np1_obj.y_pred[i] - uki_2np1_obj.y))
    errors[3, i, 4] = norm(uki_2np1_obj.θθ_cov[i])
end

ax1.plot(ites, errors[1, :, 1], "-.x", color = "C0", fillstyle="none", markevery=1, label= "UKI-1 (J=$N_ens)")
ax1.plot(ites, errors[1, :, 4], "-o",  color = "C0", fillstyle="none", markevery=1, label= "UKI-2 (J=$N_ens)")
ax1.plot(ites, errors[1, :, 2], "-^",  color = "C2", fillstyle="none", markevery=1, label= "EAKI (J=$N_ens)")
ax1.plot(ites, errors[1, :, 3], "-d",  color = "C3", fillstyle="none", markevery=1, label= "ETKI (J=$N_ens)")
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Rel. error of loga")
ax1.legend()

ax2.plot(ites, errors[2, :, 1], "-.x",  color = "C0", fillstyle="none", markevery=1, label= "UKI-1 (J=$N_ens)")
ax2.plot(ites, errors[2, :, 4], "-o",   color = "C0", fillstyle="none", markevery=1, label= "UKI-2 (J=$N_ens)")
ax2.plot(ites, errors[2, :, 2], "-^",   color = "C2", fillstyle="none", markevery=1, label= "EAKI (J=$N_ens)")
ax2.plot(ites, errors[2, :, 3], "-d",   color = "C3", fillstyle="none", markevery=1, label= "ETKI (J=$N_ens)")
ax2.set_xlabel("Iterations")
ax2.set_ylabel("Optimization error")
ax2.legend()

ax3.plot(ites, errors[3, :, 1], "-.x",  color = "C0", fillstyle="none", markevery=1, label= "UKI-1 (J=$N_ens)")
ax3.plot(ites, errors[3, :, 4], "-o",   color = "C0", fillstyle="none", markevery=1, label= "UKI-2 (J=$N_ens)")
ax3.plot(ites, errors[3, :, 2], "-^",   color = "C2", fillstyle="none", markevery=1, label= "EAKI (J=$N_ens)")
ax3.plot(ites, errors[3, :, 3], "-d",   color = "C3", fillstyle="none", markevery=1, label= "ETKI (J=$N_ens)")
ax3.set_xlabel("Iterations")
ax3.set_ylabel("Frobenius norm of covariance")
ax3.legend()
fig.tight_layout()
fig.savefig("Darcy-2D-Loss-LR.pdf")
##################################################

# compute posterior distribution by MCMC
# (uninformative) prior mean and covariance
μ0 = θ0_mean # θ_ref #
Σ0 = Array(Diagonal(fill(1.0^2.0, N_θ)))

log_likelihood_func(θ) = log_likelihood(darcy, θ, forward, y,  Σ_η)
N_iter_MCMC , n_burn_in= 2*10^6, 5*10^5
# N_iter_MCMC , n_burn_in= 2*10^3, 5*10^2


# Run
# us = PCN_Run(log_likelihood_func, μ0, Σ0, 0.04, N_iter_MCMC);

# uki_θ_mean  = ukiobj.θ_mean[end]
# uki_θθ_cov = ukiobj.θθ_cov[end]
# uki_θθ_std = sqrt.(diag(ukiobj.θθ_cov[end]))

# mcmc_θ_mean = mean(us[n_burn_in:N_iter_MCMC, :], dims=1)[:]
# mcmc_θθ_std = std(us[n_burn_in:N_iter_MCMC, :], dims=1)[:]

# θ_cov_ind = Array(1:div(N_θ*(N_θ+1), 2))
# uki_θiθj_cov, mcmc_θiθj_cov = zeros(Float64, div(N_θ*(N_θ+1), 2)), zeros(Float64, div(N_θ*(N_θ+1), 2))
# ind_ij = 1
# for i = 1:N_θ
#     for j = i:N_θ
#         global ind_ij
#         uki_θiθj_cov[ind_ij]   = uki_θθ_cov[i,j]
#         mcmc_θiθj_cov[ind_ij] = (us[n_burn_in:N_iter_MCMC, i] .- mcmc_θ_mean[i])' * (us[n_burn_in:N_iter_MCMC, j] .- mcmc_θ_mean[j])/(N_iter_MCMC - n_burn_in)

#         ind_ij += 1
#     end
# end

# save("Darcy-2D.jld2", "mean", mcmc_θ_mean, "std",  mcmc_θθ_std)

uki_θ_mean = ukiobj.θ_mean[end]
uki_θθ_cov = ukiobj.θθ_cov[end]
uki_θθ_std = sqrt.(diag(ukiobj.θθ_cov[end]))

uki_2np1_θ_mean = uki_2np1_obj.θ_mean[end]
uki_2np1_θθ_cov = uki_2np1_obj.θθ_cov[end]
uki_2np1_θθ_std = sqrt.(diag(uki_2np1_obj.θθ_cov[end]))




# θ_cov_ind = Array(1:div(N_θ*(N_θ+1), 2))
# uki_θiθj_cov  = zeros(Float64, div(N_θ*(N_θ+1), 2))
# ind_ij = 1
# for i = 1:N_θ
#     for j = i:N_θ
#         global ind_ij
#         uki_θiθj_cov[ind_ij]   = uki_θθ_cov[i,j]
#         ind_ij += 1
#     end
# end


# 
eaki_θ_mean = construct_mean(eakiobj.θ[end])
eaki_θθ_cov = construct_cov(eakiobj.θ[end])
eaki_θθ_std = sqrt.(diag(eaki_θθ_cov))
etki_θ_mean = construct_mean(etkiobj.θ[end])
etki_θθ_cov = construct_cov(etkiobj.θ[end])
etki_θθ_std = sqrt.(diag(etki_θθ_cov))

dic = load("Darcy-2D.jld2")
mcmc_θ_mean = dic["mean"] 
mcmc_θθ_std = dic["std"] 


fig, ax = PyPlot.subplots(figsize=(15,5))
θ_ind = Array(1:64) #Array(1:N_θ)
θ_ref = darcy.θ_ref



ax.plot(θ_ind , θ_ref[θ_ind], "--o", color="grey", fillstyle="none", label="Truth")
θ_ind = Array(1:64)
ax.plot(θ_ind , mcmc_θ_mean[θ_ind],"-s", color="C1", fillstyle="none" , label="MCMC")
ax.plot(θ_ind , mcmc_θ_mean[θ_ind] + 3.0*mcmc_θθ_std[θ_ind], fillstyle="none", "--s", color ="C1")
ax.plot(θ_ind , mcmc_θ_mean[θ_ind] - 3.0*mcmc_θθ_std[θ_ind], fillstyle="none", "--s", color ="C1")


θ_ind = Array(1:length(uki_θ_mean))
ax.plot(θ_ind , uki_θ_mean[θ_ind],"-x", color="C0", fillstyle="none",  label="UKI-1")
ax.plot(θ_ind , uki_θ_mean[θ_ind] + 3.0*uki_θθ_std[θ_ind], fillstyle="none", "--x", color="C0")
ax.plot(θ_ind , uki_θ_mean[θ_ind] - 3.0*uki_θθ_std[θ_ind], fillstyle="none", "--x", color="C0")

θ_ind = Array(1:length(uki_2np1_θ_mean))
ax.plot(θ_ind , uki_2np1_θ_mean[θ_ind],"-o", color="C0", fillstyle="none",  label="UKI-2")
ax.plot(θ_ind , uki_2np1_θ_mean[θ_ind] + 3.0*uki_2np1_θθ_std[θ_ind], fillstyle="none", "--o", color="C0")
ax.plot(θ_ind , uki_2np1_θ_mean[θ_ind] - 3.0*uki_2np1_θθ_std[θ_ind], fillstyle="none", "--o", color="C0")



θ_ind = Array(1:64)
ax.plot(θ_ind , eaki_θ_mean[θ_ind],"-^", color="C2", fillstyle="none",  label="EAKI")
ax.plot(θ_ind , eaki_θ_mean[θ_ind] + 3.0*eaki_θθ_std[θ_ind], fillstyle="none", "--^", color="C2")
ax.plot(θ_ind , eaki_θ_mean[θ_ind] - 3.0*eaki_θθ_std[θ_ind], fillstyle="none", "--^", color="C2")

θ_ind = Array(1:64)
ax.plot(θ_ind , etki_θ_mean[θ_ind],"-d", color="C3", fillstyle="none",  label="ETKI")
ax.plot(θ_ind , etki_θ_mean[θ_ind] + 3.0*etki_θθ_std[θ_ind], fillstyle="none",  "--d",color="C3")
ax.plot(θ_ind , etki_θ_mean[θ_ind] - 3.0*etki_θθ_std[θ_ind], fillstyle="none",  "--d", color="C3")




ax.legend(bbox_to_anchor=(0.95, 0.8))
# plot MCMC results 
ax.set_xlabel("θ indices")
fig.tight_layout()
fig.savefig("Darcy-2D-theta-LR.pdf")

# fig, ax = PyPlot.subplots(figsize=(18,6))

# θ_ind = Array(1:1000)
# ax.plot(θ_cov_ind[θ_ind] , uki_θiθj_cov[θ_ind],  "*", color="red", fillstyle="none",  label="UKI")
# ax.plot(θ_cov_ind[θ_ind] , mcmc_θiθj_cov[θ_ind], "s", color="C2",  fillstyle="none" , label="MCMC")


# ax.legend()
# # plot MCMC results 
# ax.set_xlabel("Cov(θ₍ᵢ₎, θ₍ⱼ₎)")
# fig.tight_layout()
# fig.savefig("Darcy-2D-cov.pdf")

# visulize the log permeability field
fig_logk, ax_logk = PyPlot.subplots(ncols = 6, sharex=true, sharey=true, figsize=(30,5))
for ax in ax_logk ;  ax.set_xticks([]) ; ax.set_yticks([]) ; end
color_lim = (minimum(darcy.logκ_2d), maximum(darcy.logκ_2d))

plot_field(darcy, darcy.logκ_2d, color_lim, ax_logk[1]) 
plot_field(darcy, compute_logκ_2d(darcy, mcmc_θ_mean),       color_lim, ax_logk[2]) 
plot_field(darcy, compute_logκ_2d(darcy, uki_θ_mean),        color_lim, ax_logk[3]) 
plot_field(darcy, compute_logκ_2d(darcy, uki_2np1_θ_mean),   color_lim, ax_logk[4])
plot_field(darcy, compute_logκ_2d(darcy, eaki_θ_mean ),      color_lim, ax_logk[5]) 
plot_field(darcy, compute_logκ_2d(darcy, etki_θ_mean ),      color_lim, ax_logk[6]) 


fig_logk.tight_layout()
fig_logk.savefig("Darcy-2D-logk-LR.pdf")






