include("../Plot.jl")
include("../RExKI.jl")
include("Darcy.jl")
include("../ModelError/Misfit2Diagcov.jl")

"""
Nθ=32 variables; Ny=49 data
"""


function ExKI_Run(t_mean, t_cov, θ_bar, θθ_cov,  darcy::Param_Darcy,  α_reg::Float64, N_iter::Int64 = 100)
    parameter_names = ["logκ_2d"]
    
    ens_func(θ_ens) = run_Darcy_ensemble(darcy, θ_ens)
    
    exkiobj = ExKIObj(parameter_names,
    θ_bar, 
    θθ_cov,
    t_mean, # observation
    t_cov,
    α_reg)
    
    update_cov = 1
    for i in 1:N_iter
        
        params_i = deepcopy(exkiobj.θ_bar[end])
        
        @info "L₂ error of params_i :", norm(darcy.u_ref[1:length(params_i)] - params_i), " / ",  norm(darcy.u_ref[1:length(params_i)])
        
        logκ_2d_i = compute_logκ_2d(darcy, params_i)
        
        @info "F error of logκ :", norm(darcy.logκ_2d - logκ_2d_i), " / ",  norm(darcy.logκ_2d )
        
        
        update_ensemble!(exkiobj, ens_func) 
        
        @info "F error of data_mismatch :", (exkiobj.g_bar[end] - exkiobj.g_t)'*(exkiobj.obs_cov\(exkiobj.g_bar[end] - exkiobj.g_t))
        
        if (update_cov > 0) && (i%update_cov == 0) 
            exkiobj.θθ_cov[1] = copy(exkiobj.θθ_cov[end])
        end
        
    end
    
    return exkiobj
    
end

function Darcy_Test(darcy::Param_Darcy, 
    θ0_bar::Array{Float64,1}, θθ0_cov::Array{Float64,2},
    t_mean::Array{Float64,1}, t_cov::Array{Float64, 2}, 
    α_reg::Float64, N_ite::Int64)
    kiobj = ExKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov, darcy,  α_reg, N_ite)
    return kiobj
end


# generate truth (all Fourier modes) and observations
# observation are at frame 0, Δd_t, 2Δd_t, ... nt
# with sparse points at Array(1:Δd_x:nx) × Array(1:Δd_y:ny)
function Prediction_Helper(darcy::Param_Darcy, θ::Array{Float64,1})

    logκ_2d = compute_logκ_2d(darcy, θ)
    κ_2d = exp.(logκ_2d)
    h_2d = solve_GWF(darcy, κ_2d)
    N = darcy.N
    
    return [diag(h_2d) ; (h_2d[:, Int64(N/2)] + h_2d[:, Int64(N/2) + 1])/2 ]
  
  end


# predict the velocity at future time at the central line of the domain
function Prediction(darcy, kiobj, θ_mean, θθ_cov, darcy_ref)

    xx = darcy.xx
    nx = length(xx)
    
    obs_ref = Prediction_Helper(darcy_ref, darcy_ref.u_ref)
  

    N_ens = kiobj.N_ens
    θθ_cov = (θθ_cov+θθ_cov')/2 
    θ_p = construct_sigma_ensemble(kiobj, θ_mean, θθ_cov)
    obs = zeros(Float64, N_ens, 2nx)
    Threads.@threads for i = 1:N_ens
        θ = θ_p[i, :]
        obs[i, :] = Prediction_Helper(darcy, θ)
    end
  
    obs_mean = obs[1, :]
  
    obs_cov  = construct_cov(kiobj,  obs, obs_mean)  #+ Array(Diagonal( obs_noise_level^2 * obs_mean.^2)) 
    obs_std = sqrt.(diag(obs_cov))
  
  
    # optimization related plots
    fig_disp, ax_disp = PyPlot.subplots(ncols = 2, sharex=false, sharey=false, figsize=(16, 6))
    markevery = 10
    # u velocity 
    ax_disp[1].plot(xx, obs_ref[1:nx], "--o", color="grey", fillstyle="none", label="Reference", markevery = markevery)
    ax_disp[1].plot(xx, obs_mean[1:nx], "-*r",  markevery = markevery, label="UKI")
    ax_disp[1].plot(xx, (obs_mean[1:nx] + 3obs_std[1:nx]),  "--r")
    ax_disp[1].plot(xx, (obs_mean[1:nx] - 3obs_std[1:nx]),  "--r")
    # # v velocity
    ax_disp[2].plot(xx, obs_ref[nx+1:2nx], "--o", color="grey", fillstyle="none", label="Reference", markevery = markevery)
    ax_disp[2].plot(xx, obs_mean[nx+1:2nx], "-*r",  markevery = markevery, label="UKI")
    ax_disp[2].plot(xx, (obs_mean[nx+1:2nx] + 3obs_std[nx+1:2nx]),   "--r")
    ax_disp[2].plot(xx, (obs_mean[nx+1:2nx] - 3obs_std[nx+1:2nx]),   "--r")
    # # ω 
    # ax_disp[2,1].plot(xx, obs_ref[2nx+1:3nx], "--o", color="grey", fillstyle="none", label="Reference", markevery = markevery)
    # ax_disp[2,1].plot(xx, obs_mean[2nx+1:3nx], "-*r",  markevery = markevery, label="UKI")
    # ax_disp[2,1].plot(xx, (obs_mean[2nx+1:3nx] + 3obs_std[2nx+1:3nx]),   "--r")
    # ax_disp[2,1].plot(xx, (obs_mean[2nx+1:3nx] - 3obs_std[2nx+1:3nx]),   "--r")
    # # p 
    # ax_disp[2,2].plot(xx, obs_ref[3nx+1:end], "--o", color="grey", fillstyle="none", label="Reference", markevery = markevery)
    # ax_disp[2,2].plot(xx, obs_mean[3nx+1:end], "-*r",  markevery = markevery, label="UKI")
    # ax_disp[2,2].plot(xx, (obs_mean[3nx+1:end] + 3obs_std[3nx+1:end]),   "--r")
    # ax_disp[2,2].plot(xx, (obs_mean[3nx+1:end] - 3obs_std[3nx+1:end]),   "--r")
    
    #ymin,ymax = -5.0, 10.0
    ax_disp[1].set_xlabel("X")
    ax_disp[1].set_ylabel("Pressure")
    #ax_disp.set_ylim([ymin,ymax])
    ax_disp[1].legend()

    ax_disp[2].set_xlabel("X")
    ax_disp[2].set_ylabel("Pressure")
    #ax_disp.set_ylim([ymin,ymax])
    ax_disp[2].legend()
  
    
    fig_disp.tight_layout()
    fig_disp.savefig("Darcy-Pressure.png")
    close(fig_disp)
end


###############################################################################################


rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
mysize = 20
font0 = Dict(
"font.size" => mysize,
"axes.labelsize" => mysize,
"xtick.labelsize" => mysize,
"ytick.labelsize" => mysize,
"legend.fontsize" => mysize,
)
merge!(rcParams, font0)



N, L = 80, 1.0
obs_ΔN = 10
α = 1.0
τ = 3.0
KL_trunc = 32
darcy_ref = Param_Darcy(N, obs_ΔN, L, KL_trunc, α, τ)
κ_2d = exp.(darcy_ref.logκ_2d)
h_2d = solve_GWF(darcy_ref, κ_2d)
plot_field(darcy_ref, h_2d, true, "Figs/Darcy-obs-ref.pdf")
# plot_field(darcy_ref, darcy_ref.logκ_2d, false, "Figs/Darcy-logk-ref.pdf")

myclim = [-1.2, maximum(darcy_ref.logκ_2d)]
plot_field(darcy_ref, darcy_ref.logκ_2d,  myclim,  "Figs/Darcy-logk-ref.pdf")
# observation
noise_level = 0.05
t_mean_noiseless = compute_obs(darcy_ref, h_2d)
t_cov = Array(Diagonal(noise_level^2 * t_mean_noiseless.^2))
Random.seed!(123);
# The observation error is 0.05 y_obs N(0,1), Σ_η = 0.05^2 y_obs × y_obs
t_mean = copy(t_mean_noiseless)
for i = 1:length(t_mean)
    noise = rand(Normal(0, noise_level*t_mean[i]))
    t_mean[i] += noise
end


### Learn with wrong model 
α, τ = 2.0, 5.0
darcy = Param_Darcy(N, obs_ΔN, L, KL_trunc, α, τ)
N_ite = 30
N_θ = 8
θ0_bar = zeros(Float64, N_θ)
θθ0_cov = Array(Diagonal(fill(1.0, N_θ)))
α_reg = 1.0


# first round 
kiobj = Darcy_Test(darcy, θ0_bar, θθ0_cov, t_mean, t_cov, α_reg, N_ite)
# adjust model error , update t_cov
data_misfit = (kiobj.g_bar[end] - t_mean)
n_dm = length(kiobj.g_bar[end] - t_mean)
diag_cov = Misfit2Diagcov(2, data_misfit, t_mean)
t_cov = Array(Diagonal(diag_cov))


# second round 
kiobj = Darcy_Test(darcy, θ0_bar, θθ0_cov, t_mean, t_cov, α_reg, N_ite)
@save "exkiobj.dat" kiobj


# @load "exkiobj.dat" kiobj
# n_dm = length(kiobj.g_bar[end] - t_mean)
  


Ny_Nθ = n_dm/length(kiobj.θ_bar[end])
@info "Ny/Nθ is ", Ny_Nθ
Prediction(darcy, kiobj, kiobj.θ_bar[end], kiobj.θθ_cov[end]*Ny_Nθ, darcy_ref)


logκ_2d = compute_logκ_2d(darcy, kiobj.θ_bar[end])

plot_field(darcy, logκ_2d,  myclim,  "Figs/Darcy-logk-model-error.pdf")

