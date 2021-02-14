include("../Plot.jl")
include("../RExKI.jl")
include("Darcy.jl")

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



function Compare_32()
    N_θ = 32
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
    KL_trunc = N_θ
    darcy = Param_Darcy(N, obs_ΔN, L, KL_trunc, α, τ)
    N_ite = 30
    
    κ_2d = exp.(darcy.logκ_2d)
    h_2d = solve_GWF(darcy, κ_2d)
    plot_field(darcy, h_2d, true, "Figs/Darcy-obs-ref.pdf")
    plot_field(darcy, darcy.logκ_2d, false, "Figs/Darcy-logk-ref.pdf")
    
    
    N_sample = 6
    
    fig_ite, ax_ite = PyPlot.subplots(ncols = 3, nrows=1, sharex=false, sharey=false, figsize=(18,6))
    fig_θ, ax_θ = PyPlot.subplots(ncols = 1, nrows=N_sample, sharex=true, sharey=true, figsize=(16,16))
    
    fig_logk, ax_logk = PyPlot.subplots(ncols = 3, nrows=2, sharex=true, sharey=true, figsize=(18,12))
    for ax in ax_logk ;  ax.set_xticks([]) ; ax.set_yticks([]) ; end
    clim = (minimum(darcy.logκ_2d), maximum(darcy.logκ_2d))


    θ0_bar = zeros(Float64, N_θ)
    θθ0_cov = Array(Diagonal(fill(1.0, N_θ)))
    α_reg = 1.0
    noise_level = 0.05
    
    # observation
    t_mean_noiseless = compute_obs(darcy, h_2d)
    t_cov = Array(Diagonal(noise_level^2 * t_mean_noiseless.^2))
    Random.seed!(123);
    
    
    
    
    
    # The observation error is 0.05 y_obs N(0,1), Σ_η = 0.05^2 y_obs × y_obs
    
    ites = Array(LinRange(1, N_ite, N_ite))
    errors = zeros(Float64, (3, N_ite))
    
    for n = 1:N_sample
        t_mean = copy(t_mean_noiseless)
        
        for i = 1:length(t_mean)
            noise = rand(Normal(0, noise_level*t_mean[i]))
            t_mean[i] += noise
        end
        
        kiobj = Darcy_Test(darcy, 
        θ0_bar, θθ0_cov,
        t_mean, t_cov, 
        α_reg, N_ite)
        
        θ_bar = kiobj.θ_bar
        
        for i = 1:N_ite
            
            errors[1, i] = norm(darcy.logκ_2d - compute_logκ_2d(darcy, θ_bar[i]))/norm(darcy.logκ_2d)
            errors[2, i] = 0.5*(kiobj.g_bar[i] - kiobj.g_t)'*(kiobj.obs_cov\(kiobj.g_bar[i] - kiobj.g_t))
            errors[3, i] = norm(kiobj.θθ_cov[i]) 
        end
        errors[3, 1] = norm(θθ0_cov) 
        
        
        ax_ite[1].plot(ites, errors[1, :], "-o", fillstyle="none", markevery=2 )
        ax_ite[2].plot(ites, errors[2, :], "-o", fillstyle="none", markevery=2 )
        ax_ite[3].plot(ites, errors[3, :], "-o", fillstyle="none", markevery=2 )
        
        θ_ind = Array(1:N_θ)
        ax_θ[n].plot(θ_ind , darcy.u_ref[1:N_θ], "--o", color="grey", fillstyle="none", label="Reference")

        ki_θ_bar  = kiobj.θ_bar[end]
        ki_θθ_cov = kiobj.θθ_cov[end]
        ki_θθ_std = sqrt.(diag(kiobj.θθ_cov[end]))
        ax_θ[n].plot(θ_ind , ki_θ_bar,"-*", color="red", fillstyle="none")
        ax_θ[n].plot(θ_ind , ki_θ_bar + 3.0*ki_θθ_std, color="red")
        ax_θ[n].plot(θ_ind , ki_θ_bar - 3.0*ki_θθ_std, color="red")
        ax_θ[n].set_ylabel("θ")


        im = plot_field(darcy, compute_logκ_2d(darcy, ki_θ_bar), clim, ax_logk[:][n])
        
    end
    
    
    ax_ite[1].set_xlabel("Iterations")
    ax_ite[1].set_ylabel("Relative L₂ norm error")
    ax_ite[1].grid(true)
    ax_ite[2].set_xlabel("Iterations")
    ax_ite[2].set_ylabel("Optimization error")
    ax_ite[2].grid(true)
    ax_ite[3].set_xlabel("Iterations")
    ax_ite[3].set_ylabel("Frobenius norm")
    ax_ite[3].grid(true)

    fig_ite.tight_layout()
    fig_ite.savefig("Figs/Darcy-error.png")
    
    ax_θ[N_sample].set_xlabel("θ indices")
    fig_θ.tight_layout()
    fig_θ.savefig("Figs/Darcy-theta.png")


    fig_logk.tight_layout()
    # cbar_ax = fig_logk.add_axes([0.90, 0.05, 0.02, 0.6])
    # fig_logk.colorbar(im, cbar_ax)
    
    fig_logk.savefig("Figs/Darcy-logk.pdf")
    close(fig_logk)
    
    
end


Compare_32()
