include("../Plot.jl")
include("../RExKI.jl")
include("../REKI.jl")
include("Darcy.jl")

function EKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov,  darcy, N_ens,  α_reg::Float64, N_iter::Int64 = 100)
    parameter_names = ["logκ_2d"]
    
    ens_func(θ_ens) = run_Darcy_ensemble(darcy, θ_ens)
    
    initial_params = Array(rand(MvNormal(θ0_bar, θθ0_cov), N_ens)')
    
    ekiobj = EKIObj(parameter_names,
    initial_params, 
    θ0_bar,
    θθ0_cov,
    t_mean, # observation
    t_cov,
    α_reg)
    
    for i = 1:N_iter
        params_i = dropdims(mean(ekiobj.θ[end], dims=1), dims=1) 
        
        logκ_2d_i = compute_logκ_2d(darcy, params_i)
        
        @info "F error of logκ :", norm(darcy.logκ_2d - logκ_2d_i), " / ",  norm(darcy.logκ_2d )
        
        
        update_ensemble!(ekiobj, ens_func) 
        
        @info "F error of data_mismatch :", (ekiobj.g_bar[end] - ekiobj.g_t)'*(ekiobj.obs_cov\(ekiobj.g_bar[end] - ekiobj.g_t))
        
    end
    
    return ekiobj
    
end


function UKI_Run(t_mean, t_cov, θ_bar, θθ_cov,  darcy::Param_Darcy,  α_reg::Float64, N_iter::Int64 = 100)
    parameter_names = ["logκ_2d"]
    
    ens_func(θ_ens) = run_Darcy_ensemble(darcy, θ_ens)
    
    ukiobj = UKIObj(parameter_names,
    θ_bar, 
    θθ_cov,
    t_mean, # observation
    t_cov,
    α_reg)
    
    
    for i in 1:N_iter
        
        params_i = deepcopy(ukiobj.θ_bar[end])
        
        #@info "L₂ error of params_i :", norm(darcy.u_ref[1:length(params_i)] - params_i), " / ",  norm(darcy.u_ref[1:length(params_i)])
        
        logκ_2d_i = compute_logκ_2d(darcy, params_i)
        
        @info "F error of logκ :", norm(darcy.logκ_2d - logκ_2d_i), " / ",  norm(darcy.logκ_2d )
        
        
        update_ensemble!(ukiobj, ens_func) 
        
        @info "F error of data_mismatch :", (ukiobj.g_bar[end] - ukiobj.g_t)'*(ukiobj.obs_cov\(ukiobj.g_bar[end] - ukiobj.g_t))
        
        γ = 2.0
        @info "Part1 :", (ukiobj.g_bar[end] - ukiobj.g_t)'*(ukiobj.obs_cov\(ukiobj.g_bar[end] - ukiobj.g_t))/γ
        @info "Part2 :", (1 - α_reg)*(ukiobj.θ_bar[end] - ukiobj.θ_bar[1])'*((α_reg^2*ukiobj.θθ_cov[end] + ukiobj.θθ_cov[1]/(γ-1.0))\(ukiobj.θ_bar[end] - ukiobj.θ_bar[1]))
        
        
    end
    
    return ukiobj
    
end


function ExKI_Run(t_mean, t_cov, θ_bar, θθ_cov,  darcy::Param_Darcy,  α_reg::Float64, N_iter::Int64 = 100)
    parameter_names = ["logκ_2d"]
    
    ens_func(θ_ens) = run_Darcy_ensemble(darcy, θ_ens)
    
    exkiobj = ExKIObj(parameter_names,
    θ_bar, 
    θθ_cov,
    t_mean, # observation
    t_cov,
    α_reg)
    
    
    for i in 1:N_iter
        
        params_i = deepcopy(exkiobj.θ_bar[end])
        
        @info "L₂ error of params_i :", norm(darcy.u_ref[1:length(params_i)] - params_i), " / ",  norm(darcy.u_ref[1:length(params_i)])
        
        logκ_2d_i = compute_logκ_2d(darcy, params_i)
        
        @info "F error of logκ :", norm(darcy.logκ_2d - logκ_2d_i), " / ",  norm(darcy.logκ_2d )
        
        
        update_ensemble!(exkiobj, ens_func) 
        
        @info "F error of data_mismatch :", (exkiobj.g_bar[end] - exkiobj.g_t)'*(exkiobj.obs_cov\(exkiobj.g_bar[end] - exkiobj.g_t))
        
        
    end
    
    return exkiobj
    
end

function Darcy_Test(method::String, darcy::Param_Darcy, N_θ::Int64, α_reg::Float64, N_ite::Int64, N_ens::Int64, noise_level::Float64=-1.0, 
    ax1 = nothing, ax2 = nothing)
    
    @assert(N_θ <= darcy.trunc_KL)
    
    κ_2d = exp.(darcy.logκ_2d)
    h_2d = solve_GWF(darcy, κ_2d)
    
    t_mean = compute_obs(darcy, h_2d)
    if noise_level >= 0.0
        Random.seed!(123);
        
        for i = 1:length(t_mean)
            noise = rand(Normal(0, noise_level*t_mean[i]))
            t_mean[i] += noise
        end
        
    end
    
    
    t_cov = Array(Diagonal(fill(1.0, length(t_mean))))
    
    θ0_bar = zeros(Float64, N_θ)  # mean 
    
    θθ0_cov = Array(Diagonal(fill(1.0, N_θ)))
    if method == "ExKI"
        label = "UKI"
        kiobj = ExKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov, darcy,  α_reg, N_ite)
        θ_bar = kiobj.θ_bar
        linestyle, marker = "-", "o"
    elseif method =="EnKI"
        label = "EKI"
        kiobj = EKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov, darcy,  N_ens, α_reg, N_ite)
        θ_bar = [dropdims(mean(kiobj.θ[i], dims=1), dims=1) for i = 1:N_ite ]  
        linestyle, marker = "--", "s" 
    else
        error("method: ", method, "has not implemented")
    end
    
    
    label = label*" (α = "*string(α_reg)*")"
    
    ites = Array(LinRange(1, N_ite, N_ite))
    errors = zeros(Float64, (2, N_ite))
    for i = 1:N_ite
        
        errors[1, i] = norm(darcy.logκ_2d - compute_logκ_2d(darcy, θ_bar[i]))/norm(darcy.logκ_2d)
        errors[2, i] = 0.5*(kiobj.g_bar[i] - kiobj.g_t)'*(kiobj.obs_cov\(kiobj.g_bar[i] - kiobj.g_t))
        
    end
    
    
    if (!isnothing(ax1)  &&  !isnothing(ax2))
        ax1.semilogy(ites, errors[1, :], linestyle=linestyle, marker=marker, fillstyle="none", markevery=10, label= label)
        ax1.set_ylabel("Relative L₂ norm error")
        ax1.grid(true)
        
        
        ax2.semilogy(ites, errors[2, :], linestyle=linestyle, marker = marker,fillstyle="none", markevery=10, label= label)
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("Optimization error")
        ax2.grid(true)
        ax2.legend()
    end
    
    return kiobj, θ_bar[end]
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
    α = 2.0
    τ = 3.0
    KL_trunc = 256
    darcy = Param_Darcy(N, obs_ΔN, L, KL_trunc, α, τ)
    N_ite = 50
    
    κ_2d = exp.(darcy.logκ_2d)
    h_2d = solve_GWF(darcy, κ_2d)
    plot_field(darcy, h_2d, true, "Figs/Darcy-obs-ref.pdf")
    plot_field(darcy, darcy.logκ_2d, false, "Figs/Darcy-logk-ref.pdf")
    
    
    
    
    # field plot
    fig_logk, ax_logk = PyPlot.subplots(ncols = 3, nrows=3, sharex=true, sharey=true, figsize=(12,12))
    for ax in ax_logk ;  ax.set_xticks([]) ; ax.set_yticks([]) ; end
    clim = (minimum(darcy.logκ_2d), maximum(darcy.logκ_2d))

    for noise_level_per in [0, 1, 5]
        noise_level = noise_level_per/100.0
        N_ens = 100
        
        # loss plot
        fig, (ax1, ax2) = PyPlot.subplots(nrows=2, sharex=true, figsize=(8,12))
        α_reg = 1.0
        
        
        _, θ_bar = Darcy_Test("ExKI", darcy, N_θ, α_reg, N_ite, N_ens, noise_level, ax1, ax2)
        if noise_level_per == 0
            plot_field(darcy, compute_logκ_2d(darcy, θ_bar), clim, ax_logk[1])
        end

        _, θ_bar = Darcy_Test("EnKI", darcy, N_θ, α_reg, N_ite, N_ens, noise_level, ax1, ax2)
        if noise_level_per == 0
            plot_field(darcy, compute_logκ_2d(darcy, θ_bar), clim, ax_logk[4])
        end


        α_reg = 0.5

        _, θ_bar = Darcy_Test("ExKI", darcy, N_θ, α_reg, N_ite, N_ens, noise_level, ax1, ax2)
        if noise_level_per != 0
            ax_id = (noise_level_per == 1 ? 2 : 3 ;)
            plot_field(darcy, compute_logκ_2d(darcy, θ_bar), clim, ax_logk[ax_id])
        end

        _, θ_bar = Darcy_Test("EnKI", darcy, N_θ, α_reg, N_ite, N_ens, noise_level, ax1, ax2)
        if noise_level_per != 0
            ax_id = (noise_level_per == 1 ? 5 : 6 ;)
            plot_field(darcy, compute_logκ_2d(darcy, θ_bar), clim, ax_logk[ax_id])
        end
        
        


        fig.tight_layout()
        fig.savefig("Figs/Darcy-"*string(N_θ)*"-noise-"*string(noise_level_per)*".pdf")
        close(fig)
    end

    plot_field(darcy, darcy.logκ_2d, clim, ax_logk[7])
    im = plot_field(darcy, darcy.logκ_2d, clim, ax_logk[8])
    plot_field(darcy, darcy.logκ_2d, clim, ax_logk[9])
    

    fig_logk.tight_layout()
    cbar_ax = fig_logk.add_axes([0.90, 0.05, 0.02, 0.6])
    fig_logk.colorbar(im, cbar_ax)
    
    fig_logk.savefig("Figs/Darcy-"*string(N_θ)*".pdf")
    close(fig_logk)
    
end


function Compare_8()
    N_θ = 8
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
    α = 2.0
    τ = 3.0
    KL_trunc = 256
    darcy = Param_Darcy(N, obs_ΔN, L, KL_trunc, α, τ)
    N_ite = 50
    
    κ_2d = exp.(darcy.logκ_2d)
    h_2d = solve_GWF(darcy, κ_2d)
    plot_field(darcy, h_2d, true, "Figs/Darcy-obs-ref.pdf")
    plot_field(darcy, darcy.logκ_2d, false, "Figs/Darcy-logk-ref.pdf")
    
    
    
    
    
    fig_logk, ax_logk = PyPlot.subplots(ncols = 4, nrows=1, sharex=true, sharey=true, figsize=(16,4))
    for ax in ax_logk ;  ax.set_xticks([]) ; ax.set_yticks([]) ; end
    clim = (minimum(darcy.logκ_2d), maximum(darcy.logκ_2d))

    for noise_level_per in [0, 1, 5]
        noise_level = noise_level_per/100.0
        N_ens = 100
        

        fig, (ax1, ax2) = PyPlot.subplots(nrows=2, sharex=true, figsize=(8,12))
        α_reg = 1.0
        
        _, θ_bar = Darcy_Test("ExKI", darcy, N_θ, α_reg, N_ite, N_ens, noise_level, ax1, ax2)

        _, _ = Darcy_Test("EnKI", darcy, N_θ, α_reg, N_ite, N_ens, noise_level, ax1, ax2)
       

        if noise_level_per == 0
            ax_id = 1
        elseif noise_level_per == 1
            ax_id = 2
        elseif noise_level_per == 5
            ax_id = 3
        end
        plot_field(darcy, compute_logκ_2d(darcy, θ_bar), clim, ax_logk[ax_id])
  


        α_reg = 0.5
        
        _, θ_bar = Darcy_Test("ExKI", darcy, N_θ, α_reg, N_ite, N_ens, noise_level, ax1, ax2)
      
        _, _ = Darcy_Test("EnKI", darcy, N_θ, α_reg, N_ite, N_ens, noise_level, ax1, ax2)
        


        fig.tight_layout()
        fig.savefig("Figs/Darcy-"*string(N_θ)*"-noise-"*string(noise_level_per)*".pdf")
        close(fig)
    end

    im = plot_field(darcy, darcy.logκ_2d, clim, ax_logk[4])
    

    fig_logk.tight_layout()
    cbar_ax = fig_logk.add_axes([0.90, 0.15, 0.02, 0.7])
    fig_logk.colorbar(im, cbar_ax)
    
    fig_logk.savefig("Figs/Darcy-"*string(N_θ)*".pdf")
    close(fig_logk)
    
end


Compare_32()
Compare_8()