include("../Plot.jl")
include("../RUKI.jl")
include("Darcy.jl")

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
        
        @info "L₂ error of params_i :", norm(darcy.u_ref[1:length(params_i)] - params_i), " / ",  norm(darcy.u_ref[1:length(params_i)])
        
        logκ_2d_i = compute_logκ_2d(darcy, params_i)
        
        @info "F error of logκ :", norm(darcy.logκ_2d - logκ_2d_i), " / ",  norm(darcy.logκ_2d )
        
        
        update_ensemble!(ukiobj, ens_func) 
        
        @info "F error of data_mismatch :", (ukiobj.g_bar[end] - ukiobj.g_t)'*(ukiobj.obs_cov\(ukiobj.g_bar[end] - ukiobj.g_t))
        
        
    end
    
    return ukiobj
    
end

function Darcy_Test(darcy::Param_Darcy, N_θ::Int64, α_reg::Float64, N_ite::Int64 = 100, noise_level::Float64=-1.0)
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
    
    ukiobj = UKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov, darcy,  α_reg, N_ite)
    
    return ukiobj
end







function Plot_Params(ukiobj::UKIObj, filename::String = "None")
    N_θ = 5 #first 2 components
    Nite = length(ukiobj.θ_bar)
    @info Nite
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
    errorbar(ites, θ_bar_arr[1,:], yerr=3.0*θθ_cov_arr[1,:], errorevery = 20, markevery=20, fmt="--o",fillstyle="none", label=L"\theta_{(0)}")

    errorbar(ites.+2, θ_bar_arr[2,:], yerr=3.0*θθ_cov_arr[2,:], errorevery = 20,markevery=20, fmt="--o",fillstyle="none", label=L"\theta_{(1)}")
 
    errorbar(ites.-2, θ_bar_arr[3,:], yerr=3.0*θθ_cov_arr[3,:], errorevery = 20,markevery=20, fmt="--o",fillstyle="none", label=L"\theta_{(2)}")

    errorbar(ites.+4, θ_bar_arr[4,:], yerr=3.0*θθ_cov_arr[4,:], errorevery = 20,markevery=20, fmt="--o",fillstyle="none", label=L"\theta_{(3)}")

    errorbar(ites.-4, θ_bar_arr[5,:], yerr=3.0*θθ_cov_arr[5,:], errorevery = 20,markevery=20, fmt="--o",fillstyle="none", label=L"\theta_{(4)}")
    
    
    ites = Array(LinRange(1, N_ite+10, N_ite+10))
    for i = 1:N_θ
        plot(ites, fill(darcy.u_ref[i], N_ite+10), "--", color="gray")
    end
    
    xlabel("Iterations")
    legend(loc=3)
    grid("on")
    tight_layout()
    if filename != "None"
        savefig(filename)
        close("all")
    end
    
end

function Plot_Ite_Error(ukiobj::UKIObj, label::String, filename::String = "None")
    Nite = length(ukiobj.θ_bar)
    @info Nite


    ites = Array(LinRange(1, N_ite, N_ite))
    errors = zeros(Float64, (2, N_ite))
    for i = 1:N_ite
    
        errors[1, i] = norm(darcy.logκ_2d - compute_logκ_2d(darcy, ukiobj.θ_bar[i]))/norm(darcy.logκ_2d)
        errors[2, i] = 0.5*(ukiobj.g_bar[i] - ukiobj.g_t)'*(ukiobj.obs_cov\(ukiobj.g_bar[i] - ukiobj.g_t))
    
    end
    
    plot(ites, errors[1, :], "--o", markevery=20, fillstyle="none", label= label)
    xlabel("Iterations")
    ylabel("Relative Frobenius norm error")
    grid("on")
    legend()
    tight_layout()
    if filename !="None"
        savefig(filename)
        close("all")
    end
end

function Plot_Ite_Datamismatch(ukiobj::UKIObj, label::String, filename::String = "None")
    Nite = length(ukiobj.θ_bar)
    @info Nite


    ites = Array(LinRange(1, N_ite, N_ite))
    errors = zeros(Float64, (2, N_ite))
    for i = 1:N_ite
    
        errors[1, i] = norm(darcy.logκ_2d - compute_logκ_2d(darcy, ukiobj.θ_bar[i]))/norm(darcy.logκ_2d)
        errors[2, i] = 0.5*(ukiobj.g_bar[i] - ukiobj.g_t)'*(ukiobj.obs_cov\(ukiobj.g_bar[i] - ukiobj.g_t))
    
    end
    
    plot(ites, errors[2, :], "--o", markevery=20, fillstyle="none", label= label)
    xlabel("Iterations")
    ylabel("Optimization error")
    grid("on")
    legend()
    tight_layout()
    if filename !="None"
        savefig(filename)
        close("all")
    end

end



@info "start"
N, L = 80, 1.0
obs_ΔN = 10
α = 2.0
τ = 3.0
KL_trunc = 256
darcy = Param_Darcy(N, obs_ΔN, L, KL_trunc, α, τ)


N_ite = 200
N_θ = 32
noise_level = 0.01
α_reg = 1.0
ukiobj = Darcy_Test(darcy, N_θ, α_reg, N_ite, noise_level)



κ_2d = exp.(darcy.logκ_2d)
h_2d = solve_GWF(darcy, κ_2d)
plot_field(darcy, h_2d, true, "Darcy-obs-ref.pdf")
plot_field(darcy, darcy.logκ_2d, false, "Darcy-logk-ref.pdf")
plot_field(darcy, compute_logκ_2d(darcy, ukiobj.θ_bar[N_ite]), false, "Darcy-logk-32.pdf")


####################################################################################
Plot_Params(ukiobj, "darcy_error_32.pdf")
Plot_Ite_Error(ukiobj, "Nθ=32", "Darcy-Params.pdf")
Plot_Ite_Datamismatch(ukiobj, "Nθ=32", "Darcy-Data-Mismatch.pdf")
@info "finished"
