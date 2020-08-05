include("../Plot.jl")
include("../REKI.jl")
include("Darcy.jl")

function EKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov,  darcy, N_ens,  α_reg::Float64, N_iter::Int64 = 100)
    parameter_names = ["logκ_2d"]
    
    ens_func(θ_ens) = run_Darcy_ensemble(darcy, f, θ_ens)
    
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


function Darcy_Test(darcy::Param_Darcy, N_θ::Int64, N_ens::Int64, α_reg::Float64 , N_ite::Int64 = 100, noise_level::Float64=-1.0)
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
    
    ekiobj = EKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov, darcy, N_ens, α_reg,  N_ite)
    
    return ekiobj
end




function Plot_Params(ekiobj::EKIObj, filename::String = "None")
    N_θ = 5 #first 2 components
    Nite = length(ekiobj.θ)
    @info Nite
    N_ens = ekiobj.N_ens
    θ_bar_arr = zeros(Float64, (N_θ, N_ite))
    θθ_cov_arr = zeros(Float64, (N_θ, N_ite))
    for i = 1:N_ite
        θ_bar_arr[:, i] = dropdims(mean(ekiobj.θ[i], dims=1), dims=1) 
        # diag { [θ - θ_bar]*[θ - θ_bar]'/ (J-1) } =  sqrt{ ∑(θ_i - θ_bar).^2 / J-1 }
        for j = 1:N_ens
            θθ_cov_arr[:, i] .+= (ekiobj.θ[i][j, :] - θ_bar_arr[:, i]).^2
        end

        θθ_cov_arr[:, i] = sqrt.(θθ_cov_arr[:, i]/(N_ens - 1))
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

function Plot_Ite_Error(ekiobj::EKIObj, label::String, filename::String = "None")
    Nite = length(ekiobj.θ_bar)
    @info Nite
    θ_bar = zeros(Float64, (N_θ, N_ite))
    for i = 1:N_ite
        θ_bar[:, i] = dropdims(mean(ekiobj.θ[i], dims=1), dims=1) 
    end


    ites = Array(LinRange(1, N_ite, N_ite))
    errors = zeros(Float64, (2, N_ite))
    for i = 1:N_ite
    
        errors[1, i] = norm(darcy.logκ_2d - compute_logκ_2d(darcy, θ_bar[:, i]))/norm(darcy.logκ_2d)
        errors[2, i] = 0.5*(ekiobj.g_bar[i] - ekiobj.g_t)'*(ekiobj.obs_cov\(ekiobj.g_bar[i] - ekiobj.g_t))
    
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

function Plot_Ite_Datamismatch(ekiobj::EKIObj, label::String, filename::String = "None")
    Nite = length(ekiobj.θ_bar)
    @info Nite

    θ_bar = zeros(Float64, (N_θ, N_ite))
    for i = 1:N_ite
        θ_bar[:, i] = dropdims(mean(ekiobj.θ[i], dims=1), dims=1) 
    end

    ites = Array(LinRange(1, N_ite, N_ite))
    errors = zeros(Float64, (2, N_ite))
    for i = 1:N_ite
    
        errors[1, i] = norm(darcy.logκ_2d - compute_logκ_2d(darcy, θ_bar[:, i]))/norm(darcy.logκ_2d)
        errors[2, i] = 0.5*(ekiobj.g_bar[i] - ekiobj.g_t)'*(ekiobj.obs_cov\(ekiobj.g_bar[i] - ekiobj.g_t))
    
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
N_θ = 64
noise_level = 0.01
α_reg = 1.0
N_ens = 20
ekiobj = Darcy_Test(darcy, N_θ, N_ens, α_reg, N_ite, noise_level)



κ_2d = exp.(darcy.logκ_2d)
h_2d = solve_GWF(darcy, κ_2d)
plot_field(darcy, h_2d, true, "Darcy-obs-ref.pdf")
plot_field(darcy, darcy.logκ_2d, false, "Darcy-logk-ref.pdf")
plot_field(darcy, compute_logκ_2d(darcy, ekiobj.θ_bar[N_ite]), false, "Darcy-logk-32.pdf")


####################################################################################
Plot_Params(ekiobj, "darcy_error_32.pdf")
Plot_Ite_Error(ekiobj, "Nθ=32", "Darcy-Params.pdf")
Plot_Ite_Datamismatch(ekiobj, "Nθ=32", "Darcy-Data-Mismatch.pdf")
@info "finished"



