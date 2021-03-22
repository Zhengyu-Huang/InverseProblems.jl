using Random
using Distributions
using LinearAlgebra
include("../Plot.jl")
include("../RExKI.jl")
include("Misfit2Diagcov.jl")


# a0 + a1 x + a2 x^2 .....
# b0 + b1 cos(πx) + b2 cos(2πx) .....
function polysin(x::Float64, a::Array{Float64,1} = Float64[], b::Array{Float64,1} = Float64[])
    val = 0.0
    for i = 1:length(a)
        val += a[i]*x^(i-1)
    end
    
    for i = 1:length(b)
        val += b[i]*cos(i * π * x)
    end
    
    return val
end

a_ref = [4.0; 3.0; 2.0; 1.0]
b_ref = [1.0; 0.0; 0.0; 0.0]
poly_ref = (x)->polysin(x, a_ref, b_ref)


poly_model = (x, θ)->polysin(x, θ)
forward(θ, xx_train) = [poly_model(xx_train[i], θ) for i = 1:length(xx_train)]



function ensemble(params_i::Array{Float64, 2}, xx)
    n_data = length(xx)
    N_ens,  N_θ = size(params_i)
    g_ens = zeros(Float64, N_ens,  n_data)
    
    for i = 1:N_ens 
        # g: N_ens x N_data
        g_ens[i, :] .= forward(params_i[i, :], xx)
    end
    
    return g_ens
end



function ExKI(xx::Array{Float64,1}, t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
    θ0_bar::Array{Float64,1}, θθ0_cov::Array{Float64,2}, 
    α_reg::Float64, N_iter, update_cov::Int64)
    
    @info "start ExKI"
    parameter_names = ["u₁", "u₂"]
    
    ens_func(θ_ens) = ensemble(θ_ens, xx)
    
    ukiobj = ExKIObj(parameter_names,
    θ0_bar, 
    θθ0_cov,
    t_mean, # observation
    t_cov,
    α_reg)
    
    for i in 1:N_iter
        
        update_ensemble!(ukiobj, ens_func) 
        
        @info "iter : ", i,  " norm(θ): ", norm(ukiobj.θ_bar[end]),  " norm(θθ): ", norm(ukiobj.θθ_cov[end])
        
        if (update_cov) > 0 && (i%update_cov == 0) 
            ukiobj.θθ_cov[1] = copy(ukiobj.θθ_cov[end])
        end
        
    end
    
    return ukiobj
end






function prediction(xx_test, xx_train, kiobj, θ_mean, θθ_cov, savefile=nothing)
    
    yy_test_ref = [poly_ref(xx_test[i]) for i = 1:length(xx_test)]
    yy_train_ref = kiobj.g_t 
    
    
    θθ_cov = (θθ_cov+θθ_cov')/2 
    θ_p = construct_sigma_ensemble(kiobj, θ_mean, θθ_cov)
    N_ens = kiobj.N_ens
    
    n_data = length(xx_test)
    
    obs = zeros(Float64, N_ens, n_data)
    
    for i = 1:N_ens
        θ = θ_p[i, :]
        obs[i, :] = forward(θ, xx_test)
    end
    
    obs_mean = obs[1, :]
    obs_cov  = construct_cov(kiobj,  obs, obs_mean)
    obs_std = sqrt.(diag(obs_cov))
    
    # optimization related plots
    fig_disp, ax_disp = PyPlot.subplots(ncols = 1, nrows=1, sharex=false, sharey=false, figsize=(6,4))
    
    ax_disp.plot(xx_train, yy_train_ref, "o", fillstyle="none", color="grey", label="Training (Ny = $(length(xx_train)))")
    
    ax_disp.plot(xx_test, yy_test_ref, "--", color="black", fillstyle="none", label = "Reference", markevery=10)
    
    ax_disp.plot(xx_test, obs_mean, "-*r", markevery=20, label="UKI")
    ax_disp.plot(xx_test, obs_mean + 3obs_std, "--r")
    ax_disp.plot(xx_test, obs_mean - 3obs_std, "--r")
    
    
    ax_disp.set_xlabel("X")
    ax_disp.set_ylabel("Y")
    ax_disp.legend()
    
    fig_disp.tight_layout()
    
    if savefile !== nothing
        fig_disp.savefig(savefile)
    end
end


# Consider the domain [-2, 2]
function PolyModelTest()
    for Ny in (10, 100)
        for θ0_bar in ([0.0;0.0;0.0], [0.0;0.0;0.0;0.0])
            Nθ = length(θ0_bar)
            savefile = "PolyModel_Ntheta$(Nθ)_Ny$(Ny).pdf"
            # Ny = 10
            xx_train = Array(LinRange(-2, 2, Ny))
            yy_train_ref = [poly_ref(xx_train[i]) for i = 1:length(xx_train)]
            
            t_mean = copy(yy_train_ref)

            Random.seed!(123); 
            noise_level = 0.05
            for i = 1:length(t_mean)
                noise = rand(Normal(0, 0.1)) 
                t_mean[i] += noise
            end

            t_cov = Array(Diagonal(fill(0.01, length(t_mean))))
            # θ0_bar = [0.0;0.0;0.0]
            
            θθ0_cov = Array(Diagonal(fill(1.0, length(θ0_bar))))
            α_reg = 1.0
            N_iter = 20
            update_cov = 1
            
            ukiobj = ExKI(xx_train, t_mean, t_cov, θ0_bar, θθ0_cov, α_reg, N_iter, update_cov)
            
            @info "mean: ", ukiobj.θ_bar[end], " cov : ", ukiobj.θθ_cov[end]
            
            
            data_misfit = ukiobj.g_bar[end] - yy_train_ref

            # new_cov =  sum(data_misfit.^2)/length(data_misfit)    # maximum(data_misfit.^2)
            # t_cov = Array(Diagonal(fill(new_cov, length(data_misfit))))

            diag_cov = Misfit2Diagcov(2, data_misfit, yy_train_ref)
            t_cov = Array(Diagonal(diag_cov))
            # t_cov = length(data_misfit)*Array(Diagonal(data_misfit.^2))
            
            ukiobj = ExKI(xx_train, t_mean, t_cov, θ0_bar, θθ0_cov, α_reg, N_iter, update_cov)
            θ = ukiobj.θ_bar[end]
            θθ_cov = ukiobj.θθ_cov[end] *Ny/Nθ
            
            xx_test = Array(LinRange(-2, 2, 100))
            prediction(xx_test, xx_train, ukiobj, θ, θθ_cov, savefile)
            
        end
    end
end

PolyModelTest()