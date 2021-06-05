"""
ExKIObj{FT<:AbstractFloat, IT<:Int}
Struct that is used in Unscented Kalman Inversion (ExKI)
For solving the inverse problem 
    y = G(θ) + η
    
"""
mutable struct ExKIObj{FT<:AbstractFloat, IT<:Int}
    "vector of parameter names (never used)"
     θ_names::Array{String,1}
    "a vector of arrays of size N_ensemble x N_parameters containing the mean of the parameters (in each exki iteration a new array of mean is added)"
     θ_mean::Vector{Array{FT, 1}}
     "a vector of arrays of size N_ensemble x (N_parameters x N_parameters) containing the covariance of the parameters (in each exki iteration a new array of cov is added)"
     θθ_cov::Vector{Array{FT, 2}}
     "a vector of arrays of size N_ensemble x N_y containing the predicted observation (in each exki iteration a new array of predicted observation is added)"
     y_pred::Vector{Array{FT, 1}}
     "vector of observations (length: N_y)"
     y::Array{FT, 1}
     "covariance of the observational noise"
     Σ_η
     "size of θ"
     N_θ::IT
     "size of y"
     N_y::IT
     "covariance of the artificial evolution error"
     Σ_ω::Array{FT, 2}
     "covariance of the artificial observation error"
     Σ_ν::Array{FT, 2}
     "regularization parameter"
     α_reg::FT
     "regularization vector"
     r::Array{FT, 1}
     "update frequency"
     update_freq::IT
     "current iteration number"
     iter::IT
end



"""
ExKIObj Constructor 
parameter_names::Array{String,1} : parameter name vector
θ0_mean::Array{FT} : prior mean
θθ0_cov::Array{FT, 2} : prior covariance
g_t::Array{FT,1} : observation 
obs_cov::Array{FT, 2} : observation covariance
α_reg::FT : regularization parameter toward θ0 (0 < α_reg <= 1), default should be 1, without regulariazion
"""
function ExKIObj(θ_names::Array{String,1},
                θ0_mean::Array{FT}, 
                θθ0_cov::Array{FT, 2},
                y::Array{FT,1},
                Σ_η,
                α_reg::FT,
                update_freq::IT) where {FT<:AbstractFloat, IT<:Int}

    
    N_θ = size(θ0_mean,1)
    N_y = size(y, 1)

    

    θ_mean = Array{FT,1}[]  # array of Array{FT, 2}'s
    push!(θ_mean, θ0_mean) # insert parameters at end of array (in this case just 1st entry)
    θθ_cov = Array{FT,2}[] # array of Array{FT, 2}'s
    push!(θθ_cov, θθ0_cov) # insert parameters at end of array (in this case just 1st entry)

    y_pred = Array{FT, 1}[]  # array of Array{FT, 2}'s
   

    Σ_ω = (2 - α_reg^2)*θθ0_cov
    Σ_ν = 2*Σ_η

    r = θ0_mean
    
    iter = 0

    ExKIObj{FT,IT}(θ_names, θ_mean, θθ_cov, y_pred, 
                  y,   Σ_η, 
                  N_θ, N_y, 
                  Σ_ω, Σ_ν, α_reg, r, 
                  update_freq, iter)

end




"""
update exki struct
ens_func: The function g = G(θ)
define the function as 
    ens_func(θ_ens) = MyG(phys_params, θ_ens, other_params)
use G(θ_mean) instead of FG(θ)
"""
function update_ensemble!(exki::ExKIObj{FT, IT}, forward::Function) where {FT<:AbstractFloat, IT<:Int}
    
    exki.iter += 1
    # update evolution covariance matrix
    if exki.update_freq > 0 && exki.iter%exki.update_freq == 0
        exki.Σ_ω = (2 - exki.α_reg^2)exki.θθ_cov[end]
    end

    θ_mean  = exki.θ_mean[end]
    θθ_cov = exki.θθ_cov[end]
    y = exki.y

    α_reg = exki.α_reg
    r = exki.r
    Σ_ω = exki.Σ_ω
    Σ_ν = exki.Σ_ν

    N_θ, N_y = exki.N_θ, exki.N_y
    ############# Prediction step:

    θ_p_mean  = α_reg*θ_mean + (1-α_reg)*r
    θθ_p_cov = α_reg^2*θθ_cov + Σ_ω
    

    g_mean, dg = forward(θ_p_mean)
    gg_cov = dg * θθ_p_cov * dg' + Σ_ν
    θg_cov = θθ_p_cov * dg'

    tmp = θg_cov/gg_cov

    θ_mean =  θ_p_mean + tmp*(y - g_mean)

    θθ_cov =  θθ_p_cov - tmp*θg_cov' 


    ########### Save resutls
    push!(exki.y_pred, g_mean) # N_ens x N_data
    push!(exki.θ_mean, θ_mean) # N_ens x N_params
    push!(exki.θθ_cov, θθ_cov) # N_ens x N_data
end



function ExKI_Run(s_param, forward::Function, 
    θ0_mean, θθ0_cov,
    y, Σ_η,
    α_reg,
    update_freq,
    N_iter;)
    
    θ_names = s_param.θ_names
    
    
    exkiobj = ExKIObj(θ_names,
    θ0_mean, 
    θθ0_cov,
    y,
    Σ_η,
    α_reg,
    update_freq)
    

    func(θ) = forward(s_param, θ) 
    
    for i in 1:N_iter
        
        update_ensemble!(exkiobj, func) 

        @info "optimization error at iter $(i) = ", 0.5*(exkiobj.y_pred[i] - exkiobj.y)'*(exkiobj.Σ_η\(exkiobj.y_pred[i] - exkiobj.y))
        @info "Frobenius norm of the covariance at iter $(i) = ", norm(exkiobj.θθ_cov[i])
    end
    
    return exkiobj
    
end


function plot_param_iter(exkiobj::ExKIObj{FT, IT}, θ_ref::Array{FT,1}, θ_ref_names::Array{String}) where {FT<:AbstractFloat, IT<:Int}
    
    θ_mean = exkiobj.θ_mean
    θθ_cov = exkiobj.θθ_cov
    
    N_iter = length(θ_mean) - 1
    ites = Array(LinRange(1, N_iter+1, N_iter+1))
    
    θ_mean_arr = abs.(hcat(θ_mean...))
    
    
    N_θ = length(θ_ref)
    θθ_std_arr = zeros(Float64, (N_θ, N_iter+1))
    for i = 1:N_iter+1
        for j = 1:N_θ
            θθ_std_arr[j, i] = sqrt(θθ_cov[i][j,j])
        end
    end
    
    for i = 1:N_θ
        errorbar(ites, θ_mean_arr[i,:], yerr=3.0*θθ_std_arr[i,:], fmt="--o",fillstyle="none", label=θ_ref_names[i])   
        plot(ites, fill(θ_ref[i], N_iter+1), "--", color="gray")
    end
    
    xlabel("Iterations")
    legend()
    tight_layout()
end


function plot_opt_errors(exkiobj::ExKIObj{FT, IT}, 
    θ_ref::Union{Array{FT,1}, Nothing} = nothing, 
    transform_func::Union{Function, Nothing} = nothing) where {FT<:AbstractFloat, IT<:Int}
    
    θ_mean = exkiobj.θ_mean
    θθ_cov = exkiobj.θθ_cov
    y_pred = exkiobj.y_pred
    Σ_η = exkiobj.Σ_η
    y = exkiobj.y

    N_iter = length(θ_mean) - 1
    
    ites = Array(LinRange(1, N_iter, N_iter))
    N_subfigs = (θ_ref === nothing ? 2 : 3)

    errors = zeros(Float64, N_subfigs, N_iter)
    fig, ax = PyPlot.subplots(ncols=N_subfigs, figsize=(N_subfigs*6,6))
    for i = 1:N_iter
        errors[N_subfigs - 1, i] = 0.5*(y - y_pred[i])'*(Σ_η\(y - y_pred[i]))
        errors[N_subfigs, i]     = norm(θθ_cov[i])
        
        if N_subfigs == 3
            errors[1, i] = norm(θ_ref - (transform_func === nothing ? θ_mean[i] : transform_func(θ_mean[i])))/norm(θ_ref)
        end
        
    end

    markevery = max(div(N_iter, 10), 1)
    ax[N_subfigs - 1].plot(ites, errors[N_subfigs - 1, :], linestyle="--", marker="o", fillstyle="none", markevery=markevery)
    ax[N_subfigs - 1].set_xlabel("Iterations")
    ax[N_subfigs - 1].set_ylabel("Optimization error")
    ax[N_subfigs - 1].grid()
    
    ax[N_subfigs].plot(ites, errors[N_subfigs, :], linestyle="--", marker="o", fillstyle="none", markevery=markevery)
    ax[N_subfigs].set_xlabel("Iterations")
    ax[N_subfigs].set_ylabel("Frobenius norm of the covariance")
    ax[N_subfigs].grid()
    if N_subfigs == 3
        ax[1].set_xlabel("Iterations")
        ax[1].plot(ites, errors[1, :], linestyle="--", marker="o", fillstyle="none", markevery=markevery)
        ax[1].set_ylabel("L₂ norm error")
        ax[1].grid()
    end
    
end