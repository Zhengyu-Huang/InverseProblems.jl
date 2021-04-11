"""
UKIObj{FT<:AbstractFloat, IT<:Int}
Struct that is used in Unscented Kalman Inversion (UKI)
For solving the inverse problem 
    y = G(θ) + η
    
"""
mutable struct UKIObj{FT<:AbstractFloat, IT<:Int}
    "vector of parameter names (never used)"
     θ_names::Array{String,1}
    "a vector of arrays of size N_ensemble x N_parameters containing the mean of the parameters (in each uki iteration a new array of mean is added)"
     θ_mean::Vector{Array{FT, 1}}
     "a vector of arrays of size N_ensemble x (N_parameters x N_parameters) containing the covariance of the parameters (in each uki iteration a new array of cov is added)"
     θθ_cov::Vector{Array{FT, 2}}
     "a vector of arrays of size N_ensemble x N_y containing the predicted observation (in each uki iteration a new array of predicted observation is added)"
     y_pred::Vector{Array{FT, 1}}
     "vector of observations (length: N_y)"
     y::Array{FT, 1}
     "covariance of the observational noise"
     Σ_η
     "number ensemble size (2N_θ - 1)"
     N_ens::IT
     "size of θ"
     N_θ::IT
     "size of y"
     N_y::IT
     "weights in UKI"
     c_weights::Array{FT, 1}
     mean_weights::Array{FT, 1}
     cov_weights::Array{FT, 1}
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
UKIObj Constructor 
parameter_names::Array{String,1} : parameter name vector
θ0_mean::Array{FT} : prior mean
θθ0_cov::Array{FT, 2} : prior covariance
g_t::Array{FT,1} : observation 
obs_cov::Array{FT, 2} : observation covariance
α_reg::FT : regularization parameter toward θ0 (0 < α_reg <= 1), default should be 1, without regulariazion
"""
function UKIObj(θ_names::Array{String,1},
                θ0_mean::Array{FT}, 
                θθ0_cov::Array{FT, 2},
                y::Array{FT,1},
                Σ_η,
                α_reg::FT,
                update_freq::IT;
                modified_uscented_transform::Bool = true) where {FT<:AbstractFloat, IT<:Int}

    
    N_θ = size(θ0_mean,1)
    N_y = size(y, 1)
    # ensemble size
    N_ens = 2*N_θ+1

 
    c_weights = zeros(FT, N_θ)
    mean_weights = zeros(FT, N_ens)
    cov_weights = zeros(FT, N_ens)

    # todo parameters λ, α, β

    κ = 0.0
    β = 2.0
    α = min(sqrt(4/(N_θ + κ)), 1.0)
    λ = α^2*(N_θ + κ) - N_θ


    c_weights[1:N_θ]     .=  sqrt(N_θ + λ)
    mean_weights[1] = λ/(N_θ + λ)
    mean_weights[2:N_ens] .= 1/(2(N_θ + λ))
    cov_weights[1] = λ/(N_θ + λ) + 1 - α^2 + β
    cov_weights[2:N_ens] .= 1/(2(N_θ + λ))

    if modified_uscented_transform
        mean_weights[1] = 1.0
        mean_weights[2:N_ens] .= 0.0
    end

    

    θ_mean = Array{FT,1}[]  # array of Array{FT, 2}'s
    push!(θ_mean, θ0_mean) # insert parameters at end of array (in this case just 1st entry)
    θθ_cov = Array{FT,2}[] # array of Array{FT, 2}'s
    push!(θθ_cov, θθ0_cov) # insert parameters at end of array (in this case just 1st entry)

    y_pred = Array{FT, 1}[]  # array of Array{FT, 2}'s
   

    Σ_ω = (2 - α_reg^2)*θθ0_cov
    Σ_ν = 2*Σ_η

    r = θ0_mean
    
    iter = 0

    UKIObj{FT,IT}(θ_names, θ_mean, θθ_cov, y_pred, 
                  y,   Σ_η, 
                  N_ens, N_θ, N_y, 
                  c_weights, mean_weights, cov_weights, 
                  Σ_ω, Σ_ν, α_reg, r, 
                  update_freq, iter)

end


"""
construct_sigma_ensemble
Construct the sigma ensemble, based on the mean x_mean, and covariance x_cov
"""
function construct_sigma_ensemble(uki::UKIObj{FT, IT}, x_mean::Array{FT}, x_cov::Array{FT,2}) where {FT<:AbstractFloat, IT<:Int}
    N_ens = uki.N_ens
    N_x = size(x_mean,1)
    @assert(N_ens == 2*N_x+1)

    c_weights = uki.c_weights


    chol_xx_cov = cholesky(Hermitian(x_cov)).L

    x = zeros(FT, 2*N_x+1, N_x)
    x[1, :] = x_mean
    for i = 1: N_x
        x[i+1,     :] = x_mean + c_weights[i]*chol_xx_cov[:,i]
        x[i+1+N_x, :] = x_mean - c_weights[i]*chol_xx_cov[:,i]
    end

    return x
end


"""
construct_mean x_mean from ensemble x
"""
function construct_mean(uki::UKIObj{FT, IT}, x::Array{FT,2}) where {FT<:AbstractFloat, IT<:Int}
    N_ens, N_x = size(x)

    @assert(uki.N_ens == N_ens)

    x_mean = zeros(FT, N_x)

    mean_weights = uki.mean_weights

    
    for i = 1: N_ens
        x_mean += mean_weights[i]*x[i,:]
    end

    return x_mean
end

"""
construct_cov xx_cov from ensemble x and mean x_mean
"""
function construct_cov(uki::UKIObj{FT, IT}, x::Array{FT,2}, x_mean::Array{FT}) where {FT<:AbstractFloat, IT<:Int}
    N_ens, N_x = uki.N_ens, size(x_mean,1)
    
    cov_weights = uki.cov_weights

    xx_cov = zeros(FT, N_x, N_x)

    for i = 1: N_ens
        xx_cov .+= cov_weights[i]*(x[i,:] - x_mean)*(x[i,:] - x_mean)'
    end

    return xx_cov
end

"""
construct_cov xy_cov from ensemble x and mean x_mean, ensemble y and mean y_mean
"""
function construct_cov(uki::UKIObj{FT, IT}, x::Array{FT,2}, x_mean::Array{FT}, y::Array{FT,2}, y_mean::Array{FT}) where {FT<:AbstractFloat, IT<:Int}
    N_ens, N_x, N_y = uki.N_ens, size(x_mean,1), size(y_mean,1)
    
    cov_weights = uki.cov_weights

    xy_cov = zeros(FT, N_x, N_y)

    for i = 1: N_ens
        xy_cov .+= cov_weights[i]*(x[i,:] - x_mean)*(y[i,:] - y_mean)'
    end

    return xy_cov
end



"""
update uki struct
ens_func: The function g = G(θ)
define the function as 
    ens_func(θ_ens) = MyG(phys_params, θ_ens, other_params)
use G(θ_mean) instead of FG(θ)
"""
function update_ensemble!(uki::UKIObj{FT, IT}, ens_func::Function) where {FT<:AbstractFloat, IT<:Int}
    
    uki.iter += 1
    # update evolution covariance matrix
    if uki.update_freq > 0 && uki.iter%uki.update_freq == 0
        uki.Σ_ω = (2 - uki.α_reg^2)uki.θθ_cov[end]
    end

    θ_mean  = uki.θ_mean[end]
    θθ_cov = uki.θθ_cov[end]
    y = uki.y

    α_reg = uki.α_reg
    r = uki.r
    Σ_ω = uki.Σ_ω
    Σ_ν = uki.Σ_ν

    N_θ, N_y, N_ens = uki.N_θ, uki.N_y, uki.N_ens
    ############# Prediction step:

    θ_p_mean  = α_reg*θ_mean + (1-α_reg)*r
    θθ_p_cov = α_reg^2*θθ_cov + Σ_ω
    


    ############ Generate sigma points
    θ_p = construct_sigma_ensemble(uki, θ_p_mean, θθ_p_cov)
    # play the role of symmetrizing the covariance matrix
    θθ_p_cov = construct_cov(uki, θ_p, θ_p_mean)

    ###########  Analysis step
    g = zeros(FT, N_ens, N_y)
    g .= ens_func(θ_p)

    g_mean = construct_mean(uki, g)
    gg_cov = construct_cov(uki, g, g_mean) + Σ_ν
    θg_cov = construct_cov(uki, θ_p, θ_p_mean, g, g_mean)

    tmp = θg_cov/gg_cov

    θ_mean =  θ_p_mean + tmp*(y - g_mean)

    θθ_cov =  θθ_p_cov - tmp*θg_cov' 


    ########### Save resutls
    push!(uki.y_pred, g_mean) # N_ens x N_data
    push!(uki.θ_mean, θ_mean) # N_ens x N_params
    push!(uki.θθ_cov, θθ_cov) # N_ens x N_data
end



function UKI_Run(s_param, forward::Function, 
    θ0_mean, θθ0_cov,
    y, Σ_η,
    α_reg,
    update_freq,
    N_iter;
    modified_uscented_transform::Bool = true,
    θ_basis = nothing)
    
    θ_names = s_param.θ_names
    
    
    ukiobj = UKIObj(θ_names ,
    θ0_mean, 
    θθ0_cov,
    y,
    Σ_η,
    α_reg,
    update_freq;
    modified_uscented_transform = modified_uscented_transform)
    
    
    
    
    
    ens_func(θ_ens) = (θ_basis == nothing) ? 
    ensemble(s_param, θ_ens, forward) : 
    # θ_ens is N_ens × N_θ
    # θ_basis is N_θ × N_θ_high
    ensemble(s_param, θ_ens * θ_basis, forward)
    
    
    
    
    
    for i in 1:N_iter
        
        update_ensemble!(ukiobj, ens_func) 

        @info "iter : ", i, ukiobj.θ_mean[i]

        @info "iter : ", i, exp.(ukiobj.θ_mean[i])
        @info "iter : ", i, 0.5*(ukiobj.y_pred[i] - ukiobj.y)'*(ukiobj.Σ_η\(ukiobj.y_pred[i] - ukiobj.y))
        @info "iter : ", i, ukiobj.θθ_cov[i]
    end
    
    return ukiobj
    
end


function plot_param_iter(ukiobj::UKIObj{FT, IT}, θ_ref::Array{FT,1}, θ_ref_names::Array{String}) where {FT<:AbstractFloat, IT<:Int}
    
    θ_mean = ukiobj.θ_mean
    θθ_cov = ukiobj.θθ_cov
    
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


function plot_opt_errors(ukiobj::UKIObj{FT, IT}, 
    θ_ref::Union{Array{FT,1}, Nothing} = nothing, 
    transform_func::Union{Function, Nothing} = nothing) where {FT<:AbstractFloat, IT<:Int}
    
    θ_mean = ukiobj.θ_mean
    θθ_cov = ukiobj.θθ_cov
    y_pred = ukiobj.y_pred
    Σ_η = ukiobj.Σ_η
    y = ukiobj.y

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