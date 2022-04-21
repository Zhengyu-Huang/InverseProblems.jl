"""
Iterative ensemble Kalman filtering
Basic stochastic dynamical system

θₙ₊₁ = θₙ
yₙ₊₁ = G(θₙ₊₁) + Δt⁻¹ Ση
"""
mutable struct IEKFObj{FT<:AbstractFloat, IT<:Int}
    "filter_type type"
    filter_type::String
    "vector of parameter names"
    θ_names::Array{String, 1}
    "a vector of arrays of size N_ensemble x N_parameters containing the parameters (in each iteration a new array of parameters is added)"
    θ::Vector{Array{FT, 2}}
    "a vector of arrays of size N_ensemble x N_y containing the predicted observation (in each iteration a new array of predicted observation is added)"
    y_pred::Vector{Array{FT, 2}}
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
    "current iteration number"
    iter::IT
    "number of time steps"
    Nt::IT
    "end time"
    T::FT
end



"""
IEKFObj Constructor
filter_type::String : EKI， EAKI， ETKI
θ0::Array{FT} : prior particles
y::Vector{FT} : observation
Σ_η::Array{FT, 2} : observation covariance
α_reg::Float64 : regularization parameter toward θ0 (0 < α_reg <= 1), default should be 1, without regulariazion
"""
function IEKFObj(
    filter_type::String,
    θ_names::Array{String, 1},
    θ0::Array{FT, 2},
    y::Vector{FT}, # observation
    Σ_η::Array{FT, 2},
    Nt::IT,
    T::FT) where {FT<:AbstractFloat, IT<:Int}

    # ensemble size
    N_ens, N_θ = size(θ0)
    # parameters
    θ = Array{FT, 2}[] # array of Array{FT, 2}'s
    push!(θ, θ0) # insert parameters at end of array (in this case just 1st entry)
    # observations
    N_y = size(y, 1)
    y_pred = Array{FT, 1}[]  # array of Array{FT, 2}'s

    IEKFObj{FT,IT}(filter_type, θ_names, θ, y_pred, y, Σ_η, N_ens, N_θ, N_y, 0, Nt, T)
end



"""
construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
Construct the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_cov(iekf::IEKFObj{FT}, x::Array{FT,2}, x_bar::Array{FT}, y::Array{FT,2}, y_bar::Array{FT}) where {FT<:AbstractFloat}
    N_ens, N_x, N_y = iekf.N_ens, size(x_bar,1), size(y_bar,1)

    xy_cov = zeros(FT, N_x, N_y)

    for i = 1: N_ens
        xy_cov .+= (x[i,:] - x_bar)*(y[i,:] - y_bar)'
    end

    return xy_cov/(N_ens - 1)
 end


function update_ensemble!(iekf::IEKFObj{FT}, ens_func::Function) where {FT<:AbstractFloat}


    # θ: N_ens x N_θ
    N_ens, N_θ, N_y = iekf.N_ens, iekf.N_θ, iekf.N_y
    
    θ = iekf.θ[end]

    Δt = iekf.T/iekf.Nt
    Σ_η = iekf.Σ_η/Δt

    filter_type = iekf.filter_type

    ############# Prediction step

    θ_p_mean =  dropdims(mean(θ, dims=1), dims=1)  

    θ_p = copy(θ)
        
    
    ############# Analysis step
    
    # evaluation G(θ)
    g = zeros(FT, N_ens, N_y)
    g .= ens_func(θ_p)
    g_mean = dropdims(mean(g, dims=1), dims=1)
    
    
    
    # construct square root matrix for  θ̂ - m̂
    Z_p_t = copy(θ_p)
    for j = 1:N_ens;    Z_p_t[j, :] .-=  θ_p_mean;    end
    Z_p_t ./= sqrt(N_ens - 1)
    
    # construct square root matrix for  g - g_mean
    Y_p_t = copy(g)  
    for j = 1:N_ens;  Y_p_t[j, :] .-=  g_mean;  end
    Y_p_t ./= sqrt(N_ens - 1)
    
    #=
    gg_cov = construct_cov(iekf, g, g_mean, g, g_mean) + Σ_η
    θg_cov = construct_cov(iekf, θ_p, θ_p_mean, g, g_mean)
    K   = θg_cov * gg_cov⁻¹ 
        = Z_p * Y_p' * (Y_p *Y_p' + Σ_η)⁻¹ 
        = Z_p * (I + Y_p' * Σ_η⁻¹ * Y_p)⁻¹ * Y_p' + Σ_η⁻¹ 
        = Z_p_t' * (I + P * Γ * P')⁻¹ * Y_p_t * Σ_η⁻¹ 
        = Z_p_t' * P *(I + Γ)⁻¹*P' * Y_p_t * Σ_η⁻¹ 
    =#                       
    X = Y_p_t/Σ_η*Y_p_t'

    svd_X = svd(X)
    P, Γ = svd_X.U, svd_X.S
    
    # compute the mean for EAKI and ETKI
    θ_mean = θ_p_mean + Z_p_t' * (P *( (Γ .+ 1.0) .\ (P' * (Y_p_t * (Σ_η\(iekf.y - g_mean))))))
    
    
    
    if filter_type == "EKI"
        noise = rand(MvNormal(zeros(N_y), Σ_η), N_ens) 

        θ = copy(θ_p) 
        for j = 1:N_ens
            θ[j,:] += Z_p_t' * (P *( (Γ .+ 1.0) .\ (P' * (Y_p_t * (Σ_η\((iekf.y - g[j, :] - noise[:, j]))))))) # N_ens x N_θ
        end
        
    elseif filter_type == "EAKI"
        
        # update particles by Ensemble Adjustment Kalman Filter
        # Dp = F^T Σ F, Σ = F Dp F^T, Dp is non-singular

       
        
        F, sqrt_D_p, V =  trunc_svd(Z_p_t') 
        
        
        # I + Y_p_t/Σ_η*Y_p_t' = P (Γ + I) P'
        # Y = V' /(I + Y_p_t/Σ_η*Y_p_t') * V
        Y = V' * P ./ (Γ .+ 1.0)' * P' * V
        
        svd_Y = svd(Y)
        
        U, D = svd_Y.U, svd_Y.S
        
        
        A = (F .* sqrt_D_p' * U .* sqrt.(D)') * (sqrt_D_p .\ F')
        
        
        θ = similar(θ_p) 
        for j = 1:N_ens
            θ[j, :] .= θ_mean + A * (θ_p[j, :] - θ_p_mean) # N_ens x N_θ
        end
        
        ################# Debug check
        
        # θθ_p_cov = construct_cov(iekf, θ_p, θ_p_mean, θ_p, θ_p_mean)
        # θθ_cov = Z_p_t'*(I - Y_p_t/(Y_p_t'*Y_p_t + Σ_η)*Y_p_t') *Z_p_t
        # θ_mean_debug = dropdims(mean(θ, dims=1), dims=1)
        # θθ_cov_debug = construct_cov(iekf, θ, θ_mean_debug, θ, θ_mean_debug)
        # @info "mean error is ", norm(θ_mean - θ_mean_debug), " cov error is ", norm(θθ_cov - A*Z_p_t'*Z_p_t*A'), norm(θθ_cov - θθ_cov_debug)
        
        
    elseif filter_type == "ETKI"
        
        
        # update particles by Ensemble Adjustment Kalman Filter
        # Dp = F^T Σ F, Σ = F Dp F^T, Dp is non-singular
        # X = Y_p_t/Σ_η*Y_p_t'
        # svd_X = svd(X)
        
        # P, Γ = svd_X.U, svd_X.S
        
        #Original ETKF is  T = P * (Γ .+ 1)^{-1/2}, but it is biased
        T = P ./ sqrt.(Γ .+ 1)' * P'
        
        # Z_p'
        θ = similar(θ_p) 
        for j = 1:N_ens;  θ[j, :] .=  θ_p[j, :] - θ_p_mean;  end
        # Z' = （Z_p * T)' = T' * Z_p
        θ .= T' * θ 
        for j = 1:N_ens;  θ[j, :] .+=  θ_mean;  end
        
        
        ################# Debug check
        
        # Z_p_t = copy(θ_p)
        # for j = 1:N_ens;    Z_p_t[j, :] .-=  θ_p_mean;    end
        # Z_p_t ./= sqrt(N_ens - 1)
        
        
        # θθ_p_cov = construct_cov(iekf, θ_p, θ_p_mean, θ_p, θ_p_mean)
        # θθ_cov = Z_p_t'*(I - Y_p_t/(Y_p_t'*Y_p_t + Σ_η)*Y_p_t') *Z_p_t
        # θ_mean_debug = dropdims(mean(θ, dims=1), dims=1)
        # θθ_cov_debug = construct_cov(iekf, θ, θ_mean_debug, θ, θ_mean_debug)
        # @info "mean error is ", norm(θ_mean - θ_mean_debug), " cov error is ", norm(θθ_cov - Z_p_t'*T*T'*Z_p_t), norm(θθ_cov - θθ_cov_debug)
        
        
    else
        error("Filter type :", filter_type, " has not implemented yet!")
    end
    
    # Save results
    push!(iekf.θ, θ) # N_ens x N_θ
    
    
end


function IEKF_Run(
    filter_type,
    s_param, 
    forward::Function, 
    θ0_mean, θθ0_cov,
    N_ens,
    y, Σ_η,
    Nt,
    T = 1.0;
    exact_init = false)
    
    θ_names = s_param.θ_names
    
    


    θ0 = Array(rand(MvNormal(θ0_mean, θθ0_cov), N_ens)')
    N_θ = length(θ0_mean)
    # correct the  intial ensemble, to ensure exactness
    if exact_init
        # shift mean to 0
        θ0 -= (ones(N_ens) * dropdims(mean(θ0, dims=1), dims=1)')

        # #θ0_new = θ0 * X with covariance θθ0_cov
        U1, S1, V1t = svd(θ0)
        U2, S2, U2t = svd((N_ens - 1.0)*θθ0_cov)
        
        θ0 = θ0 * (V1t * (S1 .\ sqrt.(S2) .* U2t'))
   
        # #add θ0_mean
        θ0 += (ones(N_ens) * θ0_mean')

        
    end


    iekfobj = IEKFObj(filter_type, θ_names,
    θ0, 
    y,
    Σ_η,
    Nt,
    T)
    
    ens_func(θ_ens) = ensemble(s_param, θ_ens, forward) 

    for i = 1:Nt
        update_ensemble!(iekfobj, ens_func) 
    end
    
    return iekfobj
    
end




####################################################################################

"""
Iterative unscented Kalman filtering
Basic stochastic dynamical system

θₙ₊₁ = θₙ
yₙ₊₁ = G(θₙ₊₁) + Δt⁻¹ Ση
    
"""
mutable struct IUKFObj{FT<:AbstractFloat, IT<:Int}
    "vector of parameter names (never used)"
    θ_names::Array{String,1}
"a vector of arrays of size N_ensemble x N_parameters containing the mean of the parameters (in each IUKF iteration a new array of mean is added)"
    θ_mean::Vector{Array{FT, 1}}
    "a vector of arrays of size N_ensemble x (N_parameters x N_parameters) containing the covariance of the parameters (in each IUKF iteration a new array of cov is added)"
    θθ_cov::Vector{Array{FT, 2}}
    "a vector of arrays of size N_ensemble x N_y containing the predicted observation (in each IUKF iteration a new array of predicted observation is added)"
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
    "weights in IUKF"
    c_weights::Union{Array{FT, 1}, Array{FT, 2}}
    mean_weights::Array{FT, 1}
    cov_weights::Array{FT, 1}
    "current iteration number"
    iter::IT
    "number of time steps"
    Nt::IT
    "end time"
    T::FT
end



"""
IUKFObj Constructor 
θ_names::Array{String,1} : parameter name vector
θ0_mean::Array{FT} : prior mean
θθ0_cov::Array{FT, 2} : prior covariance
y::Array{FT,1} : observation 
Σ_η::Array{FT, 2} : observation covariance
Nt::IT : number of Iterations
T : end time
unscented_transform : "original-2n+1", "modified-2n+1", "original-n+2", "modified-n+2" 
"""
function IUKFObj(θ_names::Array{String,1},
                θ0_mean::Array{FT}, 
                θθ0_cov::Array{FT, 2},
                y::Array{FT,1},
                Σ_η,
                Nt::IT,
                T::FT;
                unscented_transform::String = "modified-2n+1") where {FT<:AbstractFloat, IT<:Int}

    
    N_θ = size(θ0_mean,1)
    N_y = size(y, 1)

    if unscented_transform == "original-2n+1" ||  unscented_transform == "modified-2n+1"

        # ensemble size
        N_ens = 2*N_θ+1

        c_weights = zeros(FT, N_θ)
        mean_weights = zeros(FT, N_ens)
        cov_weights = zeros(FT, N_ens)

        κ = 0.0
        β = 2.0
        α = min(sqrt(4/(N_θ + κ)), 1.0)
        λ = α^2*(N_θ + κ) - N_θ


        c_weights[1:N_θ]     .=  sqrt(N_θ + λ)
        mean_weights[1] = λ/(N_θ + λ)
        mean_weights[2:N_ens] .= 1/(2(N_θ + λ))
        cov_weights[1] = λ/(N_θ + λ) + 1 - α^2 + β
        cov_weights[2:N_ens] .= 1/(2(N_θ + λ))

        if unscented_transform == "modified-2n+1"
            mean_weights[1] = 1.0
            mean_weights[2:N_ens] .= 0.0
        end

    elseif unscented_transform == "original-n+2" ||  unscented_transform == "modified-n+2"

        N_ens = N_θ+2
        c_weights = zeros(FT, N_θ, N_ens)
        mean_weights = zeros(FT, N_ens)
        cov_weights = zeros(FT, N_ens)

        # todo cov parameter
        α = N_θ/(4*(N_θ+1))
	
        IM = zeros(FT, N_θ, N_θ+1)
        IM[1,1], IM[1,2] = -1/sqrt(2α), 1/sqrt(2α)
        for i = 2:N_θ
            for j = 1:i
                IM[i,j] = 1/sqrt(α*i*(i+1))
            end
            IM[i,i+1] = -i/sqrt(α*i*(i+1))
        end
        c_weights[:, 2:end] .= IM

        if unscented_transform == "original-n+2"
            mean_weights .= 1/(N_θ+1)
            mean_weights[1] = 0.0

            cov_weights .= α
            cov_weights[1] = 0.0

        else unscented_transform == "modified-n+2"
            mean_weights .= 0.0
            mean_weights[1] = 1.0
            
            cov_weights .= α
            cov_weights[1] = 0.0
        end

    else

        error("unscented_transform: ", unscented_transform, " is not recognized")
    
    end

    

    θ_mean = Array{FT,1}[]  # array of Array{FT, 2}'s
    push!(θ_mean, θ0_mean) # insert parameters at end of array (in this case just 1st entry)
    θθ_cov = Array{FT,2}[] # array of Array{FT, 2}'s
    push!(θθ_cov, θθ0_cov) # insert parameters at end of array (in this case just 1st entry)

    y_pred = Array{FT, 1}[]  # array of Array{FT, 2}'s
   
    
    iter = 0

    IUKFObj{FT,IT}(θ_names, θ_mean, θθ_cov, y_pred, 
                  y,   Σ_η, 
                  N_ens, N_θ, N_y, 
                  c_weights, mean_weights, cov_weights, iter, Nt, T)

end


"""
construct_sigma_ensemble
Construct the sigma ensemble, based on the mean x_mean, and covariance x_cov
"""
function construct_sigma_ensemble(iukf::IUKFObj{FT, IT}, x_mean::Array{FT}, x_cov::Array{FT,2}) where {FT<:AbstractFloat, IT<:Int}
    N_ens = iukf.N_ens
    N_x = size(x_mean,1)
    @assert(N_ens == 2*N_x+1 || N_ens == N_x+2)

    c_weights = iukf.c_weights


    chol_xx_cov = cholesky(Hermitian(x_cov)).L

    if ndims(c_weights) == 1
        x = zeros(FT, N_ens, N_x)
        x[1, :] = x_mean
        for i = 1: N_x
            x[i+1,     :] = x_mean + c_weights[i]*chol_xx_cov[:,i]
            x[i+1+N_x, :] = x_mean - c_weights[i]*chol_xx_cov[:,i]
        end
    elseif ndims(c_weights) == 2
        x = zeros(FT, N_ens, N_x)
        x[1, :] = x_mean
        for i = 2: N_x + 2
	    # @info chol_xx_cov,  c_weights[:, i],  chol_xx_cov * c_weights[:, i]
            x[i,     :] = x_mean + chol_xx_cov * c_weights[:, i]
        end
    else
        error("c_weights dimensionality error")
    end
    return x
end


"""
construct_mean x_mean from ensemble x
"""
function construct_mean(iukf::IUKFObj{FT, IT}, x::Array{FT,2}) where {FT<:AbstractFloat, IT<:Int}
    N_ens, N_x = size(x)

    @assert(iukf.N_ens == N_ens)

    x_mean = zeros(FT, N_x)

    mean_weights = iukf.mean_weights

    
    for i = 1: N_ens
        x_mean += mean_weights[i]*x[i,:]
    end

    return x_mean
end

"""
construct_cov xx_cov from ensemble x and mean x_mean
"""
function construct_cov(iukf::IUKFObj{FT, IT}, x::Array{FT,2}, x_mean::Array{FT}) where {FT<:AbstractFloat, IT<:Int}
    N_ens, N_x = iukf.N_ens, size(x_mean,1)
    
    cov_weights = iukf.cov_weights

    xx_cov = zeros(FT, N_x, N_x)

    for i = 1: N_ens
        xx_cov .+= cov_weights[i]*(x[i,:] - x_mean)*(x[i,:] - x_mean)'
    end

    return xx_cov
end

"""
construct_cov xy_cov from ensemble x and mean x_mean, ensemble y and mean y_mean
"""
function construct_cov(iukf::IUKFObj{FT, IT}, x::Array{FT,2}, x_mean::Array{FT}, y::Array{FT,2}, y_mean::Array{FT}) where {FT<:AbstractFloat, IT<:Int}
    N_ens, N_x, N_y = iukf.N_ens, size(x_mean,1), size(y_mean,1)
    
    cov_weights = iukf.cov_weights

    xy_cov = zeros(FT, N_x, N_y)

    for i = 1: N_ens
        xy_cov .+= cov_weights[i]*(x[i,:] - x_mean)*(y[i,:] - y_mean)'
    end

    return xy_cov
end



"""
update IUKF struct
ens_func: The function g = G(θ)
define the function as 
    ens_func(θ_ens) = MyG(phys_params, θ_ens, other_params)
use G(θ_mean) instead of FG(θ)
"""
function update_ensemble!(iukf::IUKFObj{FT, IT}, ens_func::Function) where {FT<:AbstractFloat, IT<:Int}
    
    iukf.iter += 1


    θ_mean  = iukf.θ_mean[end]
    θθ_cov = iukf.θθ_cov[end]
    y = iukf.y
    
    Δt = iukf.T/iukf.Nt 
    Σ_η = iukf.Σ_η/Δt

    N_θ, N_y, N_ens = iukf.N_θ, iukf.N_y, iukf.N_ens
    ############# Prediction step:
    
    
    

    ############ Generate sigma points
    θ_p_mean = θ_mean
    θθ_p_cov = θθ_cov
    θ_p = construct_sigma_ensemble(iukf, θ_p_mean, θθ_p_cov)
    
    ###########  Analysis step
    g = zeros(FT, N_ens, N_y)
    
    # @info "θ_p = ", θ_p
    g .= ens_func(θ_p)
    
    
    g_mean = construct_mean(iukf, g)
    gg_cov = construct_cov(iukf, g, g_mean) + Σ_η
    θg_cov = construct_cov(iukf, θ_p, θ_p_mean, g, g_mean)
    
 
    tmp = θg_cov / gg_cov

    θ_mean =  θ_p_mean + tmp*(y - g_mean)

    θθ_cov =  θθ_p_cov - tmp*θg_cov' 

    

    ########### Save resutls
    push!(iukf.y_pred, g_mean) # N_ens x N_data
    push!(iukf.θ_mean, θ_mean) # N_ens x N_params
    push!(iukf.θθ_cov, θθ_cov) # N_ens x N_data
end



function IUKF_Run(s_param, forward::Function, 
    θ0_mean, θθ0_cov,
    y, Σ_η,
    Nt;
    T = 1.0,
    unscented_transform::String = "modified-2n+1",
    θ_basis = nothing)
    
    θ_names = s_param.θ_names
    
    
    iukf = IUKFObj(θ_names ,
    θ0_mean, 
    θθ0_cov,
    y,
    Σ_η,
    Nt,
    T;
    unscented_transform = unscented_transform)
    
    
    ens_func(θ_ens) = (θ_basis == nothing) ? 
    ensemble(s_param, θ_ens, forward) : 
    # θ_ens is N_ens × N_θ
    # θ_basis is N_θ × N_θ_high
    ensemble(s_param, θ_ens * θ_basis, forward)
    
    
    for i in 1:Nt
        
        update_ensemble!(iukf, ens_func) 

        
    end
    
    return iukf
    
end


function plot_param_iter(iukf::IUKFObj{FT, IT}, θ_ref::Array{FT,1}, θ_ref_names::Array{String}) where {FT<:AbstractFloat, IT<:Int}
    
    θ_mean = iukf.θ_mean
    θθ_cov = iukf.θθ_cov
    
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


function plot_opt_errors(iukf::IUKFObj{FT, IT}, 
    θ_ref::Union{Array{FT,1}, Nothing} = nothing, 
    transform_func::Union{Function, Nothing} = nothing) where {FT<:AbstractFloat, IT<:Int}
    
    θ_mean = iukf.θ_mean
    θθ_cov = iukf.θθ_cov
    y_pred = iukf.y_pred
    Σ_η = iukf.Σ_η
    y = iukf.y

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







