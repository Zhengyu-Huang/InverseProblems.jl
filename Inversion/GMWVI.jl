using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions
include("GaussianMixture.jl")
include("QuadratureRule.jl")

#改变了更新规则，其中w的速度变为两倍，C的规则全变，m规则不变
"""
GMWVIObj{FT<:AbstractFloat, IT<:Int}
Struct that is used in sampling e^{-Phi_r} with Gaussian mixture gradient descent
"""
mutable struct GMWVIObj{FT<:AbstractFloat, IT<:Int}
    "object name"
    name::String
    "a vector of arrays of size (N_modes) containing the modal weights of the parameters"
    logx_w::Vector{Array{FT, 1}}
    "a vector of arrays of size (N_modes x N_parameters) containing the modal means of the parameters"
    x_mean::Vector{Array{FT, 2}}
    "a vector of arrays of size (N_modes x N_parameters x N_parameters) containing the modal covariances of the parameters"
    xx_cov::Union{Vector{Array{FT, 3}}, Nothing}
    "number of modes"
    N_modes::IT
    "size of x"
    N_x::IT
    "current iteration number"
    iter::IT
    "update covariance or not"
    update_covariance::Bool
    "Cholesky, SVD"
    sqrt_matrix_type::String
    "expectation of Gaussian mixture and its derivatives"
    quadrature_type_GM::String
    c_weight_GM::FT
    c_weights_GM::Union{Array{FT, 2}, Nothing}
    mean_weights_GM::Array{FT, 1}
    N_ens_GM::IT
    "whether correct Hessian approximation"
    Hessian_correct_GM::Bool
    "sample points"
    N_ens::IT
    "quadrature points for expectation, 
     random_sampling,  mean_point,  unscented_transform"
    quadrature_type::String
    "expectation of Gaussian mixture and its derivatives"
    c_weight_Phi::FT
    c_weights::Array{FT, 2}
    mean_weights::Array{FT, 1}
    "weight clipping"
    w_min::FT

    "predicted Phi_r"
    Phi_r_pred::Vector
end



"""
GMWVIObj Constructor 
"""
function GMWVIObj(# initial condition
                x0_w::Array{FT, 1},
                x0_mean::Array{FT, 2},
                xx0_cov::Union{Array{FT, 3}, Nothing};
                update_covariance::Bool = true,
                sqrt_matrix_type::String = "Cholesky",
                # setup for Gaussian mixture part
                quadrature_type_GM::String = "cubature_transform_o5",
                c_weight_GM::FT = sqrt(3.0),
                N_ens_GM::IT = -1,
                Hessian_correct_GM::Bool = true,
                quadrature_type = "unscented_transform",
                c_weight_Phi::FT = sqrt(3.0),
                N_ens::IT = -1,
                w_min::FT = 1.0e-15) where {FT<:AbstractFloat, IT<:Int}

    N_modes, N_x = size(x0_mean)

    logx_w = Array{FT,1}[]      # array of Array{FT, 1}'s
    push!(logx_w, log.(x0_w))   # insert parameters at end of array (in this case just 1st entry)
    x_mean = Array{FT,2}[]      # array of Array{FT, 2}'s
    push!(x_mean, x0_mean)      # insert parameters at end of array (in this case just 1st entry)
    xx_cov = Array{FT,3}[]      # array of Array{FT, 2}'s
    push!(xx_cov, xx0_cov)      # insert parameters at end of array (in this case just 1st entry)
    
    iter = 0
    N_ens_GM, c_weights_GM, mean_weights_GM = generate_quadrature_rule(N_x, quadrature_type_GM; c_weight=c_weight_GM, N_ens=N_ens_GM)
    
    N_ens, c_weights, mean_weights = generate_quadrature_rule(N_x, quadrature_type; c_weight=c_weight_Phi, N_ens=N_ens)
    
    name = "GMWVI"

    Phi_r_pred = []
    GMWVIObj(name,
            logx_w, x_mean, xx_cov, N_modes, N_x,
            iter, update_covariance,
            sqrt_matrix_type,
            ## Gaussian mixture expectation
            quadrature_type_GM, c_weight_GM, c_weights_GM, mean_weights_GM, N_ens_GM, Hessian_correct_GM,
            ## potential function expectation
            N_ens, quadrature_type,
            c_weight_Phi, c_weights, mean_weights, w_min, Phi_r_pred)
end




"""
update gmgd struct
ens_func: The function g = G(x)
define the function as 
    ens_func(x_ens) = MyG(phys_params, x_ens, other_params)
use G(x_mean) instead of FG(x)
"""
function update_ensemble!(gmgd::GMWVIObj{FT, IT}, func::Function, dt_max::FT) where {FT<:AbstractFloat, IT<:Int}
    
    update_covariance = gmgd.update_covariance
    sqrt_matrix_type = gmgd.sqrt_matrix_type

    gmgd.iter += 1
    N_x,  N_modes = gmgd.N_x, gmgd.N_modes

    x_mean  = gmgd.x_mean[end]
    logx_w  = gmgd.logx_w[end]
    xx_cov  = gmgd.xx_cov[end]

    sqrt_xx_cov, inv_sqrt_xx_cov = [], []
    for im = 1:N_modes
        sqrt_cov, inv_sqrt_cov = compute_sqrt_matrix(xx_cov[im,:,:]; type=sqrt_matrix_type) 
        push!(sqrt_xx_cov, sqrt_cov)
        push!(inv_sqrt_xx_cov, inv_sqrt_cov) 
    end


    N_ens = gmgd.N_ens
 
    ############ Generate sigma points
    x_p = zeros(N_modes, N_ens, N_x)
    for im = 1:N_modes
        x_p[im,:,:] = construct_ensemble(x_mean[im,:], sqrt_xx_cov[im]; c_weights = gmgd.c_weights)
    end


    
    ########### function evaluation, for Φᵣ
    Φᵣ, ∇Φᵣ, ∇²Φᵣ = func(x_p)

    ###########  Potential term
    Φᵣ_mean, ∇Φᵣ_mean, ∇²Φᵣ_mean = zeros(N_modes), zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)
    for im = 1:N_modes
        Φᵣ_mean[im], ∇Φᵣ_mean[im,:], ∇²Φᵣ_mean[im,:,:] = compute_expectation(Φᵣ[im,:], ∇Φᵣ[im,:,:], ∇²Φᵣ[im,:,:,:], gmgd.mean_weights) 
    end



    x_mean_n = copy(x_mean)
    xx_cov_n = copy(xx_cov)
    logx_w_n = copy(logx_w)


    ###########  Entropy term
    c_weights_GM, mean_weights_GM, N_ens_GM = gmgd.c_weights_GM, gmgd.mean_weights_GM, gmgd.N_ens_GM
    logρ_mean, ∇logρ_mean, ∇²logρ_mean  = compute_logρ_gm_expectation(exp.(logx_w), x_mean, sqrt_xx_cov, inv_sqrt_xx_cov, c_weights_GM, mean_weights_GM, N_ens_GM, gmgd.Hessian_correct_GM)
    
    ########## update covariance
    for im = 1:N_modes
        dt = dt_max
        # update covariance
        if update_covariance
            Mtemp=Matrix(I, N_x, N_x)+dt*(∇²logρ_mean[im, :, :] + ∇²Φᵣ_mean[im, :, :]);
            Mtemp=Hermitian(Mtemp);
            xx_cov_n[im, :, :] =   inv( Hermitian(Mtemp*inv(xx_cov[im, :, :])*Mtemp  ))
            
            if !isposdef(xx_cov_n[im, :, :])
                @info "error! negative determinant for mode ", im,  x_mean[im, :], xx_cov[im, :, :], inv(xx_cov[im, :, :]), ∇²logρ_mean[im, :, :], ∇²Φᵣ_mean[im, :, :]
                @info " mean residual ", ∇logρ_mean[im, :] , ∇Φᵣ_mean[im, :], ∇logρ_mean[im, :] + ∇Φᵣ_mean[im, :]
                @assert(isposdef(xx_cov_n[im, :, :]))
            end
            
        else
            xx_cov_n[im, :, :] = xx_cov[im, :, :]
        end
    end


    ########## update mean
    for im = 1:N_modes
        dt = dt_max
        # update mean
        x_mean_n[im, :]  =  x_mean[im, :] - dt*(∇logρ_mean[im, :] + ∇Φᵣ_mean[im, :]) 
    end


    ########## update weights
    for im = 1:N_modes
        dt = dt_max

        # update weights
        # dlogwₖ = ∫(ρᴳᴹ - Nₖ)(logρᴳᴹ + Φᵣ)dθ
        # logwₖ +=  -Δt∫Nₖ(logρᴳᴹ + Φᵣ)dθ + Δt ∫ρᴳᴹ(logρᴳᴹ + Φᵣ)dθ
        # the second term is independent of k, it is a normalization term
        logx_w_n[im] = logx_w[im] -2* dt*(logρ_mean[im] + Φᵣ_mean[im])
    end
    # for im = 1:N_modes
    #     @info "mean = ", x_mean[im, :]
    #     @info "mean update = ", norm(∇logρ_mean[im, :] + ∇Φᵣ_mean[im, :]), norm(dt*xx_cov_n[im, :, :]*(∇logρ_mean[im, :] + ∇Φᵣ_mean[im, :]))
    # end
    # @info "Φᵣ_mean = ", Φᵣ_mean
    
    # Normalization
    w_min = gmgd.w_min
    logx_w_n .-= maximum(logx_w_n)
    logx_w_n .-= log( sum(exp.(logx_w_n)) )

    # clipping, such that wₖ ≥ w_min
    x_w_n = exp.(logx_w_n)
    clip_ind = x_w_n .< w_min
    x_w_n[clip_ind] .= w_min
    x_w_n[(!).(clip_ind)] /= (1 - sum(clip_ind)*w_min)/sum(x_w_n[(!).(clip_ind)])
    logx_w_n .= log.(x_w_n)
    
    
    ########### Save resutls
    push!(gmgd.x_mean, x_mean_n)   # N_ens x N_params
    push!(gmgd.xx_cov, xx_cov_n)   # N_ens x N_data
    push!(gmgd.logx_w, logx_w_n)   # N_ens x N_data
    push!(gmgd.Phi_r_pred, Φᵣ_mean)
end



function ensemble(x_ens, forward)
    N_modes, N_ens, N_x = size(x_ens)

    Φᵣ = zeros(N_modes, N_ens)   
    ∇Φᵣ = zeros(N_modes, N_ens, N_x)   
    ∇²Φᵣ = zeros(N_modes, N_ens, N_x, N_x)  

    for im = 1:N_modes
        for i = 1:N_ens
            Φᵣ[im, i], ∇Φᵣ[im, i, :], ∇²Φᵣ[im, i, :, :] = forward(x_ens[im, i, :])
        end
    end

    return Φᵣ, ∇Φᵣ, ∇²Φᵣ 
end



function GMWVI_Run(
    forward::Function, # return Φᵣ, ∇Φᵣ, ∇²Φᵣ
    T::FT,
    N_iter::IT,
    # Initial condition
    x0_w::Array{FT, 1}, x0_mean::Array{FT, 2}, xx0_cov::Array{FT, 3};
    update_covariance::Bool = true, 
    sqrt_matrix_type::String = "Cholesky",
    # setup for Gaussian mixture part
    quadrature_type_GM::String = "cubature_transform_o5",
    c_weight_GM::FT = sqrt(3.0),
    N_ens_GM::IT = -1,
    Hessian_correct_GM::Bool = true,
    quadrature_type = "unscented_transform",
    c_weight_Phi::FT = sqrt(3.0),
    N_ens::IT = -1,
    w_min::FT = 1.0e-15) where {FT<:AbstractFloat, IT<:Int}
    

    gmwviobj = GMWVIObj(# initial condition
        x0_w, x0_mean, xx0_cov;
        update_covariance = update_covariance,
        sqrt_matrix_type = sqrt_matrix_type,
        # setup for Gaussian mixture part
        quadrature_type_GM = quadrature_type_GM,
        c_weight_GM = c_weight_GM,
        N_ens_GM = N_ens_GM,
        Hessian_correct_GM = Hessian_correct_GM,
        quadrature_type = quadrature_type,
        c_weight_Phi = c_weight_Phi,
        N_ens = N_ens,
        w_min = w_min) 

    func(x_ens) = ensemble(x_ens, forward)
    
    dt = T/N_iter
    for i in 1:N_iter
        if i%max(1,div(N_iter, 10)) == 0  @info "iter = ", i, " / ", N_iter  end
        
        update_ensemble!(gmwviobj, func, dt) 
    end
    
    return gmwviobj

end


#——————————————————mine——————————————————————————
function Gaussian_mixture_WGFVI(func_dPhi, w0, μ0, Σ0; N_iter = 100, dt = 1.0e-3, Hessian_correct_GM=true)

    N_modes, N_θ = size(μ0)

    
    T =  N_iter * dt
    N_modes = 1
    x0_w = w0
    x0_mean = μ0
    xx0_cov = Σ0
    sqrt_matrix_type = "Cholesky"
    
    objs = []

    gmwviobj = GMWVI_Run(
    func_dPhi, 
    T,
    N_iter,
    # Initial condition
    x0_w, x0_mean, xx0_cov;
    sqrt_matrix_type = sqrt_matrix_type,
    # setup for Gaussian mixture part
    quadrature_type_GM = "mean_point",
    quadrature_type = "mean_point",
    Hessian_correct_GM = Hessian_correct_GM)
        
    push!(objs, gmwviobj)

    return objs
end


