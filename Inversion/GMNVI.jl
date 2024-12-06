using Random
using PyPlot
using Distributions
using LinearAlgebra
using ForwardDiff
using DocStringExtensions
include("QuadratureRule.jl")
include("GaussianMixture.jl")

mutable struct GMNVIObj{FT<:AbstractFloat, IT<:Int}
    "object name"
    name::String
    "a vector of arrays of size (N_modes) containing the modal weights of the parameters"
    logx_w::Vector{Array{FT, 1}} #FT是类型，每一步是一个Array{FT, 1}，存的是logwi
    "a vector of arrays of size (N_modes x N_parameters) containing the modal means of the parameters"
    x_mean::Vector{Array{FT, 2}} #每一步期望
    "a vector of arrays of size (N_modes x N_parameters x N_parameters) containing the modal covariances of the parameters"
    xx_cov::Union{Vector{Array{FT, 3}}, Nothing} #每一步的cov
    "number of modes"
    N_modes::IT
    "size of x"
    N_x::IT
    "current iteration number"
    iter::IT
    "update covariance or not"
    update_covariance::Bool
    "weather to keep covariance matrix diagonal"
    diagonal_covariance::Bool
    "Cholesky, SVD"
    sqrt_matrix_type::String
    "expectation of Gaussian mixture and its derivatives"
    quadrature_type_GM::String
    c_weight_GM::FT
    c_weights_GM::Array{FT, 2}
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
    c_weight_BIP::FT
    c_weights::Array{FT, 2}
    mean_weights::Array{FT, 1}
    "weight clipping"
    w_min::FT

    
end


function GMNVIObj(# initial condition  #初始化这个struct类
                x0_w::Array{FT, 1},
                x0_mean::Array{FT, 2},
                xx0_cov::Union{Array{FT, 3}, Nothing};
                update_covariance::Bool = true,
                diagonal_covariance::Bool = true,
                sqrt_matrix_type::String = "Cholesky",
                # setup for Gaussian mixture part
                quadrature_type_GM::String = "mean_point",
                c_weight_GM::FT = sqrt(3.0),
                N_ens_GM::IT = -1,
                Hessian_correct_GM::Bool = true,
                quadrature_type = "mean_point",
                c_weight_BIP::FT = sqrt(3.0),
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
    N_ens_GM, c_weights_GM, mean_weights_GM = generate_quadrature_rule(N_x, quadrature_type_GM; c_weight=c_weight_GM)
    
    N_ens, c_weights, mean_weights = generate_quadrature_rule(N_x, quadrature_type; c_weight=c_weight_BIP, N_ens=N_ens)

    name = "NGF-VI"
    GMNVIObj(name,
            logx_w, x_mean, xx_cov, N_modes, N_x,
            iter, update_covariance, diagonal_covariance,
            sqrt_matrix_type,
            ## Gaussian mixture expectation
            quadrature_type_GM, c_weight_GM, c_weights_GM, mean_weights_GM, N_ens_GM, Hessian_correct_GM,
            ## potential function expectation
            N_ens, quadrature_type, 
            c_weight_BIP, c_weights, mean_weights, w_min)
end

   

function update_ensemble!(gmgd::GMNVIObj{FT, IT}, func::Function, dt_max::FT) where {FT<:AbstractFloat, IT<:Int} #从某一步到下一步的步骤
    
    update_covariance = gmgd.update_covariance
    sqrt_matrix_type = gmgd.sqrt_matrix_type
    diagonal_covariance = gmgd.diagonal_covariance
    Hessian_correct = gmgd.Hessian_correct_GM

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

    ###########  Entropy term
    N_ens_GM, c_weights_GM, mean_weights_GM = gmgd.N_ens_GM, gmgd.c_weights_GM, gmgd.mean_weights_GM

    logρ_mean, ∇logρ_mean, _ = compute_logρ_gm_expectation(exp.(logx_w), x_mean, sqrt_xx_cov, inv_sqrt_xx_cov, c_weights_GM, mean_weights_GM, N_ens_GM, Hessian_correct)
    ∇²logρ_mean = zeros(N_modes, N_x, N_x)

    if diagonal_covariance
        #TODO there is no need to compute Hesssian in the next line
        logρ_mean, ∇logρ_mean, _ = compute_logρ_gm_expectation(exp.(logx_w), x_mean, sqrt_xx_cov, inv_sqrt_xx_cov, c_weights_GM, mean_weights_GM, N_ens_GM, Hessian_correct)
        # compute Hessian 
        ∇²logρ_mean = zeros(N_modes, N_x, N_x)
        for dim = 1 : N_x
            sqrt_xx_cov_dim, inv_sqrt_xx_cov_dim = [], []
            for im = 1:N_modes
                sqrt_cov_dim, inv_sqrt_cov_dim = compute_sqrt_matrix(xx_cov[im,dim:dim,dim:dim]; type=sqrt_matrix_type) 
                push!(sqrt_xx_cov_dim, sqrt_cov_dim)
                push!(inv_sqrt_xx_cov_dim, inv_sqrt_cov_dim) 
            end
            c_weights_GM_dim = zeros(1, size(c_weights_GM,2)) 
            c_weights_GM_dim[1,:] = c_weights_GM[dim,:]
            
            x_w = exp.(logx_w) / sum(exp.(logx_w))
            xs = zeros(size(x_mean, 1), size(c_weights_GM, 2), size(x_mean, 2))
            for im = 1:N_modes
                xs[im,:,:] = construct_ensemble(x_mean[im, :], sqrt_xx_cov[im]; c_weights = c_weights_GM)
            end
            
            logρ, ∇logρ, ∇²logρ = compute_logρ_gm(xs, x_w, x_mean, inv_sqrt_xx_cov, Hessian_correct)
        
            for im = 1:N_modes
                _,_,∇²logρ_mean[im,dim:dim,dim:dim] = compute_expectation(logρ[im,:], ∇logρ[im,:,dim:dim], ∇²logρ[im,:,dim:dim,dim:dim], mean_weights_GM)
            end
        end
    else
        logρ_mean, ∇logρ_mean, ∇²logρ_mean = compute_logρ_gm_expectation(exp.(logx_w), x_mean, sqrt_xx_cov, inv_sqrt_xx_cov, c_weights_GM, mean_weights_GM, N_ens_GM, Hessian_correct)
    end 

   ############ Generate sigma points
    N_ens = gmgd.N_ens
    x_p = zeros(N_modes, N_ens, N_x)

    for im = 1:N_modes
        x_p[im,:,:] = construct_ensemble(x_mean[im,:], sqrt_xx_cov[im]; c_weights = gmgd.c_weights) #采样的点
    end
    ###########  Potential term
    V, ∇V, ∇²V = func(x_p)

    Φᵣ_mean, ∇Φᵣ_mean, ∇²Φᵣ_mean = zeros(N_modes), zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)
    for im = 1:N_modes
        Φᵣ_mean[im], ∇Φᵣ_mean[im,:], ∇²Φᵣ_mean[im,:,:] = compute_expectation(V[im,:], ∇V[im,:,:], ∇²V[im,:,:,:], gmgd.mean_weights) 
    end

    x_mean_n = copy(x_mean) 
    xx_cov_n = copy(xx_cov)
    logx_w_n = copy(logx_w)


    ############# Compute time step 
    matrix_norm = []
    for i = 1 : N_modes
        push!(matrix_norm, opnorm(xx_cov[i,:,:] * (∇²logρ_mean[i, :, :] + ∇²Φᵣ_mean[i, :, :]), 2))
    end
    dt = min(dt_max,  0.99 / (maximum(matrix_norm))) # keep the matrix postive definite.


    ############## update covariance, mean, and weight
    for im = 1:N_modes
        ############## update covariance
        if update_covariance
            tempim = zeros(N_x,N_x)
            if diagonal_covariance 
                for ii = 1 : N_x
                    tempim[ii,ii] = ∇²logρ_mean[im, ii, ii] + ∇²Φᵣ_mean[im, ii, ii]
                end
            else
                tempim = ∇²logρ_mean[im,:,:] + ∇²Φᵣ_mean[im,:,:]
            end

            xx_cov_n[im, :, :] = inv(inv(xx_cov[im, :, :]) + dt * (tempim ))
        
            if det(xx_cov_n[im, :, :]) <= 0.0
                @info "error! negative determinant for mode ", im,   inv(xx_cov[im, :, :]), dt*(∇²logρ_mean[im, :, :]+∇²Φᵣ_mean[im, :, :])
                @info " mean residual ", ∇logρ_mean[im, :] , ∇Φᵣ_mean[im, :], ∇logρ_mean[im, :] + ∇Φᵣ_mean[im, :]
                @info "norm =  ", dt*norm(xx_cov[im,:,:], Inf) * norm(∇²logρ_mean[im, :, :] + ∇²Φᵣ_mean[im, :, :], Inf)
                @info "matrix = ", inv(xx_cov[im, :, :]) + dt * (tempim )
            end
            
        else
            xx_cov_n[im, :, :] = xx_cov[im, :, :]
        end
     
        ############## update means
        x_mean_n[im, :]  =  x_mean[im, :] - dt*xx_cov_n[im, :, :]*(∇logρ_mean[im, :] + ∇Φᵣ_mean[im, :]) 
        
        ############## update weights
        logx_w_n[im] = logx_w[im] - dt*(logρ_mean[im] + Φᵣ_mean[im]) #- ρlogρ_Φᵣ

    end
       
    # Normalization
    w_min = gmgd.w_min
    logx_w_n .-= maximum(logx_w_n)
    logx_w_n .-= log( sum(exp.(logx_w_n)) )
    x_w_n = exp.(logx_w_n)
    clip_ind = x_w_n .< w_min
    x_w_n[clip_ind] .= w_min
    x_w_n[(!).(clip_ind)] /= (1 - sum(clip_ind)*w_min)/sum(x_w_n[(!).(clip_ind)])
    logx_w_n .= log.(x_w_n)
    
    
    ########### Save results
    push!(gmgd.x_mean, x_mean_n)   # N_ens x N_params
    push!(gmgd.xx_cov, xx_cov_n)   # N_ens x N_data
    push!(gmgd.logx_w, logx_w_n)   # N_ens x N_data
end

function ensemble(x_ens, forward)
    N_modes, N_ens, N_x = size(x_ens)

    V = zeros(N_modes, N_ens)   
    ∇V = zeros(N_modes, N_ens, N_x)   
    ∇²V = zeros(N_modes, N_ens, N_x, N_x)  

    for im = 1:N_modes
        for i = 1:N_ens
            V[im, i], ∇V[im, i, :], ∇²V[im, i, :, :] = forward(x_ens[im, i, :])
        end
    end

    return V, ∇V, ∇²V 
end


function GMNVI_Run(
    forward::Function, 
    T::FT,
    N_iter::IT,
    # Initial condition
    x0_w::Array{FT, 1}, x0_mean::Array{FT, 2}, xx0_cov::Array{FT, 3}; 
    update_covariance::Bool = true, 
    diagonal_covariance::Bool = true,
    sqrt_matrix_type::String = "Cholesky",
    # setup for Gaussian mixture part
    quadrature_type_GM::String = "mean_point",
    c_weight_GM::FT = sqrt(3.0),
    quadrature_type = "mean_point",
    c_weight_BIP::FT = sqrt(3.0), #输入α
    N_ens::IT = -1,
    w_min::FT = 1.0e-15) where {FT<:AbstractFloat, IT<:Int}
    

    gmnviobj = GMNVIObj(# initial condition
        x0_w, x0_mean, xx0_cov;
        update_covariance = update_covariance,
        diagonal_covariance = diagonal_covariance,
        sqrt_matrix_type = sqrt_matrix_type,
        # setup for Gaussian mixture part
        quadrature_type_GM = quadrature_type_GM,
        c_weight_GM = c_weight_GM,
        quadrature_type = quadrature_type,
        c_weight_BIP = c_weight_BIP,
        N_ens = N_ens,
        w_min = w_min) 

    func(x_ens) =  ensemble(x_ens, forward)  
    
    dt = T/N_iter
    for i in 1:N_iter
        if i%div(N_iter, 10) == 0  @info "iter = ", i, " / ", N_iter  end
        
        update_ensemble!(gmnviobj, func, dt) 
    end
    
    return gmnviobj
    
end



function Gaussian_mixture_NGFVI(func_V, w0, μ0, Σ0; diagonal_covariance::Bool = true, N_iter = 100, dt = 1.0e-3)

    N_modes, N_θ = size(μ0)
    
    T =  N_iter * dt
    x0_w = w0
    x0_mean = μ0
    xx0_cov = Σ0
    sqrt_matrix_type = "Cholesky"
    
    objs = []

       
    gmnviobj = GMNVI_Run(
    func_V, 
    T,
    N_iter,
    # Initial condition
    x0_w, x0_mean, xx0_cov;
    sqrt_matrix_type = sqrt_matrix_type,
    diagonal_covariance = diagonal_covariance,
    # setup for Gaussian mixture part
    quadrature_type_GM = "mean_point",
    # setup for potential function part
    quadrature_type = "mean_point")
    
    push!(objs, gmnviobj)


    return objs
end


