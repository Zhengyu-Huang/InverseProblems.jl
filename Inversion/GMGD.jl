using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions



function compute_sqrt_matrix(C; inverse=false, type="Cholesky")
    if type == "Cholesky"
        √C = cholesky(Hermitian(C)).L
        if inverse
            √C = inv(√C)
        end
    elseif type == "SVD"
        U, D, _ = svd(Hermitian(C))
        √C = U  * Diag(sqrt.(D))
        if inverse
            √C = Diag(sqrt.(1.0/D)) * U.T  
        end
    else
        print("Type ", type, " for computing sqrt matrix has not implemented.")
    end
    return √C 
end

"""
GMGDObj{FT<:AbstractFloat, IT<:Int}
Struct that is used in sampling e^{-V} with Gaussian mixture gradient descent
"""
mutable struct GMGDObj{FT<:AbstractFloat, IT<:Int}
    "a vector of arrays of size (N_modes) containing the modal weights of the parameters"
    logx_w::Vector{Array{FT, 1}}
    "a vector of arrays of size (N_modes x N_parameters) containing the modal means of the parameters"
    x_mean::Vector{Array{FT, 2}}
    "a vector of arrays of size (N_modes x N_parameters x N_parameters) containing the modal covariances of the parameters"
    xx_cov::Union{Vector{Array{FT, 3}}, Nothing}
    "sample points"
    N_ens::IT
    "number of modes"
    N_modes::IT
    "size of x"
    N_x::IT
    "current iteration number"
    iter::IT
    "update covariance or not"
    update_covariance::Bool
    "quadrature point parameter"
    quadrature_param::FT
    "Cholesky, SVD"
    sqrt_matrix_type::String
    "quadrature points for expectation, 
     random_sampling,  mean_point,  unscented_transform"
    quadrature_type::String
    "derivative_free: 0, first_order: 1, second_order: 2"
    gradient_computation_order::Int64
    "when Bayesian_inverse_problem is true :  function is F, 
     otherwise the function is Phi_R,  Phi_R = 1/2 F ⋅ F"
    Bayesian_inverse_problem::Bool
end



"""
GMGDObj Constructor 
"""
function GMGDObj(# initial condition
                x0_w::Array{FT, 1},
                x0_mean::Array{FT, 2},
                xx0_cov::Union{Array{FT, 3}, Nothing};
                # setup
                update_covariance::Bool;
                quadrature_param::FT = 0.1,
                sqrt_matrix_type::String = "Cholesky",
                quadrature_type::String = "unscented_transform",
                Bayesian_inverse_problem::Bool = false,
                N_ens::IT = 1) where {FT<:AbstractFloat, IT<:Int}

    N_modes, N_x = size(x0_mean)

    logx_w = Array{FT,1}[]      # array of Array{FT, 1}'s
    push!(logx_w, log.(x0_w))   # insert parameters at end of array (in this case just 1st entry)
    x_mean = Array{FT,2}[]      # array of Array{FT, 2}'s
    push!(x_mean, x0_mean)      # insert parameters at end of array (in this case just 1st entry)
    xx_cov = Array{FT,3}[]      # array of Array{FT, 2}'s
    push!(xx_cov, xx0_cov)      # insert parameters at end of array (in this case just 1st entry)
    
    iter = 0

    if quadrature_type == "mean_point"
         N_ens = 1
    elseif quadrature_type == "unscented_transform"
         N_ens = 2N_x + 1
    else
        @assert(N_ens > 0)
    end

    GMGDObj(logx_w, x_mean, xx_cov, N_ens,
                  N_modes, 
                  N_x,
                  iter,
                  update_covariance,
                  ##
                  quadrature_param,
                  quadrature_type,
                  gradient_computation_order,
                  Bayesian_inverse_problem)
end


"""
construct_ensemble
Construct the ensemble, based on the mean x_mean, and covariance x_cov
"""
function construct_ensemble(gmgd::GMGDObj{FT, IT}, x_means::Array{FT,2}, x_covs::Array{FT,3}) where {FT<:AbstractFloat, IT<:Int}
    expectation_method = gmgd.expectation_method
    N_modes, N_ens = gmgd.N_modes, gmgd.N_ens
    N_x = size(x_means[1, :],1)
    
    if expectation_method == "random_sampling"
        
        xs = zeros(FT, N_modes, N_ens, N_x)
        for im = 1:N_modes
            chol_xx_cov = cholesky(Hermitian(x_covs[im,:,:])).L
            xs[im, :, :] = ones(N_ens)*x_means[im, :]' + rand(Normal(0, 1), N_ens, N_x) * chol_xx_cov'
        end

    elseif expectation_method == "mean_point"
        xs = zeros(FT, N_modes, N_ens, N_x)
        
        for im = 1:N_modes
            xs[im, 1, :] = x_means[im, :]
        end

    elseif expectation_method == "unscented_transform"
        xs = zeros(FT, N_modes, N_ens, N_x)
        
        for im = 1:N_modes
            chol_xx_cov = compute_sqrt_matrix(x_covs[im,:,:]; inverse=false, type=gmgd.sqrt_matrix_type) 
            xs[im, 1, :] = x_means[im, :]
            for i = 1: N_x
                xs[im, i+1,     :] = x_means[im, :] + c_weight*chol_xx_cov[:,i]
                xs[im, i+1+N_x, :] = x_means[im, :] - c_weight*chol_xx_cov[:,i]
            end
        end

    else 
        error("expectation_method ", expectation_method, " has not implemented!")

    end

    return xs
end


"""
construct_mean x_mean from ensemble x
"""
function construct_mean(gmgd::GMGDObj{FT, IT}, xs::Array{FT}) where {FT<:AbstractFloat, IT<:Int}
    N_modes = gmgd.N_modes
    # xs is a N_modes by N_ens by size(x) array 
    ndims_x = ndims(xs) - 2
    sizes_x = size(xs)[3:end]

    x_means = zeros(FT, N_modes, sizes_x...)
    mean_weights = 1/gmgd.N_ens
    for im = 1:N_modes
        x_means[im, repeat([:],ndims_x)...] = sum(mean_weights*xs[im, repeat([:],ndims_x+1)...], dims=1)[1, repeat([:],ndims_x)...] 
    end
    return x_means
end






function Gaussian_density_helper(x_mean::Array{FT,1}, xx_cov::Array{FT,2}, x::Array{FT,1}) where {FT<:AbstractFloat}
    return exp( -1/2*((x - x_mean)'* (xx_cov\(x - x_mean)) )) / ( sqrt(det(xx_cov)) )
end

function Gaussian_mixture_density_derivatives(x_w::Array{FT,1}, x_mean::Array{FT,2}, xx_cov::Array{FT,3}, x::Array{FT,1}) where {FT<:AbstractFloat}
    N_modes, N_x = size(x_mean)

    ρ = 0.0
    ∇ρ = zeros(N_x)
    ∇²ρ = zeros(N_x, N_x)
   
    for i = 1:N_modes
        ρᵢ   = Gaussian_density_helper(x_mean[i,:], xx_cov[i,:,:], x)
        ρ   += x_w[i]*ρᵢ
        temp = xx_cov[i,:,:]\(x_mean[i,:] - x)
        ∇ρ  += x_w[i]*ρᵢ*temp
        ∇²ρ += x_w[i]*ρᵢ*( temp * temp' - inv(xx_cov[i,:,:]) )
    end
    return ρ, ∇ρ, ∇²ρ
end



function compute_logρ_gm(x_p, x_w, x_mean, xx_cov)
    N_modes, N_ens, N_x = size(x_p)
    logρ = zeros(N_modes, N_ens)
    ∇logρ = zeros(N_modes, N_ens, N_x)
    ∇²logρ = zeros(N_modes, N_ens, N_x, N_x)
    for im = 1:N_modes
        for i = 1:N_ens
            ρ, ∇ρ, ∇²ρ = Gaussian_mixture_density_derivatives(x_w, x_mean, xx_cov, x_p[im, i, :])
            logρ[im, i]         =   log(  ρ  )
            ∇logρ[im, i, :]     =   ∇ρ/ρ
            ∇²logρ[im, i, :, :] =  (∇²ρ*ρ - ∇ρ*∇ρ')/ρ^2
        end
    end

    return logρ, ∇logρ, ∇²logρ
end




function compute_logρ_gm_expectation(x_w, x_mean, xx_cov)
    N_modes = size(x_mean, 1)

    logρ_mean, ∇logρ_mean, ∇²logρ_mean  = construct_mean(gmgd, logρ), construct_mean(gmgd, ∇logρ), construct_mean(gmgd, ∇²logρ)
    
    logρ, ∇logρ, ∇²logρ = compute_logρ_gm(x_p,  x_w, x_mean, xx_cov)
    construct_mean(gmgd, logρ), construct_mean(gmgd, ∇logρ), construct_mean(gmgd, ∇²logρ)
   return  logρ_mean, ∇logρ_mean, ∇²logρ_mean
end





function compute_expectation(gmgd, x_mean, xx_cov, V, ∇V, ∇²V)
    N_modes, N_x, _ = size(xx_cov)
    Φᵣ_mean, ∇Φᵣ_mean, ∇²Φᵣ_mean = zeros(N_modes), zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)

    if gmgd.Bayesian_inverse_problem # Φᵣ = FᵀF/2 
        if gmgd.gradient_computation_order == 0
            α = gmgd.α
            _, N_ens, N_f = size(V)
            a = zeros(N_x, N_f)
            b = zeros(N_x, N_f)
            c = zeros(N_f)

            for im = 1:N_modes
                c = V[im, 1, :]
                for i = 1:N_x
                    a[i, :] = (V[im, i+1, :] - V[im, i+N_x+1, :])/(2*α)
                    b[i, :] = (V[im, i+1, :] + V[im, i+N_x+1, :] - 2*V[im, 1, :])/(2*α^2)
                end
                inv_√x_covs = compute_sqrt_matrix(xx_cov[im,:,:]; inverse=true, type="Cholesky")
                ATA = a * a.T
                BTB = b * b.T
                BTA = b * a.T
                BTc = b * c
                cTc = c.T * c
                Φᵣ_mean[im] = 1/2*(sum(ATA) + 2*tr(ATA) + tr(BTB) + cTc)
                ∇Φᵣ_mean[im, :] = inv_√x_covs.T*(sum(BTA,dims=2) + 2*diag(BTA) + BTc)
                ∇²Φᵣ_mean[im, :, :] = 1/2*inv_√x_covs.T*( Diagonal(4*sum(ATA, dims=2) + 8*diag(ATA) + 4*ATc) + BTB)*inv_√x_covs
            end

        else 
            print("Compute expectation for Bayesian inverse problem. 
                   Gradient computation order ", gmgd.gradient_computation_order, " has not implemented.")
        end

    else # general sampling problem Φᵣ = V 
        assert(gmgd.gradient_computation_order == 2)
        Φᵣ_mean = compute_mean(gmgd, V)
        ∇Φᵣ_mean = compute_mean(gmgd, ∇V) 
        ∇²Φᵣ_mean = compute_mean(gmgd, ∇²V)


    return Φᵣ_mean, ∇Φᵣ_mean, ∇²Φᵣ_mean
end



"""
update gmgd struct
ens_func: The function g = G(x)
define the function as 
    ens_func(x_ens) = MyG(phys_params, x_ens, other_params)
use G(x_mean) instead of FG(x)
"""
function update_ensemble!(gmgd::GMGDObj{FT, IT}, func::Function, dt::FT) where {FT<:AbstractFloat, IT<:Int}
    
    metric = gmgd.metric
    update_covariance = gmgd.update_covariance
    
    gmgd.iter += 1
    N_x,  N_modes = gmgd.N_x, gmgd.N_modes

    x_mean  = gmgd.x_mean[end]
    logx_w  = gmgd.logx_w[end]
    xx_cov  = gmgd.xx_cov[end]


    ###########  Entropy term
    logρ_mean, ∇logρ_mean, ∇²logρ_mean  = compute_logρ_gm_expectation(exp.(logx_w), x_mean, xx_cov)
    
    ############ Generate sigma points
    x_p = construct_ensemble(gmgd, x_mean, xx_cov)
    ###########  Potential term
    V, ∇V, ∇²V = func(x_p)
    Φᵣ_mean, ∇Φᵣ_mean, ∇²Φᵣ_mean = compute_expectation(gmgd, x_mean, xx_cov, V, ∇V, ∇²V)

    x_mean_n = copy(x_mean)
    xx_cov_n = copy(xx_cov)
    logx_w_n = copy(logx_w)


    if metric == "Fisher-Rao"
        for im = 1:N_modes
            x_mean_n[im, :]  =  x_mean[im, :] - dt*xx_cov[im, :, :]*(∇logρ_mean[im, :] + ∇Φᵣ_mean[im, :]) 
 
            if update_covariance
                xx_cov_n[im, :, :] =  inv( inv(xx_cov[im, :, :]) + dt*(∇²logρ_mean[im, :, :] + ∇²Φᵣ_mean[im, :, :]) )
                # xx_cov_n[im, :, :] =  (1 + dt)*inv( inv(xx_cov[im, :, :]) + dt*(∇²logρ_mean[im, :, :] + ∇²V_mean[im, :, :]) )
                
                if det(xx_cov_n[im, :, :]) <= 0.0
                    @info xx_cov[im, :, :], ∇²logρ_mean[im, :, :], ∇²Φᵣ_mean[im, :, :]
                end
                
            else
                xx_cov_n[im, :, :] = xx_cov[im, :, :]
            end
            
            
            ρlogρ_Φᵣ = 0 
            for im = 1:N_modes
                ρlogρ_Φᵣ += exp(logx_w[im])*(logρ_mean[im] + Φᵣ_mean[im])
            end
            logx_w_n[im] = logx_w[im] - dt*(logρ_mean[im] + Φᵣ_mean[im] - ρlogρ_Φᵣ)
            
        end
       
    end

    # Normalization
    logx_w_n .-= maximum(logx_w_n)
    logx_w_n .-= log( sum(exp.(logx_w_n)) )


    ########### Save resutls
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

function GMGD_Run(
    forward::Function, 
    N_iter::IT,
    T::FT,
    metric::String,
    update_covariance::Bool, 
    x0_w::Array{FT, 1}, x0_mean::Array{FT, 2}, xx0_cov::Array{FT, 3},
    expectation_method::String = "unscented_transform",
    N_ens::IT = 1) where {FT<:AbstractFloat, IT<:Int}
    
    gmgdobj = GMGDObj(
    metric, 
    update_covariance, 
    x0_w, x0_mean, xx0_cov,
    expectation_method, N_ens)
     
    func(x_ens) = ensemble(x_ens, forward)  
    
    dt = T/N_iter
    for i in 1:N_iter
        update_ensemble!(gmgdobj, func, dt) 
    end
    
    return gmgdobj
    
end

######################### TEST #######################################





