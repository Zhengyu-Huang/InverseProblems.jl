using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions
include("QuadratureRule.jl")


"""
GMGDObj{FT<:AbstractFloat, IT<:Int}
Struct that is used in sampling e^{-V} with Gaussian mixture gradient descent
"""
mutable struct GMGDObj{FT<:AbstractFloat, IT<:Int}
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
    "when Bayesian_inverse_problem is true :  function is F, 
     otherwise the function is Phi_R,  Phi_R = 1/2 F ⋅ F"
    Bayesian_inverse_problem::Bool
    "Bayesian inverse problem observation dimension"
    N_f::IT
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

    "predicted phi_r"
    phi_r_pred::Vector
    
end



"""
GMGDObj Constructor 
"""
function GMGDObj(# initial condition
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
                # setup for potential function part
                Bayesian_inverse_problem::Bool = false,
                N_f::IT = 1,
                quadrature_type = "unscented_transform",
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
    N_ens_GM, c_weights_GM, mean_weights_GM = generate_quadrature_rule(N_x, quadrature_type_GM; c_weight=c_weight_GM, N_ens=N_ens_GM)
    
    N_ens, c_weights, mean_weights = generate_quadrature_rule(N_x, quadrature_type; c_weight=c_weight_BIP, N_ens=N_ens)
    
     
    name = (Bayesian_inverse_problem ? "Derivative free GMGD" : "GMGD")

    phi_r_pred = []
    GMGDObj(name,
            logx_w, x_mean, xx_cov, N_modes, N_x,
            iter, update_covariance,
            sqrt_matrix_type,
            ## Gaussian mixture expectation
            quadrature_type_GM, c_weight_GM, c_weights_GM, mean_weights_GM, N_ens_GM, Hessian_correct_GM,
            ## potential function expectation
            Bayesian_inverse_problem, N_f, N_ens, quadrature_type,
            c_weight_BIP, c_weights, mean_weights, w_min, phi_r_pred)
end



# avoid computing 1/(2π^N_x/2)
function Gaussian_density_helper(x_mean::Array{FT,1}, inv_sqrt_xx_cov, x::Array{FT,1}) where {FT<:AbstractFloat}
    return exp( -1/2*((x - x_mean)'* (inv_sqrt_xx_cov'*inv_sqrt_xx_cov*(x - x_mean)) )) * abs(det(inv_sqrt_xx_cov))
end


# avoid computing 1/(2π^N_x/2) for ρ, ∇ρ, ∇²ρ
function Gaussian_mixture_density_derivatives(x_w::Array{FT,1}, x_mean::Array{FT,2}, inv_sqrt_xx_cov, x::Array{FT,1}, Hessian_correct_GM::Bool) where {FT<:AbstractFloat}
    N_modes, N_x = size(x_mean)

    ρ = 0.0
    ∇ρ = zeros(N_x)
    ∇²ρ = zeros(N_x, N_x)
   
    for i = 1:N_modes
        temp = inv_sqrt_xx_cov[i]'*inv_sqrt_xx_cov[i]*(x_mean[i,:] - x)
        ρᵢ   = Gaussian_density_helper(x_mean[i,:], inv_sqrt_xx_cov[i], x)
    
        ρ   += x_w[i]*ρᵢ
        ∇ρ  += x_w[i]*ρᵢ*temp
        ∇²ρ += (Hessian_correct_GM ? x_w[i]*ρᵢ*( temp * temp') : x_w[i]*ρᵢ*( temp * temp' - inv_sqrt_xx_cov[i]'*inv_sqrt_xx_cov[i]))
    end

    return ρ, ∇ρ, ∇²ρ
end



function compute_logρ_gm(x_p, x_w, x_mean, inv_sqrt_xx_cov, Hessian_correct_GM)
    N_modes, N_ens, N_x = size(x_p)
    logρ = zeros(N_modes, N_ens)
    ∇logρ = zeros(N_modes, N_ens, N_x)
    ∇²logρ = zeros(N_modes, N_ens, N_x, N_x)
    for im = 1:N_modes
        for i = 1:N_ens
            ρ, ∇ρ, ∇²ρ = Gaussian_mixture_density_derivatives(x_w, x_mean, inv_sqrt_xx_cov, x_p[im, i, :], Hessian_correct_GM)
            
            logρ[im, i]         =   log(  ρ  ) - N_x/2.0 * log(2π)
            ∇logρ[im, i, :]     =   ∇ρ/ρ
            ∇²logρ[im, i, :, :] =  (Hessian_correct_GM ? ∇²ρ/ρ - (∇ρ/ρ^2)*∇ρ' - inv_sqrt_xx_cov[im]'*inv_sqrt_xx_cov[im] : (∇²ρ/ρ - (∇ρ/ρ^2)*∇ρ'))
        end
    end

    return logρ, ∇logρ, ∇²logρ
end




function compute_logρ_gm_expectation(x_w, x_mean, sqrt_xx_cov, inv_sqrt_xx_cov, c_weights_GM, mean_weights_GM, N_ens_GM, Hessian_correct_GM)
    x_w = x_w / sum(x_w)
    N_modes, N_x = size(x_mean)
    if c_weights_GM !== nothing
        N_ens_GM =  size(c_weights_GM, 2)
    end
    xs = zeros(N_modes, N_ens_GM, N_x)
    logρ_mean, ∇logρ_mean, ∇²logρ_mean = zeros(N_modes), zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)

    for im = 1:N_modes
        xs[im,:,:] = construct_ensemble(x_mean[im, :], sqrt_xx_cov[im]; c_weights = c_weights_GM, N_ens = N_ens_GM)
    end
    
    logρ, ∇logρ, ∇²logρ = compute_logρ_gm(xs, x_w, x_mean, inv_sqrt_xx_cov, Hessian_correct_GM)
   
    for im = 1:N_modes
        logρ_mean[im], ∇logρ_mean[im,:], ∇²logρ_mean[im,:,:] = compute_expectation(logρ[im,:], ∇logρ[im,:,:], ∇²logρ[im,:,:,:], mean_weights_GM)
    end

    return  logρ_mean, ∇logρ_mean, ∇²logρ_mean
end
   




"""
update gmgd struct
ens_func: The function g = G(x)
define the function as 
    ens_func(x_ens) = MyG(phys_params, x_ens, other_params)
use G(x_mean) instead of FG(x)
"""
function update_ensemble!(gmgd::GMGDObj{FT, IT}, func::Function, dt_max::FT) where {FT<:AbstractFloat, IT<:Int}
    
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


    
    ########### function evaluation, it can be F or Φᵣ
    V, ∇V, ∇²V = func(x_p)

    ###########  Potential term
    Φᵣ_mean, ∇Φᵣ_mean, ∇²Φᵣ_mean = zeros(N_modes), zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)
    for im = 1:N_modes
        Φᵣ_mean[im], ∇Φᵣ_mean[im,:], ∇²Φᵣ_mean[im,:,:] = gmgd.Bayesian_inverse_problem ? 
        compute_expectation_BIP(x_mean[im,:], inv_sqrt_xx_cov[im], V[im,:,:], gmgd.c_weight_BIP) : 
        compute_expectation(V[im,:], ∇V[im,:,:], ∇²V[im,:,:,:], gmgd.mean_weights) 
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

            xx_cov_n[im, :, :] =   inv( Hermitian(inv(xx_cov[im, :, :]) + dt*(∇²logρ_mean[im, :, :] + ∇²Φᵣ_mean[im, :, :]) ))
            
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
        x_mean_n[im, :]  =  x_mean[im, :] - dt*xx_cov_n[im, :, :]*(∇logρ_mean[im, :] + ∇Φᵣ_mean[im, :]) 
    end


    ########## update weights
    for im = 1:N_modes
        dt = dt_max

        # update weights
        # dlogwₖ = ∫(ρᴳᴹ - Nₖ)(logρᴳᴹ + Φᵣ)dθ
        # logwₖ +=  -Δt∫Nₖ(logρᴳᴹ + Φᵣ)dθ + Δt ∫ρᴳᴹ(logρᴳᴹ + Φᵣ)dθ
        # the second term is independent of k, it is a normalization term
        logx_w_n[im] = logx_w[im] - dt*(logρ_mean[im] + Φᵣ_mean[im])
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
    push!(gmgd.phi_r_pred, Φᵣ_mean)
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


function ensemble_BIP(x_ens, forward, N_f)
    N_modes, N_ens, N_x = size(x_ens)
    F = zeros(N_modes, N_ens, N_f)   
    Threads.@threads for i = 1:N_ens
        for im = 1:N_modes
            F[im, i, :] = forward(x_ens[im, i, :])
        end
    end
    
    return F, nothing, nothing
end

function GMGD_Run(
    forward::Function, 
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
    # setup for potential function part
    Bayesian_inverse_problem::Bool = false,
    N_f::IT = 1,
    quadrature_type = "unscented_transform",
    c_weight_BIP::FT = sqrt(3.0),
    N_ens::IT = -1,
    w_min::FT = 1.0e-15) where {FT<:AbstractFloat, IT<:Int}
    

    gmgdobj = GMGDObj(# initial condition
        x0_w, x0_mean, xx0_cov;
        update_covariance = update_covariance,
        sqrt_matrix_type = sqrt_matrix_type,
        # setup for Gaussian mixture part
        quadrature_type_GM = quadrature_type_GM,
        c_weight_GM = c_weight_GM,
        N_ens_GM = N_ens_GM,
        Hessian_correct_GM = Hessian_correct_GM,
        # setup for potential function part
        Bayesian_inverse_problem = Bayesian_inverse_problem,
        N_f = N_f,
        quadrature_type = quadrature_type,
        c_weight_BIP = c_weight_BIP,
        N_ens = N_ens,
        w_min = w_min) 

    func(x_ens) = Bayesian_inverse_problem ? ensemble_BIP(x_ens, forward, N_f) : ensemble(x_ens, forward)  
    
    dt = T/N_iter
    for i in 1:N_iter
        if i%max(1,div(N_iter, 10)) == 0  @info "iter = ", i, " / ", N_iter  end
        
        update_ensemble!(gmgdobj, func, dt) 
    end
    
    return gmgdobj
    
end



###### Plot function 

function Gaussian_density_1d(x_mean::Array{FT,1}, inv_sqrt_xx_cov, xx) where {FT<:AbstractFloat}
    dx = [xx[:]' ;] - repeat(x_mean, 1, length(xx))
    return exp.( -1/2*(dx .* (inv_sqrt_xx_cov'*(inv_sqrt_xx_cov*dx)))) .* abs(det(inv_sqrt_xx_cov))
end

function Gaussian_mixture_1d(x_w, x_mean, xx_cov,  xx)
    
    N_modes = length(x_w)
    inv_sqrt_xx_cov = [compute_sqrt_matrix(xx_cov[im,:,:]; type="Cholesky")[2] for im = 1:N_modes]
    
    # 1d Gaussian plot
    dx = xx[2] - xx[1]
    N_x = length(xx)
    y = zeros(N_x)
    
    for im = 1:N_modes
        y .+= x_w[im]*Gaussian_density_1d(x_mean[im,:], inv_sqrt_xx_cov[im], xx)'
    end

    y = y/(sum(y)*dx)
    
    return y 
end


function posterior_BIP_1d(func_F, xx)
    dx = xx[2] - xx[1]
    N_x = length(xx)
    y = zeros(N_x)
    for i = 1:N_x
        F = func_F([xx[i];])
        y[i] = exp(-F'*F/2)
    end
    y /= (sum(y)*dx) 

    return y
end



function posterior_1d(func_V, xx)
    dx = xx[2] - xx[1]
    N_x = length(xx)
    y = zeros(N_x)
    for i = 1:N_x
        V = func_V([xx[i];])
        y[i] = exp(-V)
    end
    y /= (sum(y)*dx) 

    return y
end
    


function visualization_1d(ax; Nx=2000, x_lim=[-4.0,4.0], func_F = nothing, func_V = nothing, objs=nothing)

    # visualization 
    x_min, x_max = x_lim
    
    xx = LinRange(x_min, x_max, Nx)
    dx = xx[2] - xx[1] 
    
    yy_ref = (func_V === nothing ? posterior_BIP_1d(func_F, xx) : posterior_1d(func_V, xx))
    color_lim = (minimum(yy_ref), maximum(yy_ref))
    
    ax[1].plot(xx, yy_ref, "--", label="Reference", color="grey", linewidth=2, fillstyle="none", markevery=25)
    ax[2].plot(xx, yy_ref, "--", label="Reference", color="grey", linewidth=2, fillstyle="none", markevery=25)
           
   
    N_obj = length(objs)
    
    N_iter = length(objs[1].logx_w) - 1
    error = zeros(N_obj, N_iter+1)
        
    for (iobj, obj) in enumerate(objs)
        for iter = 0:N_iter  
            x_w = exp.(obj.logx_w[iter+1]); x_w /= sum(x_w)
            x_mean = obj.x_mean[iter+1]
            xx_cov = obj.xx_cov[iter+1]
            yy = Gaussian_mixture_1d(x_w, x_mean, xx_cov,  xx)
            error[iobj, iter+1] = norm(yy - yy_ref,1)*dx
            
            if iter == N_iter
                ax[iobj].plot(xx, yy, "--", label="Reference", color="red", linewidth=2, fillstyle="none", markevery=25)
                N_modes = size(x_mean, 1)
                

                ax[iobj].scatter(obj.x_mean[1], exp.(obj.logx_w[1]), marker="x", color="grey") 
                ax[iobj].scatter(x_mean, x_w, marker="o", color="red", facecolors="none")

            end
        end
        
    end
    for i_obj = 1:N_obj
        ax[N_obj+1].semilogy(Array(0:N_iter), error[i_obj, :], label=objs[i_obj].name*" (K="*string(size(objs[i_obj].x_mean[1], 1))*")")
    end
    ax[N_obj+1].legend()
end



function Gaussian_density_2d(x_mean::Array{FT,1}, inv_sqrt_xx_cov, X, Y) where {FT<:AbstractFloat}
    dx = [X[:]' ; Y[:]'] - repeat(x_mean, 1, length(X))

    return reshape( exp.( -1/2*sum(dx .* ((inv_sqrt_xx_cov'*inv_sqrt_xx_cov)*dx), dims=1)) .* abs(det(inv_sqrt_xx_cov)), size(X))


end
function Gaussian_mixture_2d(x_w, x_mean, xx_cov,  X, Y)
    N_modes = length(x_w)
    inv_sqrt_xx_cov = [compute_sqrt_matrix(xx_cov[im,:,:]; type="Cholesky")[2] for im = 1:N_modes]
    # 2d Gaussian plot
    dx, dy = X[2,1] - X[1,1], Y[1,2] - Y[1,1]
    N_x, N_y = size(X)
    Z = zeros(N_x, N_y)
    
    
    for im = 1:N_modes
        Z .+= x_w[im]*Gaussian_density_2d(x_mean[im,:], inv_sqrt_xx_cov[im], X, Y)
    end

    Z = Z/(sum(Z)*dx*dy)
    
    return Z
    
end


function posterior_BIP_2d(func_F, X, Y)
    dx, dy = X[2,1] - X[1,1], Y[1,2] - Y[1,1]
    N_x, N_y = size(X)
    Z = zeros(N_x, N_y)
    for i = 1:N_x
        for j = 1:N_y
            F = func_F([X[i,j] ; Y[i,j]])
            Z[i,j] = exp(-F'*F/2)
        end
    end
    Z /= (sum(Z)*dx*dy) 

    return Z
end


function posterior_2d(func_V, X, Y)
    dx, dy = X[2,1] - X[1,1], Y[1,2] - Y[1,1]
    N_x, N_y = size(X)
    Z = zeros(N_x, N_y)
    for i = 1:N_x
        for j = 1:N_y
            V = func_V([X[i,j] ; Y[i,j]])
            Z[i,j] = exp(-V)
        end
    end
    Z /= (sum(Z)*dx*dy) 

    return Z
end



function visualization_2d(ax; Nx=2000, Ny=2000, x_lim=[-4.0,4.0], y_lim=[-4.0,4.0], func_F = nothing, func_V = nothing, objs=nothing)

    # visualization 
    x_min, x_max = x_lim
    y_min, y_max = y_lim

    xx = LinRange(x_min, x_max, Nx)
    yy = LinRange(y_min, y_max, Ny)
    dx, dy = xx[2] - xx[1], yy[2] - yy[1]
    X,Y = repeat(xx, 1, Ny), repeat(yy, 1, Nx)'   #'

    Z_ref = (func_V === nothing ? posterior_BIP_2d(func_F, X, Y) : posterior_2d(func_V, X, Y))
    color_lim = (minimum(Z_ref), maximum(Z_ref))
    ax[1].pcolormesh(X, Y, Z_ref, cmap="viridis", clim=color_lim)


   
    N_obj = length(objs)
    
    N_iter = length(objs[1].logx_w) - 1
    error = zeros(N_obj, N_iter+1)
        
    for (iobj, obj) in enumerate(objs)
        for iter = 0:N_iter  
            x_w = exp.(obj.logx_w[iter+1]); x_w /= sum(x_w)
            x_mean = obj.x_mean[iter+1]
            xx_cov = obj.xx_cov[iter+1]
            Z = Gaussian_mixture_2d(x_w, x_mean, xx_cov,  X, Y)
            error[iobj, iter+1] = norm(Z - Z_ref,1)*dx*dy
            
            if iter == N_iter
                ax[1+iobj].pcolormesh(X, Y, Z, cmap="viridis", clim=color_lim)
                N_modes = size(x_mean, 1)
                

                ax[1+iobj].scatter([obj.x_mean[1][:,1];], [obj.x_mean[1][:,2];], marker="x", color="grey") 
                ax[1+iobj].scatter([x_mean[:,1];], [x_mean[:,2];], marker="o", color="red", facecolors="none")

            end
        end
        
    end
    for i_obj = 1:N_obj
        ax[N_obj+2].semilogy(Array(0:N_iter), error[i_obj, :], label=objs[i_obj].name*" (K="*string(size(objs[i_obj].x_mean[1], 1))*")")
    end
    ax[N_obj+2].legend()
end




