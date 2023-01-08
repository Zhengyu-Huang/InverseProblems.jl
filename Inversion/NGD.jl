using LinearAlgebra
using Random
using Distributions
using EllipsisNotation
using PyPlot
"""
GDObj{FT<:AbstractFloat, IT<:Int}
Struct that is used in Natural gradient descent (NGD)
For solving the inverse problem 
    y = G(θ) + η
    
"""

mutable struct GDObj{FT<:AbstractFloat, IT<:Int}
    "a vector of arrays of size N_ensemble x N_parameters containing the mean of the parameters (in each gd iteration a new array of mean is added)"
    θ_mean::Vector{Array{FT, 1}}
    "a vector of arrays of size N_ensemble x (N_parameters x N_parameters) containing the covariance of the parameters (in each gd iteration a new array of cov is added)"
    θθ_cov::Vector{Array{FT, 2}}
    "number ensemble size"
    N_ens::IT
    "size of θ"
    N_θ::IT
    "sampling method"
    sampling_method::String
    "weights"
    c_weights::Union{Array{FT, 1}, Array{FT, 2}}
    mean_weights::Array{FT, 1}
    cov_weights::Array{FT, 1}
    "Initial time step"
    Δt::FT
    "current iteration number"
    iter::IT
    "compute gradient flag"
    compute_gradient::String
    "gradient flow"
    gradient_flow::String
end



"""
GDObj Constructor 
θ0_mean::Array{FT} : inital mean
θθ0_cov::Array{FT, 2} : initial covariance
sampling_method : "MonteCarlo", "UnscentedTransform"
"""
function GDObj(θ0_mean::Array{FT}, 
                θθ0_cov::Array{FT, 2},
                sampling_method::String, 
                N_ens::IT,
                Δt::FT,
                compute_gradient::String,
                gradient_flow::String;
                ) where {FT<:AbstractFloat, IT<:Int}

    
    N_θ = size(θ0_mean,1)


    if sampling_method == "MonteCarlo"
        # ensemble size
        c_weights = zeros(FT, N_θ, N_ens)
        mean_weights = zeros(FT, N_ens)
        cov_weights = zeros(FT, N_ens)

        Random.seed!(42)
        c_weights    .=  rand(Normal(0, 1), N_θ, N_ens)
        # shift mean to and covariance to I
        c_weights -= dropdims(mean(c_weights, dims=2), dims=2)*ones(N_ens)'
        U1, S1, _ = svd(c_weights)
        c_weights = (S1/sqrt(N_ens - 1.0) .\U1') *  c_weights 
   
        mean_weights .=  1/N_ens 
        cov_weights  .=  1/(N_ens - 1) 
        
    elseif sampling_method == "UnscentedTransform"

        N_ens = 2N_θ+1
        c_weights = zeros(FT, N_θ, N_ens)
        mean_weights = zeros(FT, N_ens)
        cov_weights = zeros(FT, N_ens)

        κ = 0.0
        β = 2.0
        α = min(sqrt(4/(N_θ + κ)), 1.0)
        # α = 1e-2
        λ = α^2*(N_θ + κ) - N_θ

        for i = 1:N_θ
            c_weights[i,i+1]         =   sqrt(N_θ + λ)
            c_weights[i,i+N_θ+1]     =  -sqrt(N_θ + λ)
        end
        mean_weights[1] = λ/(N_θ + λ)
        mean_weights[2:N_ens] .= 1/(2(N_θ + λ))
        cov_weights[1] = λ/(N_θ + λ) + 1 - α^2 + β
        cov_weights[2:N_ens] .= 1/(2(N_θ + λ))


    else

        error("sampling_method: ", sampling_method)
    
    end

    

    θ_mean = Array{FT,1}[]  # array of Array{FT, 2}'s
    push!(θ_mean, θ0_mean) # insert parameters at end of array (in this case just 1st entry)
    θθ_cov = Array{FT,2}[] # array of Array{FT, 2}'s
    push!(θθ_cov, θθ0_cov) # insert parameters at end of array (in this case just 1st entry)
    
    iter = 0

    GDObj{FT,IT}(θ_mean, θθ_cov,
                  N_ens, N_θ, 
                  sampling_method, c_weights, mean_weights, cov_weights, 
                  Δt, iter, compute_gradient, gradient_flow)

end


"""
construct_sigma_ensemble
Construct the sigma ensemble, based on the mean x_mean, and covariance x_cov
"""
function construct_sigma_ensemble(gd::GDObj{FT, IT}, x_mean::Array{FT,1}, x_cov::Array{FT,2}) where {FT<:AbstractFloat, IT<:Int}
    N_ens = gd.N_ens
    N_x = size(x_mean,1)

    c_weights = gd.c_weights
    chol_xx_cov = cholesky(Hermitian(x_cov)).L

    xs = zeros(FT, N_ens, N_x)
    xs .= (x_mean * ones(N_ens)' + chol_xx_cov * c_weights)'
    return xs
end


"""
construct_mean x_mean from ensemble x
"""
function construct_mean(gd::GDObj{FT, IT}, x::Array{FT}) where {FT<:AbstractFloat, IT<:Int}
    N_x = size(x)
    N_ens = N_x[1]
    @assert(gd.N_ens == N_ens)

    mean_weights = gd.mean_weights
    x_mean = zeros(N_x[2:end]...)

    for i = 1:N_ens
        x_mean .+= mean_weights[i] * x[i,..]
    end
    
    return x_mean
end

"""
construct_cov xx_cov from ensemble x and mean x_mean
"""
function construct_cov(gd::GDObj{FT, IT}, x::Array{FT,2}, x_mean::Array{FT}) where {FT<:AbstractFloat, IT<:Int}
    return construct_cov(gd, x, x_mean, x, x_mean)
end



"""
construct_cov xy_cov from ensemble x and mean x_mean, ensemble y and mean y_mean
"""
function construct_cov(gd::GDObj{FT, IT}, x::Array{FT},  y::Array{FT}) where {FT<:AbstractFloat, IT<:Int}
    N_x, N_y = size(x), size(y)
    mean_weights = gd.mean_weights
    x_mean, y_mean = zeros(N_x[2:end]...), zeros(N_y[2:end]...)
    for i = 1:N_ens
        x_mean .+= mean_weights[i] * x[i,..]
        y_mean .+= mean_weights[i] * y[i,..]
    end
    
    
    
    N_ens, N_x, N_y = gd.N_ens, size(x_mean), size(y_mean')
    cov_weights = gd.cov_weights
    xy_cov = zeros(FT, N_x..., N_y[2:end]...)
    for i = 1: N_ens
        xy_cov .+= cov_weights[i]*(x[i,..] - x_mean)*(y[i,..] - y_mean)'
    end

    return xy_cov
end


"""
construct_cov xy_cov from ensemble x and mean x_mean, ensemble y and mean y_mean
"""
function construct_cov(gd::GDObj{FT, IT}, x::Array{FT}, x_mean::Array{FT}, y::Array{FT}, y_mean::Array{FT}) where {FT<:AbstractFloat, IT<:Int}
    N_ens, N_x, N_y = gd.N_ens, size(x_mean), size(y_mean')
    
    cov_weights = gd.cov_weights

    xy_cov = zeros(FT, N_x..., N_y[2:end]...)
    for i = 1: N_ens
        xy_cov .+= cov_weights[i]*(x[i,..] - x_mean)*(y[i,..] - y_mean)'
    end

    return xy_cov
end

"""
construct_cov xy_cov from ensemble x and mean x_mean, ensemble y and mean y_mean
"""
function gd_cov(gd::GDObj{FT, IT}, θ::Array{FT,2}, θ_mean::Array{FT,1}, θθ_cov::Array{FT,2}, y::Array{FT, 2}) where {FT<:AbstractFloat, IT<:Int}
    N_ens, N_θ = gd.N_ens, size(θ_mean,1)
    mean_weights = gd.mean_weights

    θy_cov = zeros(FT, N_θ, N_θ)
    for i = 1: N_ens
        θy_cov .+= mean_weights[i]*((θ[i,:]-θ_mean)*(θ[i,:]-θ_mean)'- θθ_cov)*y[i,1]
    end

    return θy_cov
end



"""
update gd struct
ens_func: The function g = G(θ)
define the function as 
    ens_func(θ_ens) = MyG(phys_params, θ_ens, other_params)
use G(θ_mean) instead of FG(θ)
"""
function update_ensemble!(gd::GDObj{FT, IT}, ens_func::Function;) where {FT<:AbstractFloat, IT<:Int}
    Δt = gd.Δt
    gd.iter += 1
    # update evolution covariance matrix

    θ_mean  = gd.θ_mean[end]
    θθ_cov = gd.θθ_cov[end]
    N_θ,  N_ens = gd.N_θ, gd.N_ens
    ############# Prediction step:
    # @info "θθ_cov = ", θθ_cov,  inv(θθ_cov)
    # @info "eigen(θθ_cov) = ", eigen(θθ_cov)

    θ = construct_sigma_ensemble(gd, θ_mean, θθ_cov)


    ###########  Analysis step
    Φ = zeros(FT, N_ens, 1)
    ∇Φ = zeros(FT, N_ens, N_θ)
    ∇²Φ = zeros(FT, N_ens, N_θ, N_θ)
    
    Φ, ∇Φ, ∇²Φ = ens_func(θ)
    
    # δmean, δcov = zeros(N_θ), zeros(N_θ, N_θ)
    if gd.gradient_flow == "Fisher-Rao"
        if gd.compute_gradient == "second-order"
            θ_mean_n = θ_mean - θθ_cov * construct_mean(gd, ∇Φ) * Δt
            θθ_cov_n = θθ_cov + Δt*(θθ_cov - θθ_cov*construct_mean(gd, ∇²Φ)*θθ_cov)
            
        elseif gd.compute_gradient == "first-order"
            E∇Φ = construct_mean(gd, ∇Φ)
            θ_mean_n = θ_mean - θθ_cov * construct_mean(gd, ∇Φ) * Δt
            θθ_cov_n = θθ_cov + Δt*(θθ_cov - θθ_cov*construct_cov(gd, ∇Φ, E∇Φ, θ, θ_mean))
            
        else #gd.compute_gradient == "zeroth-order"
            EΦ = construct_mean(gd, Φ)
            θ_mean_n = θ_mean - construct_cov(gd, Φ, EΦ, θ, θ_mean)[:] * Δt
            
            # E∇Φ = construct_mean(gd, ∇Φ)
            # @info "A1 = ", construct_mean(gd, ∇²Φ), "A2 = ", θθ_cov\gd_cov(gd, θ, θ_mean, θθ_cov,  Φ, EΦ)/θθ_cov, " A3 = ", construct_cov(gd, ∇Φ, E∇Φ, θ, θ_mean)/θθ_cov
            
            θθ_cov_n = inv(inv(θθ_cov) + Δt/(1 + Δt)*(θθ_cov\gd_cov(gd, θ, θ_mean, θθ_cov,  Φ)/θθ_cov - inv(θθ_cov)))
            
        end
    elseif gd.gradient_flow == "Wasserstein"
        if gd.compute_gradient == "second-order"
            θ_mean_n = θ_mean - construct_mean(gd, ∇Φ) * Δt
            θθ_cov_n = θθ_cov + Δt*Hermitian(2I - construct_mean(gd, ∇²Φ)*θθ_cov - θθ_cov*construct_mean(gd, ∇²Φ))

            
        elseif gd.compute_gradient == "first-order"
            E∇Φ = construct_mean(gd, ∇Φ)
            θ_mean_n = θ_mean - construct_mean(gd, ∇Φ) * Δt
            
            temp = I - construct_cov(gd, ∇Φ, E∇Φ, θ, θ_mean)
            θθ_cov_n = θθ_cov + Δt*Hermitian(temp + temp')

            
        end


    elseif gd.gradient_flow == "Gradient_descent"
        if gd.compute_gradient  == "second-order"
            θ_mean_n = θ_mean - construct_mean(gd, ∇Φ) * Δt
            θθ_cov_n = θθ_cov + Δt/2*(inv(θθ_cov)  - construct_mean(gd, ∇²Φ))

#             S,V = eigen(θθ_cov_n)
#             S[S .< 0] .= 1e-3
#             θθ_cov_n = inv(V) * Diagonal(S) * V
            # @info "eigen = ", eigen(θθ_cov_n)
        elseif gd.compute_gradient == "first-order"
            E∇Φ = construct_mean(gd, ∇Φ)
            θ_mean_n = θ_mean - construct_mean(gd, ∇Φ) * Δt
            θθ_cov_n = θθ_cov + Δt/2*(I  - construct_cov(gd, ∇Φ, E∇Φ, θ, θ_mean))/θθ_cov

            
        else
            # EΦ = construct_mean(gd, Φ)
            # θ_mean_n = θ_mean - construct_cov(gd, Φ, EΦ, θ, θ_mean)[:] * Δt
            # θθ_cov_n = inv(inv(θθ_cov) + Δt/(1 + Δt)*(θθ_cov\gd_cov(gd, θ, θ_mean, θθ_cov,  Φ)/θθ_cov - inv(θθ_cov)))
            
        end
        
    else

    end
    
    # @info "θ_mean = ", θ_mean_n
    # @info "θθ_cov = ", θθ_cov_n


    ########### Save resutls
    push!(gd.θ_mean, θ_mean_n) # N_ens x N_params
    push!(gd.θθ_cov, θθ_cov_n) # N_ens x N_data
end


function ensemble_Φ(s_param, θ_ens::Array{FT,2}, Φ_func::Function)  where {FT<:AbstractFloat}
    
    N_ens,  N_θ = size(θ_ens)

    Φ = zeros(N_ens, 1)
    ∇Φ = zeros(N_ens, N_θ)
    ∇²Φ = zeros(N_ens, N_θ, N_θ)

    # Threads.@threads for i = 1:N_ens
    for i = 1:N_ens
        θ = θ_ens[i, :]
        Φ[i, 1], ∇Φ[i,:], ∇²Φ[i,:,:] = Φ_func(s_param, θ)
    end
    return  Φ, ∇Φ, ∇²Φ  
end


function NGD_Run(s_param, Φ_func::Function, 
    θ0_mean, 
    θθ0_cov,
    sampling_method,
    N_ens,
    Δt,
    N_iter,
    compute_gradient,
    gradient_flow
    )
    
    gdobj = GDObj(
    θ0_mean, 
    θθ0_cov,
    sampling_method,
    N_ens,
    Δt,
    compute_gradient,
    gradient_flow)

    
    ens_func(θ_ens) = ensemble_Φ(s_param, θ_ens, Φ_func)
    
    for i in 1:N_iter
        update_ensemble!(gdobj, ens_func) 
    end
    
    return gdobj
end

######################### TEST #######################################
#
# mutable struct Setup_Param{MAT, IT<:Int}
#     θ_names::Array{String,1}
#     G::MAT
#     N_θ::IT
#     N_y::IT
# end

# function Setup_Param(G, N_θ::IT, N_y::IT) where {IT<:Int}
#     return Setup_Param(["θ"], G, N_θ, N_y)
# end


# function forward(s_param::Setup_Param, θ::Array{FT, 1}) where {FT<:AbstractFloat}
#     G = s_param.G 
#     return G * θ
# end

# function forward_aug(s_param::Setup_Param, θ::Array{FT, 1}) where {FT<:AbstractFloat}
#     G = s_param.G 
#     return [G * θ; θ]
# end


# function compute_Φ(s_param::Setup_Param, θ::Array{FT, 1},  y::Array{FT, 1}, Σ_η::Array{FT, 2}, μ0::Array{FT, 1}, Σ0::Array{FT, 2}) where {FT<:AbstractFloat}
#     Φ   = 1/2*(y - G * θ)'*(Σ_η\(y - G * θ)) + 1/2*(μ0 - θ)'*(Σ0\(μ0 - θ))
#     ∇Φ  = -G' * (Σ_η\(y - G * θ)) - Σ0\(μ0 - θ)
#     ∇²Φ = G' * (Σ_η\G) + inv(Σ0)
#     return Φ, ∇Φ, ∇²Φ
# end



# function Two_Param_Linear_Test(problem_type::String, θ0_bar, θθ0_cov)
    
#     N_θ = length(θ0_bar)

    
#     if problem_type == "under-determined"
#         # under-determined case
#         θ_ref = [0.6, 1.2]
#         G = [1.0 2.0;]
        
#         y = [3.0;]
#         Σ_η = Array(Diagonal(fill(0.1^2, size(y))))
        
        
#     elseif problem_type == "well-determined"
#         # over-determined case
#         θ_ref = [1.0, 1.0]
#         G = [1.0 2.0; 3.0 4.0]
        
#         y = [3.0;7.0]
#         Σ_η = Array(Diagonal(fill(0.1^2, size(y))))
        
#     elseif problem_type == "over-determined"
#         # over-determined case
#         θ_ref = [1/3, 17/12.0]
#         G = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        
#         y = [3.0;7.0;10.0]
#         Σ_η = Array(Diagonal(fill(0.1^2, size(y))))
        
        
#     else
#         error("Problem type : ", problem_type, " has not implemented!")
#     end
    
#     Σ_post = inv(G'*(Σ_η\G) + inv(θθ0_cov))
#     θ_post = θ0_bar + Σ_post*(G'*(Σ_η\(y - G*θ0_bar)))
    

#     return θ_post, Σ_post, G, y, Σ_η, θ_ref
# end

# N_θ = 2
# θ0_mean = zeros(Float64, N_θ)
# θθ0_cov = Array(Diagonal(fill(1.0^2, N_θ)))
# θθ0_cov_sqrt = θθ0_cov

# prior_mean     = θ0_mean
# prior_cov      = θθ0_cov
# prior_cov_sqrt = θθ0_cov_sqrt

# problem_type = "well-determined"
# θ_post, Σ_post, G, y, Σ_η, θ_ref = Two_Param_Linear_Test(problem_type, θ0_mean, θθ0_cov)
    
# s_param = Setup_Param(G, N_θ, length(y)+N_θ)

# Φ_func(s_param, θ) = compute_Φ(s_param, θ,  y, Σ_η, θ0_mean, θθ0_cov) 

# sampling_method = "MonteCarlo"

# gradient_flow = "Gradient_descent"
# N_ens = 100
# Δt = 0.0001
# N_iter = 20000

# # Δt = 0.5
# # N_iter = 200
# compute_gradient = true
# gd_obj = NGD_Run(s_param, Φ_func, θ0_mean, θθ0_cov, sampling_method, N_ens,  Δt, N_iter, compute_gradient, gradient_flow)


# gd_errors    = zeros(N_iter+1, 2)
# for i = 1:N_iter+1
#     gd_errors[i, 1] = norm(gd_obj.θ_mean[i] .- θ_post)/norm(θ_post)
#     gd_errors[i, 2] = norm(gd_obj.θθ_cov[i] .- Σ_post)/norm(Σ_post)
# end

# ites = Array(0:N_iter)

# markevery = 5

# fig, ax = PyPlot.subplots(nrows = 1, ncols=2, sharex=false, sharey="row", figsize=(14,9))
# ax[1].semilogy(ites, gd_errors[:, 1],   "-.x", color = "C0", fillstyle="none", label="NGD", markevery = markevery)
# ax[1].set_xlabel("Iterations")
# ax[1].set_ylabel("Rel. mean error")
# ax[1].grid("on")
# ax[1].legend(bbox_to_anchor=(1.0, 1.0))


# ax[2].semilogy(ites, gd_errors[:, 2],   "-.x", color = "C0", fillstyle="none", label="NGD", markevery = markevery)
# ax[2].set_xlabel("Iterations")
# ax[2].set_ylabel("Rel. covariance error")
# ax[2].grid("on")
# ax[2].legend(bbox_to_anchor=(1.0, 1.0))
