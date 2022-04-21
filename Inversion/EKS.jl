function ensemble(s_param, θ_ens::Array{FT,2}, forward::Function)  where {FT<:AbstractFloat}
    
    N_ens,  N_θ = size(θ_ens)
    N_y = s_param.N_y
    g_ens = zeros(FT, N_ens,  N_y)
    
    Threads.@threads for i = 1:N_ens
        θ = θ_ens[i, :]
        g_ens[i, :] .= forward(s_param, θ)
    end
    
    return g_ens
end

"""
EKSObj{FT<:AbstractFloat, IT<:Int}
Structure that is used in Ensemble Kalman Sampler (EKS)
#Fields
$(DocStringExtensions.FIELDS)
"""
mutable struct EKSObj{FT<:AbstractFloat, IT<:Int}
    "vector of parameter names"
    θ_names::Array{String, 1}
    "a vector of arrays of size N_ensemble x N_parameters containing the parameters (in each EKS iteration a new array of parameters is added)"
    θ::Vector{Array{FT, 2}}
    θ0_mean::Array{FT, 1}
    θθ0_cov::Array{FT, 2}
    "a vector of arrays of size N_ensemble x N_y containing the predicted observation (in each uki iteration a new array of predicted observation is added)"
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
    "time step"
    Δt::Vector{FT}
end


# outer constructors
function EKSObj(
    θ_names::Array{String, 1},
    N_ens::IT,
    θ0_mean::Array{FT},
    θθ0_cov::Array{FT,2},
    y::Array{FT, 1},
    Σ_η) where {FT<:AbstractFloat, IT<:Int}
    
    
    N_θ = size(θ0_mean,1)
    N_y = size(y, 1)
    
    
    # generate initial assemble
    θ = Array{FT, 2}[] # array of Array{FT, 2}'s
    θ0 = Array(rand(MvNormal(θ0_mean, θθ0_cov), N_ens)')
    push!(θ, θ0) # insert parameters at end of array (in this case just 1st entry)
    
    # prediction
    y_pred = Array{FT, 1}[]  # array of Array{FT, 2}'s
    Δt = Array{FT}[]
    
    iter = 0

    EKSObj{FT,IT}(
    θ_names, θ, θ0_mean, θθ0_cov,
    y_pred, 
    y, Σ_η, 
    N_ens, N_θ, N_y, 
    iter, Δt)
end


function EKS_Run(s_param, forward::Function, 
    θ0_mean, θθ0_cov,
    N_ens,
    y, Σ_η,
    N_iter)
    
    θ_names = s_param.θ_names
    
    eksobj = EKSObj(
    θ_names,
    N_ens,
    θ0_mean,
    θθ0_cov,
    y,
    Σ_η)
    
    
    ens_func(θ_ens) = ensemble(s_param, θ_ens, forward) 
    
    
    for i in 1:N_iter
        update_ensemble!(eksobj, ens_func) 
    end
    
    return eksobj
    
end

"""
construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
Construct the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_cov(eks::EKSObj{FT}, x::Array{FT,2}, x_mean::Array{FT, 1}, y::Array{FT,2}, y_mean::Array{FT, 1}) where {FT<:AbstractFloat}
    N_ens, N_x, N_y = eks.N_ens, size(x_mean,1), size(y_mean,1)
    
    xy_cov = zeros(FT, N_x, N_y)
    
    for i = 1: N_ens
        xy_cov .+= (x[i,:] - x_mean)*(y[i,:] - y_mean)'
    end
    
    return xy_cov/(N_ens - 1)
end








function update_ensemble!(eks::EKSObj{FT}, ens_func::Function) where {FT<:AbstractFloat}
    

    # u: N_ens × N_par
    # g: N_ens × N_obs

    prior_mean = eks.θ0_mean
    prior_cov = eks.θθ0_cov
    
    N_ens, N_θ, N_y = eks.N_ens, eks.N_θ, eks.N_y

    u = eks.θ[end]
    g = zeros(FT, N_ens, N_y)
    g .= ens_func(u)

   
    # u_mean: N_par × 1
    u_mean = mean(u', dims=2)
    # g_mean: N_obs × 1
    g_mean = mean(g', dims=2)
    # g_cov: N_obs × N_obs
    g_cov = cov(g, corrected=false)
    # u_cov: N_par × N_par
    u_cov = cov(u, corrected=false)

    # Building tmp matrices for EKS update:
    E = g' .- g_mean
    R = g' .- eks.y
    # D: N_ens × N_ens
    D = (1/eks.N_ens) * (E' * (eks.Σ_η \ R))

    
    Δt = 1/(norm(D) + 1e-8)
    
    
    # @info norm(u_cov)
    noise = MvNormal(Matrix(Hermitian(u_cov)))

    # @info "u mean: ", dropdims(mean(u, dims=1), dims=1)

    # the paper has prior_mean, now the prior is added
    implicit = (1 * Matrix(I, size(u)[2], size(u)[2]) + Δt * (prior_cov' \ u_cov')') \
                  (u'
                    .- Δt * ( u' .- u_mean) * D
                    .+ Δt * u_cov * (prior_cov \ prior_mean)
                    # TODO correction
                    .+ Δt * (length(u_mean) + 1)/N_ens * (u' .- u_mean)
                  )

    # @info "implicit mean: ", dropdims(mean(implicit', dims=1), dims=1)

    u = implicit' + sqrt(2*Δt) * rand(noise, eks.N_ens)'
    # @info "new u mean: ", dropdims(mean(rand(noise, eks.N_ens)', dims=1), dims=1)

    # @info sqrt(2*Δt),  "new u mean: ", sqrt(2*Δt) * dropdims(mean(rand(noise, eks.N_ens)', dims=1), dims=1)


    
    
    # Save results
    push!(eks.θ, u) # N_ens x N_θ
    push!(eks.y_pred, g)

    eks.iter += 1
    push!(eks.Δt, Δt)
    
end





### Test
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


# function Two_Param_Linear_Test(problem_type::String, θ0_bar, θθ0_cov)
    
#     N_θ = length(θ0_bar)

    
#     if problem_type == "under-determined"
#         # under-determined case
#         θ_ref = [0.6, 1.2]
#         G = [1.0 2.0;]
        
#         y = [3.0;]
#         Σ_η = Array(Diagonal(fill(0.1^2, size(y))))
        
        
    
#     elseif problem_type == "over-determined"
#         # over-determined case
#         θ_ref = [1/3, 8.5/6]
#         G = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        
#         y = [3.0;7.0;10.0]
#         Σ_η = Array(Diagonal(fill(0.1^2, size(y))))
        
        
#     else
#         error("Problem type : ", problem_type, " has not implemented!")
#     end
    
#     Σ_post = inv(G'*(Σ_η\G) + inv(θθ0_cov))
#     θ_post = θ0_bar + Σ_post*(G'*(Σ_η\(y - G*θ0_bar)))
    

#     return θ_post, Σ_post, G, y, Σ_η
# end


# function construct_cov(x::Array{FT,2}) where {FT<:AbstractFloat}
    
#     x_mean = dropdims(mean(x, dims=1), dims=1)
#     N_ens, N_x = size(x)
    
    
#     x_cov = zeros(FT, N_x, N_x)
    
#     for i = 1: N_ens
#         x_cov .+= (x[i,:] - x_mean)*(x[i,:] - x_mean)'
#     end
    
#     return x_cov/(N_ens - 1)
# end

# problem_type  = "over-determined"
# FT = Float64
# uki_objs = Dict()
# mean_errors = Dict()


# α_reg = 1.0
# update_freq = 1
# N_iter = 20
# N_θ = 2
# θ0_mean = zeros(FT, N_θ)
# θθ0_cov = Array(Diagonal(fill(1.0^2, N_θ)))
# θθ0_cov_sqrt = θθ0_cov


# θ_post, Σ_post, G, y, Σ_η = Two_Param_Linear_Test(problem_type, θ0_mean, θθ0_cov)
    
# N_y = length(y)

# s_param = Setup_Param(G, N_θ, N_y)
# s_param_aug = Setup_Param(G, N_θ, N_y+N_θ)



# eks_errors = zeros(FT, N_iter+1, 2)

# for i = 1:N_iter+1
#     eks_errors[i, 1] = norm(dropdims(mean(eks_obj.θ[i], dims=1), dims=1) .- θ_post)
#     eks_errors[i, 2] = norm(construct_cov(eks_obj.θ[i]) .- Σ_post)
    
# end


# semilogy(eks_errors[:, 1])
# semilogy(eks_errors[:, 2])