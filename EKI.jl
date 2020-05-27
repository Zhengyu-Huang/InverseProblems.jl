

using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions


"""
    EKIObj{FT<:AbstractFloat, IT<:Int}
Structure that is used in Ensemble Kalman Inversion (EKI)
#Fields
$(DocStringExtensions.FIELDS)
"""
struct EKIObj{FT<:AbstractFloat, IT<:Int}
    "a vector of arrays of size N_ensemble x N_parameters containing the parameters (in each EKI iteration a new array of parameters is added)"
     u::Vector{Array{FT, 2}}
     "vector of parameter names"
     unames::Vector{String}
     "vector of observations (length: N_data); mean of all observation samples"
     g_t::Vector{FT}
     "covariance of the observational noise, which is assumed to be normally distributed"
     cov::Array{FT, 2}
     "ensemble size"
     N_ens::IT
     "vector of arrays of size N_ensemble x N_data containing the data G(u) (in each EKI iteration a new array of data is added)"
     g::Vector{Array{FT, 2}}
     "vector of errors"
     err::Vector{FT}
end

# outer constructors
function EKIObj(parameters::Array{FT, 2},
                parameter_names::Vector{String},
                t_mean,
                t_cov::Array{FT, 2}) where {FT<:AbstractFloat}

    # ensemble size
    N_ens = size(parameters)[1]
    IT = typeof(N_ens)
    # parameters
    u = Array{FT, 2}[] # array of Array{FT, 2}'s
    push!(u, parameters) # insert parameters at end of array (in this case just 1st entry)
    # observations
    g = Vector{FT}[]
    # error store
    err = []

    EKIObj{FT,IT}(u, parameter_names, t_mean, t_cov, N_ens, g, err)
end


"""
    construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
Construct the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_initial_ensemble(N_ens::IT, priors; rng_seed=42) where {IT<:Int}
    N_params = length(priors)
    params = zeros(N_ens, N_params)
    # Ensuring reproducibility of the sampled parameter values
    Random.seed!(rng_seed)
    for i in 1:N_params
        prior_i = priors[i]
        params[:, i] = rand(prior_i, N_ens)
    end

    return params
end

function compute_error(eki)
    meang = dropdims(mean(eki.g[end], dims=1), dims=1)
    diff = eki.g_t - meang
    X = eki.cov \ diff # diff: column vector
    newerr = dot(diff, X)
    push!(eki.err, newerr)
end


function update_ensemble!(eki::EKIObj{FT}, g) where {FT}
    # u: N_ens x N_params
    u = eki.u[end]

    u_bar = fill(FT(0), size(u)[2])
    # g: N_ens x N_data
    g_bar = fill(FT(0), size(g)[2])

    cov_ug = fill(FT(0), size(u)[2], size(g)[2])
    cov_gg = fill(FT(0), size(g)[2], size(g)[2])

    # update means/covs with new param/observation pairs u, g
    for j = 1:eki.N_ens

        u_ens = u[j, :]
        g_ens = g[j, :]

        # add to mean
        u_bar += u_ens
        g_bar += g_ens

        #add to cov
        cov_ug += u_ens * g_ens' # cov_ug is N_params x N_data
        cov_gg += g_ens * g_ens'
    end

    u_bar = u_bar / eki.N_ens
    g_bar = g_bar / eki.N_ens
    cov_ug = cov_ug / eki.N_ens - u_bar * g_bar'
    cov_gg = cov_gg / eki.N_ens - g_bar * g_bar'

    @show u_bar, g_bar
    @show cov_ug, cov_gg

    # update the parameters (with additive noise too)
    noise = rand(MvNormal(zeros(size(g)[2]), eki.cov), eki.N_ens) # N_data * N_ens
    y = (eki.g_t .+ noise)' # add g_t (N_data) to each column of noise (N_data x N_ens), then transp. into N_ens x N_data
    
    @show size(y), mean(y, dims=1)
    @show size(y-g), mean(y-g, dims=1)
    tmp = (cov_gg + eki.cov) \ (y - g)' # N_data x N_data \ [N_ens x N_data - N_ens x N_data]' --> tmp is N_data x N_ens
    
    @info cov_gg,  eki.cov, cov_gg + eki.cov
    @show size(tmp), mean(tmp, dims=2)
    u += (cov_ug * tmp)' # N_ens x N_params

    # store new parameters (and observations)
    push!(eki.u, u) # N_ens x N_params
    push!(eki.g, g) # N_ens x N_data

    compute_error(eki)

end




function dot_Γ(x,y, Γ) 
    return x'*(Γ\y)
end

function update_ensemble_eks!(eki::EKIObj{FT}, g) where {FT}
    # u: N_ens x N_params
    u = eki.u[end]
    u_prior = eki.u[1]
    N_ens, N_params = size(u)

    θs = vec([u[i,:] for i = 1:N_ens])
    fθs = vec([g[i,:] for i = 1:N_ens])
    prior = vec([u_prior[i,:] for i = 1:N_ens])

    obs = eki.g_t
    space = eki.cov



    covθ = cov(θs)
    meanθ = mean(θs)

    m = mean(fθs)

    J = length(θs)
    CG = [dot_Γ(fθk - m, fθj - obs, space)/J for fθj in fθs, fθk in fθs]

    Δt = FT(1) / (norm(CG) + sqrt(eps(FT)))

    implicit = lu( I + Δt .* (covθ * inv(cov(prior))) ) # todo: incorporate means
    Z = covθ * (cov(prior) \ mean(prior))

    noise = MvNormal(covθ)

    # θs .- Δt .* (CG*(θs .- Ref(meanθ))) .+ Δt .* Ref(covθ * (cov(prob.prior) \ mean(prob.prior)))

    # compute next set of θs
    map(enumerate(θs)) do (j, θj)
        X = sum(enumerate(θs)) do (k, θk)
            CG[j,k]*(θk-meanθ)
        end
        rhs = θj .- Δt .* X .+ Δt .* Z
        u[j,:] .= (implicit \ rhs) .+ sqrt(2*Δt)*rand(noise)
    end
    

    push!(eki.u, u) # N_ens x N_params
    push!(eki.g, g) # N_ens x N_data
    compute_error(eki)
end
