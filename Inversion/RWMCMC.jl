using Random
using Distributions


function log_bayesian_posterior(s_param, θ::Array{Float64,1}, forward::Function, 
    y::Array{Float64,1},  Σ_η::Array{Float64,2}, 
    μ0::Array{Float64,1}, Σ0::Array{Float64,2})

    Gu = forward(s_param, θ)
    Φ = - 0.5*(y - Gu)'/Σ_η*(y - Gu) - 0.5*(θ - μ0)'/Σ0*(θ - μ0)
    return Φ

end

function log_likelihood(s_param, θ::Array{Float64,1}, forward::Function, 
    y::Array{Float64,1},  Σ_η::Array{Float64,2})

    Gu = forward(s_param, θ)
    Φ = - 0.5*(y - Gu)'/Σ_η*(y - Gu) 
    return Φ

end


"""
When the density function is Φ/Z, 
The f_density function return log(Φ) instead of Φ
"""
function RWMCMC_Run(log_bayesian_posterior::Function, θ0::Array{FT,1}, step_length::FT, n_ite::IT; seed::IT=11) where {FT<:AbstractFloat, IT<:Int}
    
    N_θ = length(θ0)
    θs = zeros(Float64, n_ite, N_θ)
    fs = zeros(Float64, n_ite)
    
    θs[1, :] .= θ0
    fs[1] = log_bayesian_posterior(θ0)
    
    Random.seed!(seed)
    for i = 2:n_ite
        θ_p = θs[i-1, :] 
        θ = θ_p + step_length * rand(Normal(0, 1), N_θ)
        
        
        fs[i] = log_bayesian_posterior(θ)
        α = min(1.0, exp(fs[i] - fs[i-1]))
        if α > rand(Uniform(0, 1))
            # accept
            θs[i, :] = θ
            fs[i] = fs[i]
            @info i
        else
            # reject
            θs[i, :] = θ_p
            fs[i] = fs[i-1]
        end

    end
    
    return θs 
end



# θ_s: N_ens/2 by N_θ matrix, the set
# θ_c: N_ens/2 by N_θ matrix, the complementary set
function emcee_Propose(θ_s::Array{FT,2}, θ_c::Array{FT,2}; a::FT = 2.0) where {FT<:AbstractFloat, IT<:Int}
    Ns, N_θ = size(θ_s)
        
    zz = ((a - 1.0) * rand(Uniform(0, 1), Ns) .+ 1).^2.0 / a
    factors = (N_θ - 1.0) * log.(zz)

    rint = rand(1:Ns, Ns,)
    return θ_c[rint, :] - (θ_c[rint, :] - θ_s) .* zz, factors

end

"""
When the density function is Φ/Z, 
The log_bayesian_posterior function return log(Φ) instead of Φ
θ0 : initial ensemble of size N_ens by N_θ
"""
function emcee_Run(log_bayesian_posterior::Function, θ0::Array{FT,2}, n_ite::IT; random_split::Bool = true, a::FT = 2.0, seed::IT=11) where {FT<:AbstractFloat, IT<:Int}
    Random.seed!(seed)

    N_ens, N_θ = size(θ0)
    @assert(N_ens >= 2N_θ)
    @assert(N_ens % 2 == 0)

    θs = zeros(Float64, n_ite, N_ens, N_θ)
    fs = zeros(Float64, n_ite, N_ens)
    
    θs[1, :, :] .= θ0
    for k = 1:N_ens
        fs[1, k] = log_bayesian_posterior(θ0[k, :])
    end

    nsplit = 2
    N_s = div(N_ens, 2)
    
    all_inds = Array(1:N_ens)
    inds = all_inds .% nsplit # 2 group
    log_probs = zeros(Float64, N_s)
    for i_t = 2:n_ite

        if random_split
            shuffle!(inds)
        end
        for split = 0:1
            # boolean array of the current set
            s_inds = (inds .== split)
            # boolean array for the complementary set
            c_inds = (inds .!= split)


            s, c = θs[i_t - 1, s_inds, :], θs[i_t - 1, c_inds, :]
             
            q, factors = emcee_Propose(s, c; a = a)
            for i = 1:N_s
                log_probs[i] = log_bayesian_posterior(q[i, :])
            end

            # Loop over the walkers and update them accordingly.
            for i = 1:N_s
                j = all_inds[s_inds][i]

                # j is the index
                # @info i, factors[i] , log_probs[i] , fs[i_t - 1, j]
                
                α = min(1.0, exp(factors[i] + log_probs[i] - fs[i_t - 1, j]))
                if α > rand(Uniform(0, 1))
                    # accept
                    θs[i_t, j, :] = q[i, :]
                    fs[i_t, j] = log_probs[i]
                else
                    # reject
                    θs[i_t, j, :] = θs[i_t - 1, j, :]
                    fs[i_t, j] = fs[i_t - 1, j]
                end
            end
        end
    end
    
    return θs
end






