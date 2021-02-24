using Random
using Distributions


function log_bayesian_posterior(s_param, θ::Array{Float64,1}, forward::Function, 
    y::Array{Float64,1},  Σ_η::Array{Float64,2}, 
    μ0::Array{Float64,1}, Σ0::Array{Float64,2})

    Gu = forward(s_param, θ)
    Φ = - 0.5*(y - Gu)'/Σ_η*(y - Gu) - 0.5*(θ - μ0)'/Σ0*(θ - μ0)
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
        θs[i, :] = (rand(Bernoulli(α)) ? θ : θ_p)
        fs[i] = (rand(Bernoulli(α)) ? fs[i] : fs[i-1])
    end
    
    return θs
    
end


