using Random
using Distributions

"""
When the density function is Φ/Z, 
The f_density function return log(Φ) instead of Φ
"""
function RWMCMC(f_log_density::Function, u0::Array{Float64,1}, step_length::Float64, n_ite::Int64)
    
    n = length(u0)
    us = zeros(Float64, n_ite, n)
    fs = zeros(Float64, n_ite)
    
    us[1, :] .= u0
    fs[1] = f_log_density(u0)
    
    Random.seed!(11)
    for i = 2:n_ite
        u_p = us[i-1, :] 
        u = u_p + step_length * rand(Normal(0, 1), n)
        
        
        fs[i] = f_log_density(u)
        α = min(1.0, exp(fs[i] - fs[i-1]))
        us[i, :] = (rand(Bernoulli(α)) ? u : u_p)
        fs[i] = (rand(Bernoulli(α)) ? fs[i] : fs[i-1])
    end
    
    return us
    
end


