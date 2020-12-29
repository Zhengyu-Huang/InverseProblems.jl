using Random
using Distributions
using PyPlot
using LinearAlgebra
include("../Plot.jl")
include("../RExKI.jl")
include("../RWMCMC.jl")
# 
Nθ, Ny = 200, 150
Random.seed!(123);
c = 20.0
A = rand(Normal(0, 1), (Ny, Nθ))
B = rand(Normal(0, 1), (Ny, Nθ))

function forward(u::Array{Float64,1}, args)
    return A*u + sin.(c*B*u)
end


function ensemble(params_i::Array{Float64, 2})
    

    
    N_ens,  N_θ = size(params_i)
    
    g_ens = zeros(Float64, N_ens,  Ny)
    
    for i = 1:N_ens 
        # g: N_ens x N_data
        g_ens[i, :] .= forward(params_i[i, :], nothing)
    end
    
    return g_ens
end

function f_posterior(u::Array{Float64,1}, args, obs::Array{Float64,1}, obs_cov::Array{Float64,2}, μ0::Array{Float64,1}, cov0::Array{Float64,2})
    Gu = forward(u, args)
    
    Φ = - 0.5*(obs - Gu)'/obs_cov*(obs - Gu) - 0.5*(u - μ0)'/cov0*(u - μ0)
    
    return Φ
end



function ExKI(t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
    θ0_bar::Array{Float64,1}, θθ0_cov::Array{Float64,2}, 
    α_reg::Float64, N_iter, update_cov::Int64)
    
    
    parameter_names = ["u₁", "u₂"]
    
    ens_func(θ_ens) = ensemble(θ_ens)
    
    ukiobj = ExKIObj(parameter_names,
    θ0_bar, 
    θθ0_cov,
    t_mean, # observation
    t_cov,
    α_reg)

    θref = 2.0*ones(length(θ0_bar))
    
    for i in 1:N_iter
        
        update_ensemble!(ukiobj, ens_func) 
        
        # @info "g_bar = ", ukiobj.g_bar[i], "g_t = ", ukiobj.g_t
        @info "loss = ", (ukiobj.g_bar[i] - ukiobj.g_t)'/ukiobj.obs_cov*(ukiobj.g_bar[i] - ukiobj.g_t)
        @info "norm(θ) = ", norm(ukiobj.θ_bar[i])
        @info "relative error = ", norm(ukiobj.θ_bar[i] - θref)/norm(θref)
        @info "norm(θ_cov) = ", norm(ukiobj.θθ_cov[i])
        
        if (update_cov) > 0 && (i%update_cov == 0) 
            ukiobj.θθ_cov[1] = copy(ukiobj.θθ_cov[end])
        end
        
    end
    
    return ukiobj
end




function Sin_Rand_Posterior_Plot()
    
    # prior and covariance
    θ_ref = fill(2.0, Nθ)
    
    obs = forward(θ_ref, nothing)
    
    obs_cov = Array(Diagonal(fill(0.1^2, Ny))) 
    
    μ0 =  fill(0.0, Nθ)
    cov_sqr0  = Array(Diagonal(fill(2.0, Nθ))) 
    cov0 = cov_sqr0 * cov_sqr0 
    
    # compute posterior distribution by MCMC
    # f_density(u) = f_posterior(u, nothing, obs, obs_cov, μ0, cov0) 
    # step_length = 1.0
    # n_ite , n_burn_in= 1000000, 100000
    # us = RWMCMC(f_density, μ0, step_length, n_ite)
    for update_cov in [0]
        
        
        # compute posterior distribution the uki method 
        α_reg,  N_iter = 0.99, 500
        ukiobj = ExKI(obs, obs_cov,  μ0, cov0 , α_reg,  N_iter, update_cov)
        
        
        xx = Array(1:Nθ)
        uki_θ_bar  = ukiobj.θ_bar[end]
        uki_θθ_cov = ukiobj.θθ_cov[end]
        uki_θθ_std = sqrt.(diag(ukiobj.θθ_cov[end]))
        
        fig, ax = PyPlot.subplots(figsize=(18,6))
        ax.plot(xx, uki_θ_bar)
        # ax.plot(xx, uki_θ_bar + 3.0*uki_θθ_std)
        # ax.plot(xx, uki_θ_bar - 3.0*uki_θθ_std)
        
        # plot MCMC results 
        
        
        
        fig.savefig("Sin_Rand_UKI_update_cov"*string(update_cov)*".png")
        close("all")
    end
    
end


Sin_Rand_Posterior_Plot()