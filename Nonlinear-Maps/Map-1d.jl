using Random
using Distributions
using PyPlot
using LinearAlgebra
include("../Plot.jl")
include("../RUKI.jl")
include("../RExKI.jl")
include("../RWMCMC.jl")
include("../SMC.jl")



function p2(u::Array{Float64,1}, args)
    p_order = 2
    u1 = u[1]
    return [u1^p_order ;]
end

function p3(u::Array{Float64,1}, args)
    p_order = 3
    u1 = u[1]
    return [u1^p_order ;]
end

function exp10(u::Array{Float64,1}, args)
    u1 = u[1]
    return [exp(u1/10.0) ;]
end


function ensemble(params_i::Array{Float64, 2})
    
    n_data = 1
    
    N_ens,  N_θ = size(params_i)
    
    g_ens = zeros(Float64, N_ens,  n_data)
    
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



function UKI(t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
    θ0_bar::Array{Float64,1}, θθ0_cov::Array{Float64,2}, 
    α_reg::Float64, N_iter::Int64, update_cov::Int64)
    
    
    parameter_names = ["u₁", "u₂"]
    
    ens_func(θ_ens) = ensemble(θ_ens)
    
    ukiobj = UKIObj(parameter_names,
    θ0_bar, 
    θθ0_cov,
    t_mean, # observation
    t_cov,
    α_reg)
    
    for i in 1:N_iter
        
        update_ensemble!(ukiobj, ens_func) 
        # @info ukiobj.θ_bar[i, :]
        # @info ukiobj.g_bar[i, :]
        
        @info "g_bar = ", ukiobj.g_bar[i], "g_t = ", ukiobj.g_t
        @info "loss = ", (ukiobj.g_bar[i] - ukiobj.g_t)'/ukiobj.obs_cov*(ukiobj.g_bar[i] - ukiobj.g_t)
        
        
        if (update_cov) > 0 && (i%update_cov == 0) 
            ukiobj.θθ_cov[1] = copy(ukiobj.θθ_cov[end])
        end
        
        
    end
    
    return ukiobj
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
    
    for i in 1:N_iter
        
        update_ensemble!(ukiobj, ens_func) 
        
        # @info "θ_bar = ", ukiobj.θ_bar[i, :]
        # @info "g_bar = ", ukiobj.g_bar[i], "g_t = ", ukiobj.g_t
        # @info "loss = ", (ukiobj.g_bar[i] - ukiobj.g_t)'/ukiobj.obs_cov*(ukiobj.g_bar[i] - ukiobj.g_t)
        
        
        if (update_cov) > 0 && (i%update_cov == 0) 
            ukiobj.θθ_cov[1] = copy(ukiobj.θθ_cov[end])
        end
        
    end
    
    return ukiobj
end

function EnKI(t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
    θ0_bar::Array{Float64,1}, θθ0_cov_sqr::Array{Float64,2}, 
    α_reg::Float64, N_ens::Int64, N_iter::Int64 = 100)
    
    
    parameter_names = ["u₁", "u₂"]
    
    ens_func(θ_ens) = ensemble(θ_ens)
    
    enkiobj = EnKIObj("ETKI",
    parameter_names,
    N_ens,
    θ0_bar,
    θθ0_cov_sqr,
    t_mean,
    t_cov,
    α_reg) 
    
    for i in 1:N_iter
        
        update_ensemble!(enkiobj, ens_func) 

        
        @info "loss = ", (enkiobj.g_bar[i] - enkiobj.g_t)'/enkiobj.obs_cov*(enkiobj.g_bar[i] - enkiobj.g_t)
    end
    
    return enkiobj
end



function SMC(t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
    θ0_bar::Array{Float64,1}, θθ0_cov::Array{Float64,2}, 
    N_ens::Int64, step_length::Float64,
    M_threshold::Float64,
    N_t::Int64,
    T::Float64 = 1.0)
    
    parameter_names = ["u₁"]
    smcobj = SMCObj(parameter_names , θ0_bar , θθ0_cov , t_mean , t_cov , N_ens , step_length, M_threshold, N_t, T) 
    
    ens_func(θ_ens) = ensemble(θ_ens)
    
    
    for i in 1:N_t
        
        time_advance!(smcobj, ens_func)
        
        
    end
    
    return smcobj
end


function Map_Posterior_Plot(forward_func::Function)
    @info "start Map test: ", string(forward_func)
    # prior and covariance
    obs = forward_func([2.0;], nothing);
    obs_cov = reshape([0.1^2], (1,1))
    μ0 = [1.0;] 
    cov_sqr0    = reshape([10.0], (1, 1))
    cov0 = cov_sqr0 * cov_sqr0 

    global forward = forward_func
    
    # compute posterior distribution by MCMC
    f_density(u) = f_posterior(u, nothing, obs, obs_cov, μ0, cov0) 
    step_length = 1.0
    n_ite , n_burn_in= 5000000, 100000
    us = RWMCMC(f_density, μ0, step_length, n_ite)

    N_ens = 1000
    M_threshold = Float64(N_ens)
    N_t = 100
    smcobj = SMC(obs, obs_cov, μ0, cov0,  N_ens, 1.0, M_threshold, N_t)

    for update_cov in [5, 0]
        
        # compute posterior distribution the uki method 
        α_reg,  N_iter = 1.0, 100
        ukiobj = ExKI(obs, obs_cov,  μ0, cov0 , α_reg,  N_iter, update_cov)
        
        
        Nx = 1000;
        uki_θ, uki_θθ_std = ukiobj.θ_bar[end][1], min(max(5*sqrt(ukiobj.θθ_cov[end][1,1]), 0.05), 5)

        xx = Array(LinRange(uki_θ - uki_θθ_std, uki_θ + uki_θθ_std, Nx))
        zz = similar(xx)
    
        nrows, ncols = 2, 1
        fig, ax = PyPlot.subplots(nrows=nrows, ncols=ncols, sharex=true, sharey=true, figsize=(12,12))
        for irow = 1:nrows
            for icol = 1:ncols
                # plot UKI results 
                ites = div(N_iter, nrows*ncols)*((irow-1)*ncols + icol)
                uki_θ_bar = ukiobj.θ_bar[ites][1]
                uki_θθ_cov = ukiobj.θθ_cov[ites][1,1]

                @show uki_θ_bar,  uki_θθ_cov
                for ix = 1:Nx  
                    temp = xx[ix] - uki_θ_bar
                    zz[ix] = exp(-0.5*(temp/uki_θθ_cov*temp)) / (sqrt(2 * pi * uki_θθ_cov))
                end
                ax[irow, icol].plot(xx, zz, label="UKI")

                # plot SMC results 
                θ = smcobj.θ[end]
                weights = smcobj.weights[end]
                ax[irow, icol].hist(θ, bins = 100, weights = weights, density = true, histtype = "step", label="SMC")
                
                # plot MCMC results 
                ax[irow, icol].hist(us[n_burn_in:end, 1], bins = 100, density = true, histtype = "step", label="MCMC")

                
                ax[irow, icol].legend()
            end
        end
        
        fig.savefig(string(forward_func)*"_UKI_update_cov"*string(update_cov)*".png")
        close("all")
    end
    
end


Map_Posterior_Plot(exp10)
Map_Posterior_Plot(p2) 
Map_Posterior_Plot(p3)