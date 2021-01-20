using Random
using Distributions
using PyPlot
using LinearAlgebra
include("../Plot.jl")
include("../UKS.jl")
include("../RWMCMC.jl")



function p2(u::Array{Float64,1}, args)
    p_order = 2
    u1, u2 = u[1] + u[2]
    return [u1^p_order + u2 ;]
end

function p3(u::Array{Float64,1}, args)
    p_order = 3
    u1 = u[1]
    return [u1^p_order ;]
end

function exp10(u::Array{Float64,1}, args)
    u1 = u[1]
    u2 = u[2]
    return [1/(1 + exp(u1 + u2/2.0)) ;]
end

function xinv(u::Array{Float64,1}, args)
    u1 = u[1]
    return [1.0/u1 ;]
    
end

function signp3(u::Array{Float64,1}, args)
    u1 = u[1]
    
    return [sign(u1) + u1^3 ;]
    
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



function UKS_Run(t_mean, t_cov, θ_bar, θθ_cov,  N_iter::Int64 = 100, T::Float64 = 10.0)
    parameter_names = ["θ"]
    
    ens_func(θ_ens) = ensemble(θ_ens)
    
    UKF_modify = true
    
    Δt0 = Float64(T/N_iter)
    
    uksobj = UKSObj(parameter_names,
    θ_bar, 
    θθ_cov,
    t_mean, # observation
    t_cov,
    UKF_modify,
    Δt0)
    
    
    for i in 1:N_iter
        update_ensemble!(uksobj, ens_func) 
        # @info i , " / ", N_iter
        
    end
    
    return uksobj
    
end


function Map_Posterior_Plot(forward_func::Function, plot⁻::Bool = true)
    @info "start Map test: ", string(forward_func)
    # prior and covariance
    
    obs_cov = reshape([0.1^2], (1,1))

    obs = forward_func([2.0;2.0], nothing)
    @info "obs is :", obs
    # force it the the paper choice
    obs .= 0.08
    
    
    
    # prior distribution
    μ0,  cov_sqr0   = [1.0; 1.0], [1.0 0.0; 0.0 1.0]
    cov0  = cov_sqr0  * cov_sqr0 
    
    
    global forward = forward_func
    
    # compute posterior distribution by MCMC
    f_density(u) = f_posterior(u, nothing, obs, obs_cov, μ0, cov0) 
    step_length = 1.0
    n_ite , n_burn_in= 5000000, 1000000
    us = RWMCMC(f_density, μ0, step_length, n_ite)
    @info "MCMC min max = ", minimum(us), maximum(us)
    
    
    
    # compute posterior distribution the uks method 
    T,  N_iter = 10.0, 200000
    uksobj = UKS_Run(obs, obs_cov,  μ0, cov0 ,  N_iter, T)
    
    
    nrows, ncols = 1, 1
    fig, ax = PyPlot.subplots(nrows=nrows, ncols=ncols, sharex=true, sharey=true, figsize=(6,6))
    
    
    # plot MCMC results 
    
    everymarker = 100
    ax.scatter(us[n_burn_in:everymarker:end, 1], us[n_burn_in:everymarker:end, 2], s = 1)
    
    x_low, y_low = minimum(us[n_burn_in:end, :], dims=1)
    x_high, y_high = maximum(us[n_burn_in:end, :], dims=1)

    mcmc_mean = sum(us[n_burn_in:end, :], dims=1)/size(us[n_burn_in:end, :],1)
    mcmc_cov  = (us[n_burn_in:end, :] .- mcmc_mean)' *(us[n_burn_in:end, :] .- mcmc_mean) /(size(us[n_burn_in:end, :],1) - 1)
    @info "MCMC mean = ", mcmc_mean
    @info "MCMC cov = ", mcmc_cov

    # plot UKS results 
    Nx = 200; Ny = 200
    xx = Array(LinRange(x_low, x_high, Nx))
    yy = Array(LinRange(y_low, y_high, Ny))
    X,Y = repeat(xx, 1, Ny), repeat(yy, 1, Nx)'
    Z = zeros(Float64, Nx, Ny)
    
    
    ites = N_iter
    uks_θ_bar = uksobj.θ_bar[ites]
    uks_θθ_cov = uksobj.θθ_cov[ites]
    det_θθ_cov = det(uks_θθ_cov)
    @info uks_θ_bar, uks_θθ_cov
    for ix = 1:Nx
        for iy = 1:Ny
            temp = [xx[ix] - uks_θ_bar[1]; yy[iy] - uks_θ_bar[2]]
            Z[ix, iy] = exp(-0.5*(temp'/uks_θθ_cov*temp)) / (2 * pi * sqrt(det_θθ_cov))
        end
    end
    ax.contour(X, Y, Z, 10)
    
    fig.savefig("exp10_UKS.png")
    close("all")
    
end


Map_Posterior_Plot(exp10)
# Map_Posterior_Plot(p2) 
# Map_Posterior_Plot(p3)
# Map_Posterior_Plot(xinv)
# Map_Posterior_Plot(xinv, false)
# Map_Posterior_Plot(signp3)