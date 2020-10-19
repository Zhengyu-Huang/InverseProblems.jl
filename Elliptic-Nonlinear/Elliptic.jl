using Random
using Distributions
using PyPlot
using LinearAlgebra
include("../Plot.jl")
include("../RUKI.jl")
include("../RExKI.jl")
include("../REnKI.jl")
# p(x) = u₂x + exp(-u₁)(-x²/2 + x/2)

function forward(u::Array{Float64,1}, args)
    x1, x2 = 0.25, 0.75
    u1, u2 = u
    p = (x) -> u2*x + exp(-u1)*(-x^2/2 + x/2)
    return [p(x1) ; p(x2)]
end


function dforward(u::Array{Float64,1}, args)
    x1, x2 = 0.25, 0.75
    u1, u2 = u
    dpdu = (x) -> [-exp(-u1)*(-x^2/2 + x/2) ; x]
    return [dpdu(x1)' ; dpdu(x2)']
end

function backward(p::Array{Float64,1}, args)
    x1, x2 = 0.25, 0.75
    
    #
    #  x1  -x1^2/2 + x1/2      u2           = p1
    #  x2  -x2^2/2 + x2/2   exp(-u1)        = p2
    #
    #        u2           = q1
    #      exp(-u1)       = q2
    #    
    
    X = [x1  -x1^2/2+x1/2; x2  -x2^2/2+x2/2]
    
    q = X\p
    
    return [-log(abs(q[2])) ; q[1]]
end

function ensemble(params_i::Array{Float64, 2})
    
    n_data = 2
    
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

function RWMH(f_density::Function, u0::Array{Float64,1}, step_length::Float64, n_ite::Int64)
    
    n = length(u0)
    us = zeros(Float64, n_ite, n)
    fs = zeros(Float64, n_ite)
    
    us[1, :] .= u0
    fs[1] = f_density(u0)
    
    Random.seed!(666)
    for i = 2:n_ite
        u_p = us[i-1, :] 
        u = u_p + step_length * rand(Normal(0, 1), n)
        
        
        fs[i] = f_density(u)
        α = min(1.0, exp(fs[i] - fs[i-1]))
        us[i, :] = (rand(Bernoulli(α)) ? u : u_p)
        fs[i] = (rand(Bernoulli(α)) ? fs[i] : fs[i-1])
    end
    
    return us
    
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
        # @info ukiobj.θ_bar[i, :]
        # @info ukiobj.g_bar[i, :]
        
        @info "loss = ", (enkiobj.g_bar[i] - enkiobj.g_t)'/enkiobj.obs_cov*(enkiobj.g_bar[i] - enkiobj.g_t)
        
        
    end
    
    return enkiobj
end



function Ellitic_Posterior_Plot()

    # prior and covariance
    obs = [27.5; 79.7]
    obs_cov = [0.1^2   0.0; 0.0  0.1^2]
    μ0 = [0.0; 0.0] 
    cov_sqr0    = [1.0  0.0; 0.0 10.0]
    cov0 = cov_sqr0 * cov_sqr0 
    
    # compute posterior distribution by MCMC
    f_density(u) = f_posterior(u, nothing, obs, obs_cov, μ0, cov0) 
    step_length = 1.0
    n_ite , n_burn_in= 1000000, 100000
    us = RWMH(f_density, μ0, step_length, n_ite)
    for update_cov in [0,5]
        
        
        # compute posterior distribution the uki method 
        α_reg,  N_iter = 1.0, 30
        ukiobj = ExKI(obs, obs_cov,  μ0, cov0 , α_reg,  N_iter, update_cov)
        
        Nx = 100; Ny = 200
        xx = Array(LinRange(-4, -2, Nx))
        yy = Array(LinRange(103, 106, Ny))
        X,Y = repeat(xx, 1, Ny), repeat(yy, 1, Nx)'
        Z = zeros(Float64, Nx, Ny)
        
        
        nrows, ncols = 2, 3
        fig, ax = PyPlot.subplots(nrows=nrows, ncols=ncols, sharex=true, sharey=true, figsize=(18,12))
        for irow = 1:nrows
            for icol = 1:ncols
                # plot UKI results 
                ites = 5*((irow-1)*ncols + icol)
                uki_θ_bar = ukiobj.θ_bar[ites]
                uki_θθ_cov = ukiobj.θθ_cov[ites]
                det_θθ_cov = det(uki_θθ_cov)
                for ix = 1:Nx
                    for iy = 1:Ny
                        temp = [xx[ix] - uki_θ_bar[1]; yy[iy] - uki_θ_bar[2]]
                        Z[ix, iy] = exp(-0.5*(temp'/uki_θθ_cov*temp)) / (2 * pi * sqrt(det_θθ_cov))
                    end
                end
                ax[irow, icol].contour(X, Y, Z, 50)
                
                # plot MCMC results 
                
                everymarker = 1
                ax[irow, icol].scatter(us[n_burn_in:everymarker:end, 1], us[n_burn_in:everymarker:end, 2], s = 1)
            end
        end
        
        fig.savefig("Elliptic_UKI_update_cov"*string(update_cov)*".png")
        close("all")
    end
    
end


Ellitic_Posterior_Plot()