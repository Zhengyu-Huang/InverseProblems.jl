using Random
using Distributions
using PyPlot
using LinearAlgebra
include("../Plot.jl")
include("../RUKI.jl")
include("../RExKI.jl")
include("../LSKI.jl")
include("../RWMCMC.jl")
include("../SMC.jl")
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


function EnKI_Run(filter_type, t_mean, t_cov, θ_ref, θ0_bar, 
    θθ0_cov,  N_ens, α_reg::Float64, N_iter::Int64 = 100, update_cov::Int64 = 0)

    parameter_names = ["θ"]

    ens_func(θ_ens) = ensemble(θ_ens)

    @info typeof(θθ0_cov)

    ekiobj = EnKIObj(filter_type,
    parameter_names,
    N_ens,
    θ0_bar,
    θθ0_cov,
    t_mean, # observation
    t_cov,
    α_reg)

    #### Daniel Huang todo
    ekiobj.Σ_ω .= 0.0 
    ekiobj.Σ_ν ./= 2.0
    ####
    errors = zeros(Float64, N_iter+1)
    θ_ref_norm = norm(θ_ref)

    θ_bar = dropdims(mean(ekiobj.θ, dims=1), dims=1)
    errors[1] = norm(θ_bar .- θ_ref)/θ_ref_norm

    for i in 1:N_iter

        update_ensemble!(ekiobj, ens_func)
        θ_bar .= dropdims(mean(ekiobj.θ, dims=1), dims=1)
        
        @info "error is :", norm(θ_bar .- θ_ref)/θ_ref_norm

    end

    return ekiobj

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
    n_ite , n_burn_in= 5000000, 1000000
    us = RWMCMC_Run(f_density, μ0, step_length, n_ite)

    uki_cov0 = [1.0  0.0; 0.0 100.0]
    for update_cov in [0,1]
        
        
        # compute posterior distribution the uki method 
        α_reg,  N_iter = 1.0, 30
        ukiobj = ExKI(obs, obs_cov,  μ0, uki_cov0 , α_reg,  N_iter, update_cov)
        
        Nx = 100; Ny = 200
        xx = Array(LinRange(-4, -2, Nx))
        yy = Array(LinRange(103, 106, Ny))
        X,Y = repeat(xx, 1, Ny), repeat(yy, 1, Nx)'
        Z = zeros(Float64, Nx, Ny)
        
        
        ncols = 3
        fig, ax = PyPlot.subplots(ncols=ncols, sharex=true, sharey=true, figsize=(15,5))
            for icol = 1:ncols
                # plot UKI results 
                ites = 5*icol
                uki_θ_bar = ukiobj.θ_bar[ites]
                uki_θθ_cov = ukiobj.θθ_cov[ites]
                det_θθ_cov = det(uki_θθ_cov)
                for ix = 1:Nx
                    for iy = 1:Ny
                        temp = [xx[ix] - uki_θ_bar[1]; yy[iy] - uki_θ_bar[2]]
                        Z[ix, iy] = exp(-0.5*(temp'/uki_θθ_cov*temp)) / (2 * pi * sqrt(det_θθ_cov))
                    end
                end
                ax[icol].contour(X, Y, Z, 50)
                
                # plot MCMC results 
                
                everymarker = 1
                ax[icol].scatter(us[n_burn_in:everymarker:end, 1], us[n_burn_in:everymarker:end, 2], s = 1)
            end
  
        fig.tight_layout()
        fig.savefig("Elliptic_UKI_update_cov"*string(update_cov)*".png")
        close("all")
    end
    
end

function Ellitic_Posterior_Compare_Plot()

    # prior and covariance
    obs = [27.5; 79.7]
    obs_cov = [0.1^2   0.0; 0.0  0.1^2]
     
    cov_sqr0    = [1.0  0.0; 0.0 10.0]
    cov0 = cov_sqr0 * cov_sqr0 

    μ0 = [0.0; 0.0]
    uki_cov0 = [1.0  0.0; 0.0 100.0]


    # compute posterior distribution by MCMC
    f_density(u) = f_posterior(u, nothing, obs, obs_cov, μ0, cov0) 
    step_length = 1.0
    n_ite , n_burn_in= 5000000, 1000000
    us_mcmc = RWMCMC_Run(f_density, μ0, step_length, n_ite)


    everymarker = 1
    ncols = 3
    nrows = 2
    fig, ax = PyPlot.subplots(ncols=ncols, nrows = nrows, sharex=false, sharey=false, figsize=(15,10))
    for i = 1:ncols*nrows
        ax[i].scatter(us_mcmc[n_burn_in:everymarker:end, 1], us_mcmc[n_burn_in:everymarker:end, 2], s = 1)
    end


    # emmce
   
    N_ens = 100
    θ0 = rand(MvNormal(μ0, uki_cov0), N_ens)
    θ0 = Array(θ0')
    n_ite = 500
    us_emcee = emcee_Run(f_density, θ0, n_ite);
    n_burn_in = 1
    ax[1,1].scatter(us_emcee[n_burn_in:100, :, 1], us_emcee[n_burn_in:100, :, 2], s = 2, color="C1", label="emcee (100)")
    ax[2,1].scatter(us_emcee[n_burn_in:end, :, 1], us_emcee[n_burn_in:end, :, 2], s = 2, color="red", label="emcee (500)")
    
    # compute posterior distribution by SMC
    N_ens = 100
    M_threshold = Float64(N_ens)*0.6
    step_length = 1.0

    N_t = 100
    smcobj = SMC(obs, obs_cov, μ0, cov0,  N_ens, step_length, M_threshold, N_t)
    θ = smcobj.θ[end]
    weights = smcobj.weights[end]
    θ_p = copy(θ)
    for i = 1:N_ens
        θ[i, :] .= θ_p[sample(Weights(weights)), :]
    end
    ax[1,2].scatter(θ[:, 1], θ[:, 2], s = 2, color="C1", label="SMC ($(N_t))")


    N_t = 500
    smcobj = SMC(obs, obs_cov, μ0, cov0,  N_ens, 1.0, M_threshold, N_t)
    θ = smcobj.θ[end]
    weights = smcobj.weights[end]
    θ_p = copy(θ)
    for i = 1:N_ens
        θ[i, :] .= θ_p[sample(Weights(weights)), :]
    end
    ax[2,2].scatter(θ[:, 1], θ[:, 2], s = 2, color="red", label="SMC ($(N_t))")
    
    # ensemble transform inversion

    filter_type = "ETKI"
    N_ens = 100
    α_reg,  N_iter = 1.0, 30
    update_cov = 0

    θ_ref = [-3.0, 104.0]

    N_iter = 1
    ekiobj = EnKI_Run(filter_type, obs, obs_cov,θ_ref, μ0, uki_cov0,  
    N_ens, α_reg, N_iter, update_cov)
    ax[1,3].scatter(ekiobj.θ[:, 1], ekiobj.θ[:, 2], s = 2, color="C1", label="ETKI (1)")
    
    N_iter = 30
    ekiobj = EnKI_Run(filter_type, obs, obs_cov,θ_ref, μ0, uki_cov0,  
    N_ens, α_reg, N_iter, update_cov)
    ax[2,3].scatter(ekiobj.θ[:, 1], ekiobj.θ[:, 2], s = 2, color="red", label="ETKI (30)")
    

    
    for i = 1:ncols*nrows
        if i != 3
            ax[i].set_xlim([-4, 0])
            ax[i].set_ylim([103, 106])
        end
        ax[i].legend()
    end

    fig.tight_layout()
    fig.savefig("Elliptic_Comp.png")
    close("all")
end


# Ellitic_Posterior_Plot()

Ellitic_Posterior_Compare_Plot()