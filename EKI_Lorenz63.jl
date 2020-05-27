using JLD2
using Statistics
using LinearAlgebra

include("EKI.jl")
include("Lorenz63.jl")

# fix σ, learn r and β
# f = [σ*(x[2]-x[1]); x[1]*(r-x[3])-x[2]; x[1]*x[2]-β*x[3]]
# σ, r, β = 10.0, 28.0, 8.0/3.0

function run_Lorenz_ensemble(params_i, Tobs, Tspinup, Δt)

    T = Tobs + Tspinup
    nt = Int64(T/Δt)
    nspinup = Int64(Tspinup/Δt)
    
    N_ensemble,  N_parameters = size(params_i)
    #N_data = 9
    N_data = 2

    g_ens = zeros(Float64,N_ensemble,  N_data)

    for i = 1:N_ensemble
        σ, r, β = 10.0, 28.0, 8.0/3.0
        #r, β = params_i[i, :]
        r = params_i[i, :]
        μ = [-8.67139571762; 4.98065219709; 25; σ; r; β]
        xs = compute_foward_RK4(μ, Δt, nt)

        x1, x2, x3 = xs[1,:]', xs[2,:]', xs[3,:]'

        #obs = [x1; x2; x3; x1.^2; x2.^2; x3.^2; x2.*x3; x3.*x1; x1.*x2]
        obs = [x3;x3.^2]
        
        # g: N_ens x N_data

        g_ens[i, :] = mean(obs[:, nspinup : end], dims = 2)
    end

    @info "g_ens", mean(g_ens, dims=1)

    return g_ens

end

function Data_Gen(T::Float64 = 360.0, Tobs::Float64 = 10.0, Tspinup::Float64 = 30.0, Δt::Float64 = 0.01)
    
    nt = Int64(T/Δt)
    nspinup = Int64(Tspinup/Δt)
    σ, r, β = 10.0, 28.0, 8.0/3.0
    #σ, r, β = 10.0, 8.0/3.0, 28.0

    μ = [-8.67139571762; 4.98065219709; 25; σ; r; β]
    xs = compute_foward_RK4(μ, Δt, nt)

    
    x1, x2, x3 = xs[1,:]', xs[2,:]', xs[3,:]'

    #obs = [x1; x2; x3; x1.^2; x2.^2; x3.^2; x2.*x3; x3.*x1; x1.*x2]
    obs = [x3;x3.^2]
  

    ##############################################################################

    n_obs_box = Int64((T - Tspinup)/Tobs)
    obs_box = zeros(size(obs, 1), n_obs_box)

    n_obs = Int64(Tobs/Δt)

    for i = 1:n_obs_box
        
        n_obs_start = nspinup + (i - 1)*n_obs + 1

        @info n_obs_start , n_obs_start + n_obs - 1

        obs_box[:, i] = mean(obs[:, n_obs_start : n_obs_start + n_obs - 1], dims = 2)
    end

    #t_mean = vec(mean(obs_box, dims=2))
    t_mean = vec(obs_box[:, 1])

    t_cov = zeros(Float64, size(obs, 1), size(obs, 1))
    for i = 1:n_obs_box
        t_cov += (obs_box[:,i] - t_mean) *(obs_box[:,i] - t_mean)'
    end

    t_cov ./= (n_obs_box - 1)
    
    @save "t_cov.jld2" t_cov
    @save "t_mean.jld2" t_mean
    return t_mean, t_cov
end

function EKI_Run(t_mean, t_cov, Tobs::Float64 = 10.0, Tspinup::Float64 = 30.0, Δt::Float64 = 0.01)


    parameter_names = ["r", "β"]
    # initial distribution is 
    μ0 = [1.2 ; 3.3]       #mean 
    σ0 = [0.5,  0.15]      #standard deviation

    # μ0 = [3.3 ; 1.2]       #mean 
    # σ0 = [0.15,  0.5]      #standard deviation
    
    # prior normal
    # priors = [Distributions.Normal(μ0[1], σ0[1]),  
    # Distributions.Normal(μ0[2], σ0[2])]

    priors = [Distributions.Normal(μ0[1], σ0[1])]
    
    N_ens = 500
    initial_params = construct_initial_ensemble(N_ens, priors; rng_seed=6)
    
    # observation t_mean, observation covariance matrix t_cov
    
    ekiobj = EKIObj(initial_params, parameter_names, t_mean, t_cov)
    
    
    # EKI iterations
    N_iter = 100

    for i in 1:N_iter
        # Note that the parameters are exp-transformed for use as input
        # to Cloudy
        params_i = deepcopy(ekiobj.u[end])

        @info "params_i : ", mean(params_i, dims=1)
        
        g_ens = run_Lorenz_ensemble(params_i, Tobs, Tspinup, Δt)

        update_ensemble!(ekiobj, g_ens) 
        #update_ensemble_eks!(ekiobj, g_ens) 

    end
    
end





Tobs = 10.0
Tspinup = 30.0
T = Tspinup + 30*Tobs
Δt = 0.01

Data_Gen(T, Tobs, Tspinup, Δt)
@load "t_mean.jld2"
@load "t_cov.jld2"

@info "t_mean is ", t_mean
@info "t_cov is ", t_cov
# t_cov .*= 0.0
# t_cov += 1e-4*I
t_cov[1,2] = 0.0
t_cov[2,1] = 0.0
# t_cov .*= 0.0
EKI_Run(t_mean, t_cov, Tobs, Tspinup, Δt)