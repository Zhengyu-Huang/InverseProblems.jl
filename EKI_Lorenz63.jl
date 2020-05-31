using JLD2
using Statistics
using LinearAlgebra

include("EKI.jl")
include("UKI.jl")
include("Lorenz63.jl")


function compute_obs(xs::Array{Float64,2})
    # N_x × N_t
    x1, x2, x3 = xs[1,:]', xs[2,:]', xs[3,:]'
    obs = [x1; x2; x3; x1.^2; x2.^2; x3.^2]
end

# fix σ, learn r and β
# f = [σ*(x[2]-x[1]); x[1]*(r-x[3])-x[2]; x[1]*x[2]-β*x[3]]
# σ, r, β = 10.0, 28.0, 8.0/3.0
function run_Lorenz_ensemble(params_i, Tobs, Tspinup, Δt)
    T = Tobs + Tspinup
    nt = Int64(T/Δt)
    nspinup = Int64(Tspinup/Δt)
    
    N_ens,  N_θ = size(params_i)
 

    g_ens = Vector{Float64}[]

    for i = 1:N_ens
        # σ, r, β = 10.0, 28.0, 8.0/3.0
        σ, r, β = params_i[i, :]
        σ, β = abs(σ), abs(β)

        
        μ = [-8.67139571762; 4.98065219709; 25; σ; r; β]
        xs = compute_foward(μ, Δt, nt)
        obs = compute_obs(xs)
        # g: N_ens x N_data
        push!(g_ens, dropdims(mean(obs[:, nspinup : end], dims = 2), dims=2)) 
    end
    return hcat(g_ens...)'
end


function Data_Gen(T::Float64 = 360.0, Tobs::Float64 = 10.0, Tspinup::Float64 = 30.0, Δt::Float64 = 0.01)
    
    nt = Int64(T/Δt)
    nspinup = Int64(Tspinup/Δt)
    σ, r, β = 10.0, 28.0, 8.0/3.0
    #σ, r, β = 10.0, 8.0/3.0, 28.0

    μ = [-8.67139571762; 4.98065219709; 25; σ; r; β]
    xs = compute_foward(μ, Δt, nt)

    obs = compute_obs(xs)
  
    ##############################################################################

    n_obs_box = Int64((T - Tspinup)/Tobs)
    obs_box = zeros(size(obs, 1), n_obs_box)

    n_obs = Int64(Tobs/Δt)

    for i = 1:n_obs_box
        n_obs_start = nspinup + (i - 1)*n_obs + 1
        obs_box[:, i] = mean(obs[:, n_obs_start : n_obs_start + n_obs - 1], dims = 2)
    end

    #t_mean = vec(mean(obs_box, dims=2))
    t_mean = vec(obs_box[:, 1])

    t_cov = zeros(Float64, size(obs, 1), size(obs, 1))
    for i = 1:n_obs_box
        t_cov += (obs_box[:,i] - t_mean) *(obs_box[:,i] - t_mean)'
    end

    @info obs_box
    t_cov ./= (n_obs_box - 1)
    
    @save "t_cov.jld2" t_cov
    @save "t_mean.jld2" t_mean
    return t_mean, t_cov
end


function Data_Gen(n_obs_box::Int64 = 360.0, Tobs::Float64 = 10.0, Tspinup::Float64 = 30.0, Δt::Float64 = 0.01)
    
    T = Tobs + Tspinup
    nt = Int64(T/Δt)
    nspinup = Int64(Tspinup/Δt)
    n_obs = Int64(Tobs/Δt)
    σ, r, β = 10.0, 28.0, 8.0/3.0

    obs_box = Vector{Float64}[]


    Random.seed!(6)
    σ_x0 = zeros(Float64, 3, 3)
    σ_x0[1,1] = σ_x0[2,2] = σ_x0[3,3] = 10.0
    dist = Distributions.MvNormal([-8.67139571762; 4.98065219709; 25], σ_x0)
    x0 = rand(dist, n_obs_box)

    for i = 1:n_obs_box

        μ = [x0[1,i]; x0[2,i]; x0[3,i]; σ; r; β]
        xs = compute_foward_RK4(μ, Δt, nt)
        obs = compute_obs(xs)
        
        push!(obs_box, dropdims(mean(obs[:, nspinup + 1 : nspinup + n_obs], dims = 2), dims =2))

    end

    t_mean = vec(mean(obs_box))

    t_cov = zeros(Float64, size(t_mean, 1), size(t_mean, 1))

    @info size(t_cov), size(obs_box[1]),  size(t_mean)
    for i = 1:n_obs_box
        t_cov += (obs_box[i] - t_mean) *(obs_box[i] - t_mean)'
    end

    @info obs_box
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



function UKI_Run(t_mean, t_cov, Tobs::Float64 = 10.0, Tspinup::Float64 = 30.0, Δt::Float64 = 0.01)

    parameter_names = ["r", "β"]
    # initial distribution is 
    θ_bar = [5.0 ; 5.0;  5.0]           # mean 
    θθ_cov = [0.5^2  0.0    0.0; 
              0.0    0.5^2  0.0;      # standard deviation
              0.0    0.0    0.5^2;]
    # θ_bar = [1.2]           # mean 
    # θθ_cov = reshape([0.5^2],1,1)       # standard deviation

    ens_func(θ_ens) = run_Lorenz_ensemble(θ_ens, Tobs, Tspinup, Δt)

    ukiobj = UKIObj(parameter_names,
                θ_bar, 
                θθ_cov,
                t_mean, # observation
                t_cov)
    

    # UKI iterations
    N_iter = 100

    for i in 1:N_iter
        # Note that the parameters are exp-transformed for use as input
        # to Cloudy
        params_i = deepcopy(ukiobj.θ_bar[end])

        @info "At iter ", i, " params_i : ", params_i
        
        update_ensemble!(ukiobj, ens_func) 

        # if i%10 == 1
        #     reset_θθ0_cov!(ukiobj)
        # end
    end

    @info "θ is ", ukiobj.θ_bar[end], " θ_ref is ",  28.0, 8.0/3.0
    
end

Tobs = 20.0
Tspinup = 30.0
T = Tspinup + 30*Tobs
Δt = 0.01

Data_Gen(T, Tobs, Tspinup, Δt)
#Data_Gen(30, Tobs, Tspinup, Δt)
    
@load "t_mean.jld2"
@load "t_cov.jld2"

@info t_mean
@info t_cov
# error("t_mean")

@info "t_mean is ", t_mean
@info "t_cov is ", t_cov
# t_cov .*= 0.0
# t_cov += 1e-4*I
# t_cov[1,2] = 0.0
# t_cov[2,1] = 0.0
# t_cov = Array(Diagonal((t_mean*0.05).^2))
# t_cov .*= 0.0
#EKI_Run(t_mean, t_cov, Tobs, Tspinup, Δt)
UKI_Run(t_mean, t_cov, Tobs, Tspinup, Δt)