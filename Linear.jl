using JLD2
using Statistics
using LinearAlgebra

include("EKI.jl")
include("UKI.jl")

function run_linear_ensemble(params_i, G)
    
    N_ens,  N_θ = size(params_i)
    g_ens = Vector{Float64}[]

    for i = 1:N_ens
        θ = params_i[i, :]
        # g: N_ens x N_data
        push!(g_ens, G*θ) 
    end

    return hcat(g_ens...)'
end



function Data_Gen(θ, G, Σ_η)
    # y = Gθ + η
    t_mean, t_cov = G*θ, Σ_η

    
    @save "t_mean.jld2" t_mean
    @save "t_cov.jld2" t_cov

    return t_mean, t_cov

end

function EKI_Run(t_mean, t_cov, G)

    parameter_names = ["θ"]
    ny, nθ = size(G)

    # initial distribution is 
    μ0 = ones(Float64, nθ)   # mean     
    σ0 = Array(Diagonal(sqrt.(fill(0.01, nθ))))  #standard deviation

    

    priors = [Distributions.Normal(μ0[i], σ0[i]) for i=1:nθ]
    
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
        
        g_ens = run_linear_ensemble(params_i, G)

        update_ensemble!(ekiobj, g_ens) 
        #update_ensemble_eks!(ekiobj, g_ens) 

    end
    
end



function UKI_Run(t_mean, t_cov, G)
    parameter_names = ["θ"]

    ny, nθ = size(G)
    # initial distribution is 
    θ_bar = ones(Float64, nθ)  # mean 
    #θθ_cov = Array(Diagonal(fill(0.01, nθ)))     # standard deviation
    θθ_cov =[0.15333333333333332 -0.049999999999999996; -0.05 0.016666666666666666]
    ens_func(θ_ens) = run_linear_ensemble(θ_ens, G)

    ukiobj = UKIObj(parameter_names,
                θ_bar, 
                θθ_cov,
                t_mean, # observation
                t_cov)
    

    # EKI iterations
    N_iter = 100

    for i in 1:N_iter
        # Note that the parameters are exp-transformed for use as input
        # to Cloudy
        params_i = deepcopy(ukiobj.θ_bar[end])

        @info "params_i : ", params_i
        
        update_ensemble!(ukiobj, ens_func) 

        # if i%10 == 1 
        #     reset_θθ0_cov!(ukiobj)
        # end

    end
    
end


θ = [3.0, 2.0]
G = [1.0 2.0; 3.0 9.0]
Σ_η = [0.02 0.01; 0.01 0.03]
Σ_θ = inv(G)*Σ_η*inv(G)'
Data_Gen(θ, G, Σ_η)

@load "t_mean.jld2"
@load "t_cov.jld2"

@info "t_mean is ", t_mean
@info "t_cov is ", t_cov


#EKI_Run(t_mean, t_cov, G)
UKI_Run(t_mean, t_cov, G)

@show θ, Σ_θ 