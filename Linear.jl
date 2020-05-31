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



function UKI_Run(t_mean, t_cov, θ_bar, θθ_cov,  G, update_cov::Int64 = 0)
    parameter_names = ["θ"]
    
    ny, nθ = size(G)
    # initial distribution is 

    @info θ_bar, θθ_cov
    θ_bar = zeros(Float64, nθ)  # mean 
    θθ_cov = Array(Diagonal(fill(0.5^2, nθ)))     # standard deviation
    
    @info θ_bar, θθ_cov

    #todo delete
    #θθ_cov =[0.02 0.01; 0.01 0.03]
    ens_func(θ_ens) = run_linear_ensemble(θ_ens, G)
    
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
        
        @info "At iter ", i, " θ: ", params_i
        
        update_ensemble!(ukiobj, ens_func) 
        
        if (update_cov) > 0 && (i%update_cov == 1) 
            reset_θθ0_cov!(ukiobj)
        end
        
    end
    
    return ukiobj
    
end

function Linear_Test(update_cov::Int64 = 0, case::String = "square")
    γ = 2.0

    nθ = 2
    θ0_bar = zeros(Float64, nθ)  # mean 
    θθ0_cov = Array(Diagonal(fill(0.5^2, nθ)))     # standard deviation


    if case == "square"
        # Square matrix case
        t_mean = [3.0;7.0]

        t_cov = Array(Diagonal(fill(0.1^2, size(t_mean) )))
        G = [1.0 2.0; 3.0 4.0]
        
        
        ukiobj = UKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov, G, update_cov)
        @info "θ ~ N ( " ukiobj.θ_bar[end], ukiobj.θθ_cov[end], " )"
        θθ_cov_opt = inv(G)*t_cov*inv(G)'
        @info "optimal error : ", inv(G)*t_cov*inv(G)' - ukiobj.θθ_cov[end]
        @info "suboptimal error :", inv(ukiobj.θθ_cov[end]) - 1/γ*inv(θθ_cov_opt) - (γ-1)/γ*inv((γ-1)/γ*ukiobj.θθ_cov[end] + 1/γ*ukiobj.θθ_cov[1])
        
    end
    
    if (case == "under-determined")
        # under-determined case
        t_mean = [3.0;]
        
        t_cov = Array(Diagonal(fill(0.1^2, size(t_mean) )))
        G = [1.0 2.0;]
        
        ukiobj = UKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov, G, update_cov)
        @info "θ ~ N ( " ukiobj.θ_bar[end], ukiobj.θθ_cov[end], " )"

        
    end
    
    if (case == "over-determined")
        # over-determined case
        t_mean = [3.0;7.0;10.0]
        
        t_cov = Array(Diagonal(fill(0.1^2, size(t_mean) )))
        G = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        
        ukiobj = UKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov, G, update_cov)
        @info "θ ~ N ( " ukiobj.θ_bar[end], ukiobj.θθ_cov[end], " )"

        @info "optimal error : ", inv(ukiobj.θθ_cov[end]) - G'*inv(t_cov)*G
        #@info "suboptimal error :", inv(ukiobj.θθ_cov[end]) - 1/γ*inv(θθ_cov_opt) - (γ-1)/γ*inv((γ-1)/γ*ukiobj.θθ_cov[end] + 1/γ*t_cov)
        
    end
    return ukiobj
end

function Hilbert_Test()
    nθ = 100
    θ0_bar = zeros(Float64, nθ)  # mean 
    θθ0_cov = Array(Diagonal(fill(0.5^2, nθ)))     # standard deviation

    G = zeros(nθ, nθ)
    for i = 1:nθ
        for j = 1:nθ
            G[i,j] = 1/(i + j - 1)
        end
    end
    
    θ_ref = fill(1.0, nθ)
    t_mean = G*θ_ref 
    t_cov = Array(Diagonal(fill(0.5^2, nθ)))
    
    ukiobj = UKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov, G)
    @info "θ ~ N ( " ukiobj.θ_bar[end], ukiobj.θθ_cov[end], " )"
end

# ukiobj = Linear_Test(0, "square")
# ukiobj = Linear_Test(10, "square")

# ukiobj = Linear_Test(0, "under-determined")
# ukiobj = Linear_Test(10, "under-determined")

# ukiobj = Linear_Test(0, "over-determined")
# ukiobj = Linear_Test(10, "over-determined")
Hilbert_Test()