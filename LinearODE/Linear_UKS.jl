using JLD2
using Statistics
using LinearAlgebra

include("../Plot.jl")
include("../RExKI.jl")
include("../UKS.jl")

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


function UKS_Run(t_mean, t_cov, θ_bar, θθ_cov,  G,  N_iter::Int64 = 100, T::Float64 = 10.0)
    parameter_names = ["θ"]
    
    ny, nθ = size(G)
    
    ens_func(θ_ens) = run_linear_ensemble(θ_ens, G)
    
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


function Linear_Test(case::String = "square", N_ite::Int64 = 100, T::Float64 = 10.0)
    
    nθ = 2
    θ0_bar = zeros(Float64, nθ)  # mean 
    θθ0_cov = Array(Diagonal(fill(1.0^2, nθ))) # cov
    
    
    if case == "square"
        # Square matrix case
        t_mean = [3.0;7.0]
        
        t_cov = Array(Diagonal(fill(0.1^2, size(t_mean) )))
        G = [1.0 2.0; 3.0 4.0]
        
    elseif (case == "under-determined")
        # under-determined case
        t_mean = [3.0;]
        
        t_cov = Array(Diagonal(fill(0.1^2, size(t_mean) )))
        G = [1.0 2.0;]
        
    elseif (case == "over-determined")
        # over-determined case
        t_mean = [3.0;7.0;10.0]
        
        t_cov = Array(Diagonal(fill(0.1^2, size(t_mean) )))
        G = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        
    else
        error("unrecognized test : ", case)
    end
    
    uksobj = UKS_Run(t_mean, t_cov, θ0_bar, θθ0_cov, G, N_ite, T)
    
    return uksobj, θ0_bar, θθ0_cov, t_mean, t_cov, G
end

mission = "2params"

if mission == "2params"
    
    algorithms = "UKS"
    tests = ["square", "under-determined", "over-determined"]
    θ_mean_ref = Dict()
    θθ_cov_ref = Dict()
    
    
    T, N_iter = 10.0, 200000
    uksobj = Dict()
    for test in tests
        @info "Start : ", test
        uksobj[test], θ0_bar, θθ0_cov, t_mean, t_cov, G = Linear_Test(test, N_iter, T)
        θ_mean_ref[test] = inv(inv(θθ0_cov) + G' * (t_cov\G))*(G'*(t_cov\t_mean) + θθ0_cov\θ0_bar)
        θθ_cov_ref[test] = inv(inv(θθ0_cov) + G' * (t_cov\G))
    end
    
    
    # Plot θ error
    
    ites = Array(LinRange(0, T, N_iter))
    errors = Dict()
    for test in tests
        errors["mean", test] = zeros(Float64, N_iter)
        errors["cov", test] = zeros(Float64, N_iter)
        for i = 1:N_iter
            errors["mean", test][i] = norm(uksobj[test].θ_bar[i]  .- θ_mean_ref[test])
            errors["cov", test][i]  = norm(uksobj[test].θθ_cov[i] .- θθ_cov_ref[test])
        end
    end
    
    
    
    semilogy(ites, errors["mean", "square"], "--o", fillstyle="none", markevery = 20000, label="NS")
    semilogy(ites, errors["mean", "over-determined"], "--s", fillstyle="none",markevery = 20000, label="OD")
    semilogy(ites, errors["mean", "under-determined"], "--^", fillstyle="none",markevery = 20000, label="UD")
    xlabel("Time")
    ylabel("\$L_2\$ norm error")
    grid("on")
    legend(bbox_to_anchor=(1.1, 1.05))
    tight_layout()
    savefig("UKS-Linear-mean.pdf")
    close("all")


    semilogy(ites, errors["cov", "square"], "--o", fillstyle="none",markevery = 20000, label="NS")
    semilogy(ites, errors["cov", "over-determined"], "--s", fillstyle="none",markevery = 20000, label="OD")
    semilogy(ites, errors["cov", "under-determined"], "--^", fillstyle="none",markevery = 20000, label="UD")
    xlabel("Time")
    ylabel("Frobenius norm error")
    grid("on")
    legend(bbox_to_anchor=(1.1, 1.05))
    tight_layout()
    savefig("UKS-Linear-cov.pdf")
    close("all")
    
end

@info "finished"




