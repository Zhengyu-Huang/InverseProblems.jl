using JLD2
using Statistics
using LinearAlgebra
using PyPlot
include("../Plot.jl")
include("../REnKI.jl")
include("../TRUKI.jl")
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


function TRUKI_Run(t_mean, t_cov, θ_bar, θθ_cov_sqr,  G,  N_r, α_reg::Float64,  N_iter::Int64 = 100)
    parameter_names = ["θ"]
    
    ens_func(θ_ens) = run_linear_ensemble(θ_ens, G)
    
    trukiobj = TRUKIObj(parameter_names,
    N_r,
    θ_bar, 
    θθ_cov_sqr,
    t_mean, # observation
    t_cov,
    α_reg)
    
    for i in 1:N_iter
        
        update_ensemble!(trukiobj, ens_func) 
        
    end
    
    return trukiobj
    
end

function LRRUKI_Run(t_mean, t_cov, θ_bar, θθ_cov,  G,  N_r, α_reg::Float64,  N_iter::Int64 = 100)
    parameter_names = ["θ"]
    
    ens_func(θ_ens) = run_linear_ensemble(θ_ens, G)
    
    lrrukiobj = LRRUKIObj(parameter_names,
    N_r,
    θ_bar, 
    θθ_cov,
    t_mean, # observation
    t_cov,
    α_reg)
    
    for i in 1:N_iter
        
        update_ensemble!(lrrukiobj, ens_func) 
        
    end
    
    return lrrukiobj
    
end


function EnKI_Run(filter_type, t_mean, t_cov, θ0_bar, θθ0_cov,  G,  N_ens, α_reg::Float64, N_iter::Int64 = 100)
    parameter_names = ["θ"]
    

    
    ens_func(θ_ens) = run_linear_ensemble(θ_ens, G)
    
    
    
    ekiobj = EnKIObj(filter_type, 
    parameter_names,
    N_ens, 
    θ0_bar,
    θθ0_cov,
    t_mean, # observation
    t_cov,
    α_reg)
    
    
    for i in 1:N_iter
        
        update_ensemble!(ekiobj, ens_func) 
        
    end
    
    return ekiobj
    
end


function Linear_Test(problem_type::String, nθ::Int64 = 10, N_r::Int64 = 1, α_reg::Float64 = 1.0, N_ite::Int64 = 1000)
    
    
    h = 1/(nθ + 1)

    if problem_type == "Elliptic"
        # -p'' + p = f        p(0) = p(1) = 0
        # discretize 0,h,2h, ..., 1,     h = 1/(nθ + 1)
        # -(p(i-1) - 2p(i) + p(i+1))/h^2 + p(i) = f(i)
        #  (-p(i-1) + 2p(i) - p(i+1))/h^2 + p(i) = f(i)
        
        
        G = Tridiagonal(fill(-1.0/h^2, nθ-1), fill(1+2.0/h^2, nθ), fill(-1.0/h^2, nθ-1));
    elseif problem_type == "Identity"
        G = I
    else
        error("Problem type : ", problem_type, " has not implemented!")
    end
    x = LinRange(h, 1-h, nθ)

    f = fill(1.0, nθ)
    
    θ_ref = G\f

    t_mean = f
    t_cov = Array(Diagonal(fill(1.0^2, nθ)))


    θ0_bar = zeros(Float64, nθ)  # mean 
    θθ0_cov = Array(Diagonal(fill(10.0^2, nθ)))     # standard deviation

    Z0_cov = ones(Float64, nθ, N_r)
    
    for i = 1:N_r
        Z0_cov[:, i] = sin.(i *  pi*x) 
    end


    θθ0_cov = Z0_cov * Z0_cov' * 1 + Array(Diagonal(fill(1e-10, nθ))) 
    
    
    enkiobj = EnKI_Run("EnKI", t_mean, t_cov, θ0_bar, θθ0_cov, G, 2N_r+1, α_reg, N_ite)
    eakiobj = EnKI_Run("EAKI", t_mean, t_cov, θ0_bar, θθ0_cov, G, 2N_r+1, α_reg, N_ite)
    etkiobj = EnKI_Run("ETKI", t_mean, t_cov, θ0_bar, θθ0_cov, G, 2N_r+1, α_reg, N_ite)
    trukiobj = TRUKI_Run(t_mean, t_cov, θ0_bar, Z0_cov, G, N_r, α_reg, N_ite)
    
    # Plot
    ites = Array(LinRange(1, N_ite+1, N_ite+1))
    errors = zeros(Float64, (4, N_ite+1))
    for i = 1:N_ite+1
        
        
        θ_bar = dropdims(mean(enkiobj.θ[i], dims=1), dims=1)
        errors[1, i] = norm(θ_bar .- θ_ref)

        θ_bar = dropdims(mean(eakiobj.θ[i], dims=1), dims=1)
        errors[2, i] = norm(θ_bar .- θ_ref)

        θ_bar = dropdims(mean(etkiobj.θ[i], dims=1), dims=1)
        errors[3, i] = norm(θ_bar .- θ_ref)

        errors[4, i] = norm(trukiobj.θ_bar[i] .- θ_ref)
        
        
    end

    # error("stop")

    errors ./= norm(θ_ref)
    
    semilogy(ites, errors[1, :], "--o", fillstyle="none", markevery=10, label= "EnKI")
    semilogy(ites, errors[2, :], "--o", fillstyle="none", markevery=10, label= "EAKI")
    semilogy(ites, errors[3, :], "--o", fillstyle="none", markevery=10, label= "ETKI")
    semilogy(ites, errors[4, :], "--o", fillstyle="none", markevery=10, label= "TRUKI")
    
    xlabel("Iterations")
    ylabel("\$L_2\$ norm error")
    #ylim((0.1,15))
    grid("on")
    legend()
    tight_layout()
    
    savefig("Linear-"*string(nθ)*".pdf")
    close("all")
    
    return trukiobj
end




problem_type = "Elliptic"  #"Identity"# "Elliptic"  #"Identity" #"Poisson_1D" 
Linear_Test(problem_type, 200, 5, 1.0, 50)







