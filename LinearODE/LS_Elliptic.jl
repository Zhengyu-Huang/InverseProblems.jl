using JLD
using Statistics
using LinearAlgebra
using PyPlot
include("../Plot.jl")
include("../LSKI.jl")

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


function TRUKI_Run(t_mean, t_cov, θ_ref, θ_bar, θθ_cov_sqr,  G,  N_r, α_reg::Float64,  N_iter::Int64 = 100)
    parameter_names = ["θ"]
    
    ens_func(θ_ens) = run_linear_ensemble(θ_ens, G)
    
    trukiobj = TRUKIObj(parameter_names,
    N_r,
    θ_bar, 
    θθ_cov_sqr,
    t_mean, # observation
    t_cov,
    α_reg)
    

    θ_ref_norm = norm(θ_ref)
    errors = zeros(Float64, N_iter+1)
    errors[1] = norm(trukiobj.θ_bar .- θ_ref)/θ_ref_norm
    
    for i in 1:N_iter
        
        update_ensemble!(trukiobj, ens_func) 
        errors[i+1] = norm(trukiobj.θ_bar .- θ_ref)/θ_ref_norm
    end
    
    return errors
    
end


function EnKI_Run(filter_type, t_mean, t_cov, θ_ref, θ0_bar, θθ0_cov,  G,  N_ens, α_reg::Float64, N_iter::Int64 = 100)
    parameter_names = ["θ"]
    
    
    
    ens_func(θ_ens) = run_linear_ensemble(θ_ens, G)
    
    @info typeof(θθ0_cov)
    
    ekiobj = EnKIObj(filter_type, 
    parameter_names,
    N_ens, 
    θ0_bar,
    θθ0_cov,
    t_mean, # observation
    t_cov,
    α_reg)
    
    errors = zeros(Float64, N_iter+1)
    θ_ref_norm = norm(θ_ref)
    
    θ_bar = dropdims(mean(ekiobj.θ, dims=1), dims=1)
    errors[1] = norm(θ_bar .- θ_ref)/θ_ref_norm

    for i in 1:N_iter
        
        update_ensemble!(ekiobj, ens_func) 
        θ_bar .= dropdims(mean(ekiobj.θ, dims=1), dims=1)
        errors[i+1] = norm(θ_bar .- θ_ref)/θ_ref_norm
        
    end
    
    return errors
    
end


function Linear_Test(problem_type::String, low_rank_prior::Bool = true, nθ::Int64 = 10, N_r::Int64 = 1, α_reg::Float64 = 1.0, N_ite::Int64 = 1000)
    
    
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

    # θ_ref = fill(1.0, nθ)
    # f = G*θ_ref
    
    t_mean = f
    t_cov = Diagonal(fill(1.0^2, nθ))
    
    
    θ0_bar = zeros(Float64, nθ)  # mean 
    
    if !low_rank_prior
        θθ0_cov = Diagonal(fill(10.0^2, nθ))    # covariance
        Z0_cov = Array(Diagonal(fill(10.0, nθ)))       # square root of the covariance
        
    else
        Z0_cov = ones(Float64, nθ, N_r)
        for i = 1:N_r
            Z0_cov[:, i] = 10.0 * sin.(i *  pi*x) 
        end
        
        θθ0_cov = Z0_cov * Z0_cov' + Array(Diagonal(fill(1e-10, nθ))) 
        
    end

    ites = Array(LinRange(1, N_ite+1, N_ite+1))
    errors = zeros(Float64, (4, N_ite+1))


    errors[1, :] = EnKI_Run("EnKI", t_mean, t_cov, θ_ref, θ0_bar, θθ0_cov, G, 2N_r+1, α_reg, N_ite)
    @info "finish EnKI"
    errors[2, :] = EnKI_Run("EAKI", t_mean, t_cov, θ_ref, θ0_bar, θθ0_cov, G, 2N_r+1, α_reg, N_ite)
    @info "finish EAKI"
    errors[3, :] = EnKI_Run("ETKI", t_mean, t_cov, θ_ref, θ0_bar, θθ0_cov, G, 2N_r+1, α_reg, N_ite)
    @info "finish ETKI"
    errors[4, :] = TRUKI_Run(t_mean, t_cov, θ_ref, θ0_bar, Z0_cov, G, N_r, α_reg, N_ite)
    @info "finish TRUKI"

    # Plot
    
    @save "errorslr"*string(low_rank_prior)*".jld"  errors
    

    markevery = max(div(N_ite, 10),1)
    semilogy(ites, errors[1, :], "--o", fillstyle="none", markevery=markevery, label= "EnKI")
    semilogy(ites, errors[2, :], "--o", fillstyle="none", markevery=markevery, label= "EAKI")
    semilogy(ites, errors[3, :], "--o", fillstyle="none", markevery=markevery, label= "ETKI")
    semilogy(ites, errors[4, :], "--o", fillstyle="none", markevery=markevery, label= "TRUKI")
   
    xlabel("Iterations")
    ylabel("\$L_2\$ norm error")
    #ylim((0.1,15))
    grid("on")
    legend()
    tight_layout()
 
    
    savefig("Elliptic-"*string(nθ)*"lr"*string(low_rank_prior)*".pdf")
    close("all")
    
    
end




problem_type = "Elliptic"  

Linear_Test(problem_type, false, 50, 5, 1.0, 1000000)


@info "Finish"


