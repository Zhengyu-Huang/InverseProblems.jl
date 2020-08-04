using JLD2
using Statistics
using LinearAlgebra

include("../Plot.jl")
include("../RUKI.jl")
include("../RExKI.jl")
include("../REKI.jl")

function run_linear_ensemble(params_i, G, Δobs)
    
    N_ens,  N_θ = size(params_i)
    n_data = div(size(G,1), Δobs)
    g_ens = zeros(Float64, N_ens, n_data)
    
    for i = 1:N_ens
        θ = params_i[i, :]
        # g: N_ens x N_data
        g_ens[i, :] .= (G*θ)[1:Δobs:end] 
    end
    
    return g_ens
end


function RUKI_Run(t_mean, t_cov, θ_bar, θθ_cov,  G,  Δobs, α_reg::Float64, N_iter::Int64 = 100)
    parameter_names = ["θ"]
    ny, nθ = size(G)
    ens_func(θ_ens) = run_linear_ensemble(θ_ens, G, Δobs)
    
    ukiobj = UKIObj(parameter_names,
    θ_bar, 
    θθ_cov,
    t_mean, # observation
    t_cov,
    α_reg)
    
    for i in 1:N_iter
        params_i = deepcopy(ukiobj.θ_bar[end])
        update_ensemble!(ukiobj, ens_func) 
    end
    
    return ukiobj
end

function RExKI_Run(t_mean, t_cov, θ_bar, θθ_cov,  G,  Δobs, α_reg::Float64, N_iter::Int64 = 100)
    parameter_names = ["θ"]
    ny, nθ = size(G)
    ens_func(θ_ens) = run_linear_ensemble(θ_ens, G, Δobs)
    
    exkiobj = ExKIObj(parameter_names,
    θ_bar, 
    θθ_cov,
    t_mean, # observation
    t_cov,
    α_reg)
    
    for i in 1:N_iter
        params_i = deepcopy(exkiobj.θ_bar[end])
        update_ensemble!(exkiobj, ens_func) 
    end
    
    return exkiobj
end


function REKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov,  G,  Δobs, N_ens,  α_reg::Float64, N_iter::Int64 = 100)
    parameter_names = ["θ"]
    
    ny, nθ = size(G)
    
    ens_func(θ_ens) = run_linear_ensemble(θ_ens, G, Δobs)
    
    priors = [Distributions.Normal(θ0_bar[i], sqrt(θθ0_cov[i,i])) for i=1:nθ]
    
    initial_params = construct_initial_ensemble(N_ens, priors; rng_seed=6)
    
    ekiobj = EKIObj(parameter_names,
    initial_params, 
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

function Hilbert_Test(method::String, nθ::Int64 = 10, Δobs::Int64 = 2, N_ite::Int64 = 1000, N_ens::Int64 = 20,
    α_reg::Float64 = 1.0, noise_level::Float64 = -1.0, ax1 = nothing, ax2 = nothing)
    
    θ0_bar = zeros(Float64, nθ)  # mean 
    θθ0_cov = Array(Diagonal(fill(0.5^2, nθ)))     # standard deviation
    
    G = zeros(nθ, nθ)
    for i = 1:nθ
        for j = 1:nθ
            G[i,j] = 1/(i + j - 1)
        end
    end
    
    θ_ref = fill(1.0, nθ)
    t_mean = (G*θ_ref)[1:Δobs:end]
    
    if noise_level > 0.0
        noise = similar(t_mean)
        Random.seed!(42);
        for i = 1:length(t_mean)
            noise[i] = rand(Normal(0, noise_level*abs(t_mean[i])))
        end
        t_mean .+= noise
    end
    
    t_cov = Array(Diagonal(fill(0.5^2, size(t_mean, 1))))
    
    
    
    # Plot
    ites = Array(LinRange(1, N_ite, N_ite))
    errors = zeros(Float64, (2,N_ite))
    
    
    if method =="ExKI"
        exkiobj = RExKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov, G, Δobs, α_reg, N_ite)
        label = "UKI+"
        
        for i = 1:N_ite
            errors[1, i] = norm(exkiobj.θ_bar[i] .- θ_ref)/norm(θ_ref)
            errors[2, i] = 0.5*(exkiobj.g_bar[i] - t_mean)'*(t_cov\(exkiobj.g_bar[i] - t_mean))
        end
        linestyle, marker = "-", "o"
        @info norm(exkiobj.θθ_cov[end])
    elseif method == "EnKI"
        label = "EnKI"
        enkiobj = REKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov, G, Δobs, N_ens, α_reg, N_ite)
        θ_bar = [dropdims(mean(enkiobj.θ[i], dims=1), dims=1) for i = 1:N_ite]
        
        for i = 1:N_ite
            errors[1, i] = norm(θ_bar[i] .- θ_ref)/norm(θ_ref)
            errors[2, i] = 0.5*(enkiobj.g_bar[i] - t_mean)'*(t_cov\(enkiobj.g_bar[i] - t_mean))
        end
        linestyle, marker = "--", "s"
    else
        error("method: ", method,  " is not recognized.")
    end
    
    label = label*" (α = "*string(α_reg)*")"
    
    
    if (!isnothing(ax1)  &&  !isnothing(ax2))
        ax1.semilogy(ites, errors[1, :], linestyle=linestyle, marker=marker, fillstyle="none", markevery=50, label= label)
        ax1.set_ylabel("Relative L₂ norm error")
        ax1.grid(true)
        
        
        ax2.semilogy(ites, errors[2, :], linestyle=linestyle, marker = marker,fillstyle="none", markevery=50, label= label)
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("Optimization error")
        ax2.grid(true)
        ax2.legend()
    end
    
    return (method =="ExKI" ? exkiobj : enkiobj)
    
    
end


function Compare()
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    mysize = 20
    font0 = Dict(
    "font.size" => mysize,
    "axes.labelsize" => mysize,
    "xtick.labelsize" => mysize,
    "ytick.labelsize" => mysize,
    "legend.fontsize" => mysize,
    )
    merge!(rcParams, font0)


    Nθ = 10
    Δobs = 2
    N_ite = 1000
    
    noise_level = 0.05
    N_ens = 50
    fig, (ax1, ax2) = PyPlot.subplots(nrows=2, sharex=true, figsize=(8,12))
    α_reg = 1.0
    Hilbert_Test("EnKI", Nθ, Δobs, N_ite, N_ens, α_reg, noise_level, ax1, ax2)
    Hilbert_Test("ExKI", Nθ, Δobs, N_ite, N_ens, α_reg, noise_level, ax1, ax2)
    
    α_reg = 0.9
    Hilbert_Test("EnKI", Nθ, Δobs, N_ite, N_ens, α_reg, noise_level, ax1, ax2)
    Hilbert_Test("ExKI", Nθ, Δobs, N_ite, N_ens, α_reg, noise_level, ax1, ax2)
    
    α_reg = 0.5
    Hilbert_Test("EnKI", Nθ, Δobs, N_ite, N_ens, α_reg, noise_level, ax1, ax2)
    Hilbert_Test("ExKI", Nθ, Δobs, N_ite, N_ens, α_reg, noise_level, ax1, ax2)
    
    fig.tight_layout()
    savefig("Figs/Hilbert-noise-5.pdf")
    close("all")
    
    
    noise_level = 0.01
    fig, (ax1, ax2) = PyPlot.subplots(nrows=2, sharex=true, figsize=(8,12))
    α_reg = 1.0
    Hilbert_Test("EnKI", Nθ, Δobs, N_ite, N_ens, α_reg, noise_level, ax1, ax2)
    Hilbert_Test("ExKI", Nθ, Δobs, N_ite, N_ens, α_reg, noise_level, ax1, ax2)
    
    α_reg = 0.9
    Hilbert_Test("EnKI", Nθ, Δobs, N_ite, N_ens, α_reg, noise_level, ax1, ax2)
    Hilbert_Test("ExKI", Nθ, Δobs, N_ite, N_ens, α_reg, noise_level, ax1, ax2)
    
    α_reg = 0.5
    Hilbert_Test("EnKI", Nθ, Δobs, N_ite, N_ens, α_reg, noise_level, ax1, ax2)
    Hilbert_Test("ExKI", Nθ, Δobs, N_ite, N_ens, α_reg, noise_level, ax1, ax2)
    
    fig.tight_layout()
    savefig("Figs/Hilbert-noise-1.pdf")
    close("all")
    
    noise_level = -1.0
    fig, (ax1, ax2) = PyPlot.subplots(nrows=2, sharex=true, figsize=(8,12))
    α_reg = 1.0
    Hilbert_Test("EnKI", Nθ, Δobs, N_ite, N_ens, α_reg, noise_level, ax1, ax2)
    Hilbert_Test("ExKI", Nθ, Δobs, N_ite, N_ens, α_reg, noise_level, ax1, ax2)
    
    α_reg = 0.9
    Hilbert_Test("EnKI", Nθ, Δobs, N_ite, N_ens, α_reg, noise_level, ax1, ax2)
    Hilbert_Test("ExKI", Nθ, Δobs, N_ite, N_ens, α_reg, noise_level, ax1, ax2)
    
    α_reg = 0.5
    Hilbert_Test("EnKI", Nθ, Δobs, N_ite, N_ens, α_reg, noise_level, ax1, ax2)
    Hilbert_Test("ExKI", Nθ, Δobs, N_ite, N_ens, α_reg, noise_level, ax1, ax2)
    
    fig.tight_layout()
    savefig("Figs/Hilbert-noise-0.pdf")
    close("all")
end


function Study_Cov()
    
    Nθ = 10
    Δobs = 2
    N_ite = 1000
    
    noise_level = 0.05
    N_ens = 50
    α_reg = 1.0
    exkiobj1 = Hilbert_Test("ExKI", Nθ, Δobs, N_ite, N_ens, α_reg, noise_level)
    
    α_reg = 0.9
    exkiobj2 = Hilbert_Test("ExKI", Nθ, Δobs, N_ite, N_ens, α_reg, noise_level)
  
    α_reg = 0.5
    exkiobj3 = Hilbert_Test("ExKI", Nθ, Δobs, N_ite, N_ens, α_reg, noise_level)
    
    # Plot
    ites = Array(LinRange(1, N_ite, N_ite))
    Fro_norm = zeros(Float64, (3,N_ite))
    for i = 1:N_ite 
        Fro_norm[:, i] .= norm(exkiobj1.θθ_cov[i]), norm(exkiobj2.θθ_cov[i]), norm(exkiobj3.θθ_cov[i])
    end

    semilogy(ites, Fro_norm[1, :], "--o",fillstyle="none", markevery=50, label= "α = 1.0")
    semilogy(ites, Fro_norm[2, :], "--o",fillstyle="none", markevery=50, label= "α = 0.9")
    semilogy(ites, Fro_norm[3, :], "--o",fillstyle="none", markevery=50, label= "α = 0.5")
    xlabel("Iterations")
    ylabel("Frobenius norm")
    tight_layout()
    legend()
    grid("on")
    savefig("Figs/Hilbert-cov.pdf")
    close("all")
    
end

Compare()
Study_Cov()