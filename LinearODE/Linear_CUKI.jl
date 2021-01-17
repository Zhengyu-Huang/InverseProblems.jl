using JLD2
using Statistics
using LinearAlgebra

include("../Plot.jl")
include("../RExKI.jl")
include("../UKI-Variants.jl")

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


function UKI_Run(algorithm, t_mean, t_cov, θ_bar, θθ_cov,  G,  N_iter::Int64 = 100)
    parameter_names = ["θ"]
    
    ny, nθ = size(G)
    
    #todo delete
    #θθ_cov =[0.02 0.01; 0.01 0.03]
    ens_func(θ_ens) = run_linear_ensemble(θ_ens, G)
    
    if algorithm == "UKI"
        ukiobj = ExKIObj(parameter_names,
        θ_bar, 
        θθ_cov,
        t_mean, # observation
        t_cov,
        1.0)
    elseif algorithm == "UKICTL" || algorithm == "UKS"
        UKF_modify = true
        λ_reg = 0.0
        Δt0 = 10.0/N_iter

        ukiobj = CUKIObj(algorithm, parameter_names,
        θ_bar, 
        θθ_cov,
        t_mean, # observation
        t_cov,
        UKF_modify,
        λ_reg,
        Δt0)
        
        
        
    else 
        error("unrecognized algorithm : ", algorithm)
    end
    
    
    for i in 1:N_iter
        
        params_i = deepcopy(ukiobj.θ_bar[end])
        
        update_ensemble!(ukiobj, ens_func) 
        @info i , " / ", N_iter
        
    end
    
    return ukiobj
    
end


function EKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov,  G,  N_ens, N_iter::Int64 = 100)
    parameter_names = ["θ"]
    
    ny, nθ = size(G)
    
    ens_func(θ_ens) = run_linear_ensemble(θ_ens, G)
    
    priors = [Distributions.Normal(θ0_bar[i], sqrt(θθ0_cov[i,i])) for i=1:nθ]
    
    initial_params = construct_initial_ensemble(N_ens, priors; rng_seed=6)
    
    
    ekiobj = EKIObj(parameter_names,
    initial_params, 
    θθ0_cov,
    t_mean, # observation
    t_cov)
    
    
    for i in 1:N_iter
        
        update_ensemble!(ekiobj, ens_func) 
        
    end
    
    return ekiobj
    
end

function Linear_Test(algorithm::String, case::String = "square", N_ite::Int64 = 100)
    γ = 2.0
    
    nθ = 2
    θ0_bar = zeros(Float64, nθ)  # mean 
    θθ0_cov = Array(Diagonal(fill(0.5^2, nθ)))     # standard deviation
    if algorithm == "UKICTL" || algorithm == "UKS"
        θθ0_cov = Array(Diagonal(fill(1.0^2, nθ)))
    end

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

    ukiobj = UKI_Run(algorithm, t_mean, t_cov, θ0_bar, θθ0_cov, G, N_ite)

    return ukiobj
end

function Hilbert_Test(nθ::Int64 = 10, N_ite::Int64 = 1000)
    
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
    
    ukiobj = CUKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov, G, N_ite)
    ekiobj_1 = EKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov, G, 2nθ+1, N_ite)
    ekiobj_2 = EKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov, G, 100nθ+1, N_ite)
    
    
    # Plot
    ites = Array(LinRange(1, N_ite+1, N_ite+1))
    errors = zeros(Float64, (3,N_ite+1))
    for i = 1:N_ite+1
        errors[1, i] = norm(ukiobj.θ_bar[i] .- 1.0)
        
        θ_bar = dropdims(mean(ekiobj_1.θ[i], dims=1), dims=1)
        errors[2, i] = norm(θ_bar .- 1.0)
        
        θ_bar = dropdims(mean(ekiobj_2.θ[i], dims=1), dims=1)
        errors[3, i] = norm(θ_bar .- 1.0)
    end
    
    
    semilogy(ites, errors[1, :], "--o", fillstyle="none", markevery=50, label= "CUKI")
    semilogy(ites, errors[2, :], "--o", fillstyle="none", markevery=50, label= "EnKI (\$J=2N_{θ}+1)\$")
    semilogy(ites, errors[3, :], "--o", fillstyle="none", markevery=50, label= "EnKI (\$J=100N_{θ}+1)\$")
    xlabel("Iterations")
    ylabel("\$L_2\$ norm error")
    #ylim((0.1,15))
    grid("on")
    legend()
    tight_layout()
    
    savefig("Hilbert-"*string(nθ)*".pdf")
    close("all")
    
    return ukiobj, ekiobj_1, ekiobj_2
end

# mission : "2params" "Hilbert"
mission = "2params"

if mission == "2params"
    
    algorithms = ["UKICTL", "UKS", "UKI"]
    tests = ["square", "under-determined", "over-determined"]
    θ_ref = Dict()
    θ_ref["square"] = [1;1]
    θ_ref["under-determined"] = [0.6, 1.2]
    θ_ref["over-determined"] = [1/3, 8.5/6]
    
    # Linear_Test("UKICTL", "over-determined", 10000)
    # error("stop")
    N_iter = 100000
    ukiobj = Dict()
    for algorithm in algorithms
        for test in tests
            @info "Start : ", algorithm, test
            ukiobj[algorithm, test] = Linear_Test(algorithm, test, N_iter)
        end
    end
    
    
    # Plot θ error

    ites = Array(LinRange(1, N_iter, N_iter))
    errors = Dict()
    for algorithm in algorithms
        for test in tests
            errors[algorithm, test] = zeros(Float64, N_iter)
            for i = 1:N_iter
                errors[algorithm, test][i] = norm(ukiobj[algorithm, test].θ_bar[i] .- θ_ref[test])
            end
        end
    end

    
    
    semilogy(ites, errors["UKI", "square"], "--", fillstyle="none", label="UKI NS")
    semilogy(ites, errors["UKICTL", "square"], "--", fillstyle="none", label="UKICTL NS")
    semilogy(ites, errors["UKS", "square"], "--", fillstyle="none", label="UKS NS")
    semilogy(ites, errors["UKI", "under-determined"], "--", fillstyle="none", label="UKI UD")
    semilogy(ites, errors["UKICTL", "under-determined"], "--", fillstyle="none", label="UKICTL UD")
    semilogy(ites, errors["UKS", "under-determined"], "--", fillstyle="none", label="UKS UD")
    semilogy(ites, errors["UKI", "over-determined"], "--", fillstyle="none", label="UKI OD")
    semilogy(ites, errors["UKICTL", "over-determined"], "--", fillstyle="none", label="UKICTL OD")
    semilogy(ites, errors["UKS", "over-determined"], "--", fillstyle="none", label="UKS OD")
    xlabel("Iterations")
    ylabel("\$L_2\$ norm error")
    grid("on")
    legend(bbox_to_anchor=(1.1, 1.05))
    tight_layout()
    savefig("Linear-para.pdf")
    close("all")
    
    
    # # Plot θθ_cov error
    # N_ite = 50
    # ites = Array(LinRange(1, N_ite+1, N_ite+1))
    # errors = zeros(Float64, (6, N_ite+1))
    # for i = 1:N_ite+1
    #     errors[1, i] = norm(ukiobj_ssub.θθ_cov[i] .- ukiobj_ssub.θθ_cov[end])
    #     errors[2, i] = norm(ukiobj_sopt.θθ_cov[i] .- ukiobj_sopt.θθ_cov[end])
    #     errors[3, i] = norm(ukiobj_usub.θθ_cov[i] .- ukiobj_usub.θθ_cov[end])
    #     errors[4, i] = norm(ukiobj_uopt.θθ_cov[i] .- ukiobj_uopt.θθ_cov[end])
    #     errors[5, i] = norm(ukiobj_osub.θθ_cov[i] .- ukiobj_osub.θθ_cov[end])
    #     errors[6, i] = norm(ukiobj_oopt.θθ_cov[i] .- ukiobj_oopt.θθ_cov[end])
    # end
    
    
    # semilogy(ites, errors[1, :], "--o", fillstyle="none", label="NS")
    # semilogy(ites, errors[2, :], "--o", fillstyle="none", label="NS (update)")
    # semilogy(ites, errors[5, :], "--o", fillstyle="none", label="OD")
    # semilogy(ites, errors[6, :], "--o", fillstyle="none", label="OD (update)")
    # xlabel("Iterations")
    # ylabel("Frobenius norm error")
    # grid("on")
    # legend()
    # tight_layout()
    # savefig("Linear-Sigma.pdf")
    
    # close("all")
    
    
    # # Plot θθ_cov diverge
    # N_ite = 50
    # ites = Array(LinRange(1, N_ite+1, N_ite+1))
    # errors = zeros(Float64, (6, N_ite+1))
    # for i = 1:N_ite+1
    #     errors[1, i] = norm(ukiobj_ssub.θθ_cov[i])
    #     errors[2, i] = norm(ukiobj_sopt.θθ_cov[i])
    #     errors[3, i] = norm(ukiobj_usub.θθ_cov[i])
    #     errors[4, i] = norm(ukiobj_uopt.θθ_cov[i])
    #     errors[5, i] = norm(ukiobj_osub.θθ_cov[i])
    #     errors[6, i] = norm(ukiobj_oopt.θθ_cov[i])
    # end
    
    
    # semilogy(ites, errors[3, :], "--o", fillstyle="none", label="UD")
    # semilogy(ites, errors[4, :], "--o", fillstyle="none", label="UD (update)")
    # xlabel("Iterations")
    # ylabel("Frobenius norm error")
    # grid("on")
    # legend()
    # tight_layout()
    # savefig("Linear-Under-Sigma.pdf")
    # close("all")
    
    
else
    
    Hilbert_Test(10,  1000)
    Hilbert_Test(100, 1000)
end

@info "finished"




