using JLD2
using Statistics
using LinearAlgebra

include("../Plot.jl")
include("../UKI.jl")
include("../EKI.jl")

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


function UKI_Run(t_mean, t_cov, θ_bar, θθ_cov,  G,  N_iter::Int64 = 100,  update_cov::Int64 = 0)
    parameter_names = ["θ"]
    
    ny, nθ = size(G)
    
    #todo delete
    #θθ_cov =[0.02 0.01; 0.01 0.03]
    ens_func(θ_ens) = run_linear_ensemble(θ_ens, G)
    
    ukiobj = UKIObj(parameter_names,
    θ_bar, 
    θθ_cov,
    t_mean, # observation
    t_cov)
    
    
    for i in 1:N_iter

        params_i = deepcopy(ukiobj.θ_bar[end])
        
        #@info "At iter ", i, " θ: ", params_i
        
        update_ensemble!(ukiobj, ens_func) 
        
        if (update_cov) > 0 && (i%update_cov == 0) 
            reset_θθ0_cov!(ukiobj)
        end
        
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

function Linear_Test(update_cov::Int64 = 0, case::String = "square", N_ite::Int64 = 100)
    γ = 2.0
    
    nθ = 2
    θ0_bar = zeros(Float64, nθ)  # mean 
    θθ0_cov = Array(Diagonal(fill(0.5^2, nθ)))     # standard deviation
    
    if case == "square"
        # Square matrix case
        t_mean = [3.0;7.0]
        
        t_cov = Array(Diagonal(fill(0.1^2, size(t_mean) )))
        G = [1.0 2.0; 3.0 4.0]
        
        
        ukiobj = UKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov, G, N_ite, update_cov)
        θθ_cov_opt_inv = G'*inv(t_cov)*G
        if (update_cov == 0)
            @info "suboptimal error :", inv(ukiobj.θθ_cov[end]) - 1/γ*θθ_cov_opt_inv - (γ-1)/γ*inv((γ-1)/γ*ukiobj.θθ_cov[end] + 1/γ*ukiobj.θθ_cov[1])
        else
            @info "optimal error : ", inv(θθ_cov_opt_inv) - ukiobj.θθ_cov[end]
        end
        
    end
    
    if (case == "under-determined")
        # under-determined case
        t_mean = [3.0;]
        
        t_cov = Array(Diagonal(fill(0.1^2, size(t_mean) )))
        G = [1.0 2.0;]
        
        ukiobj = UKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov, G, N_ite, update_cov)
        
        
    end
    
    if (case == "over-determined")
        # over-determined case
        t_mean = [3.0;7.0;10.0]
        
        t_cov = Array(Diagonal(fill(0.1^2, size(t_mean) )))
        G = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        
        ukiobj = UKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov, G, N_ite, update_cov)
        
        
        θθ_cov_opt_inv = G'*inv(t_cov)*G
        
        if (update_cov == 0)
            @info "suboptimal error :", inv(ukiobj.θθ_cov[end]) - 1/γ*θθ_cov_opt_inv - (γ-1)/γ*inv((γ-1)/γ*ukiobj.θθ_cov[end] + 1/γ*ukiobj.θθ_cov[1])
        else
            @info "optimal error : ", inv(θθ_cov_opt_inv) - ukiobj.θθ_cov[end]
        end
        
    end
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
    
    ukiobj = UKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov, G, N_ite)
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
 
    
    semilogy(ites, errors[1, :], "--o", fillstyle="none", markevery=50, label= "UKI")
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
# mission = "2params"
  mission = "Hilbert"
if mission == "2params"
    ukiobj_ssub = Linear_Test(0, "square", 10000)
    ukiobj_sopt = Linear_Test(5, "square", 10000)
    ukiobj_sopt.θθ_cov[1] = Array(Diagonal(fill(0.5^2, 2 )))
    θ_s = [1;1]
    
    ukiobj_usub = Linear_Test(0, "under-determined", 50)
    ukiobj_uopt = Linear_Test(5, "under-determined", 50)
    ukiobj_uopt.θθ_cov[1] = Array(Diagonal(fill(0.5^2, 2 )))
    θ_u = [0.6, 1.2]
    
    ukiobj_osub = Linear_Test(0, "over-determined", 10000)
    ukiobj_oopt = Linear_Test(5, "over-determined", 10000)
    ukiobj_oopt.θθ_cov[1] = Array(Diagonal(fill(0.5^2, 2 )))
    θ_o = [1/3, 8.5/6]

    # Plot θ error
    N_ite = 50
    ites = Array(LinRange(1, N_ite+1, N_ite+1))
    errors = zeros(Float64, (6, N_ite+1))
    for i = 1:N_ite+1
        errors[1, i] = norm(ukiobj_ssub.θ_bar[i] .- θ_s)
        errors[2, i] = norm(ukiobj_sopt.θ_bar[i] .- θ_s)
        errors[3, i] = norm(ukiobj_usub.θ_bar[i] .- θ_u)
        errors[4, i] = norm(ukiobj_uopt.θ_bar[i] .- θ_u)
        errors[5, i] = norm(ukiobj_osub.θ_bar[i] .- θ_o)
        errors[6, i] = norm(ukiobj_oopt.θ_bar[i] .- θ_o)
    end
    
    
    semilogy(ites, errors[1, :], "--o", fillstyle="none", label="NS")
    semilogy(ites, errors[2, :], "--o", fillstyle="none", label="NS (update)")
    semilogy(ites, errors[3, :], "--o", fillstyle="none", label="UD")
    semilogy(ites, errors[4, :], "--o", fillstyle="none", label="UD (update)")
    semilogy(ites, errors[5, :], "--o", fillstyle="none", label="OD")
    semilogy(ites, errors[6, :], "--o", fillstyle="none", label="OD (update)")
    xlabel("Iterations")
    ylabel("\$L_2\$ norm error")
    grid("on")
    legend(bbox_to_anchor=(1.1, 1.05))
    tight_layout()
    savefig("Linear-para.pdf")
    close("all")
    

    # Plot θθ_cov error
    N_ite = 50
    ites = Array(LinRange(1, N_ite+1, N_ite+1))
    errors = zeros(Float64, (6, N_ite+1))
    for i = 1:N_ite+1
        errors[1, i] = norm(ukiobj_ssub.θθ_cov[i] .- ukiobj_ssub.θθ_cov[end])
        errors[2, i] = norm(ukiobj_sopt.θθ_cov[i] .- ukiobj_sopt.θθ_cov[end])
        errors[3, i] = norm(ukiobj_usub.θθ_cov[i] .- ukiobj_usub.θθ_cov[end])
        errors[4, i] = norm(ukiobj_uopt.θθ_cov[i] .- ukiobj_uopt.θθ_cov[end])
        errors[5, i] = norm(ukiobj_osub.θθ_cov[i] .- ukiobj_osub.θθ_cov[end])
        errors[6, i] = norm(ukiobj_oopt.θθ_cov[i] .- ukiobj_oopt.θθ_cov[end])
    end
    
    
    semilogy(ites, errors[1, :], "--o", fillstyle="none", label="NS")
    semilogy(ites, errors[2, :], "--o", fillstyle="none", label="NS (update)")
    semilogy(ites, errors[5, :], "--o", fillstyle="none", label="OD")
    semilogy(ites, errors[6, :], "--o", fillstyle="none", label="OD (update)")
    xlabel("Iterations")
    ylabel("Frobenius norm error")
    grid("on")
    legend()
    tight_layout()
    savefig("Linear-Sigma.pdf")

    close("all")


    # Plot θθ_cov diverge
    N_ite = 50
    ites = Array(LinRange(1, N_ite+1, N_ite+1))
    errors = zeros(Float64, (6, N_ite+1))
    for i = 1:N_ite+1
        errors[1, i] = norm(ukiobj_ssub.θθ_cov[i])
        errors[2, i] = norm(ukiobj_sopt.θθ_cov[i])
        errors[3, i] = norm(ukiobj_usub.θθ_cov[i])
        errors[4, i] = norm(ukiobj_uopt.θθ_cov[i])
        errors[5, i] = norm(ukiobj_osub.θθ_cov[i])
        errors[6, i] = norm(ukiobj_oopt.θθ_cov[i])
    end
    
    
    semilogy(ites, errors[3, :], "--o", fillstyle="none", label="UD")
    semilogy(ites, errors[4, :], "--o", fillstyle="none", label="UD (update)")
    xlabel("Iterations")
    ylabel("Frobenius norm error")
    grid("on")
    legend()
    tight_layout()
    savefig("Linear-Under-Sigma.pdf")
    close("all")
    
    
else
    
    Hilbert_Test(10,  1000)
    Hilbert_Test(100, 1000)
end

@info "finished"




