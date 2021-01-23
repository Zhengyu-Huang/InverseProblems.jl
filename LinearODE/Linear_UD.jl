using JLD2
using Statistics
using LinearAlgebra

include("../Plot.jl")
include("../RExKI.jl")


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


function UKI_Run(t_mean, t_cov, θ_bar, θθ_cov,  G,  N_iter::Int64 = 100,  alpha::Float64 = 1.0)
    parameter_names = ["θ"]
    
    ny, nθ = size(G)
    
    #todo delete
    #θθ_cov =[0.02 0.01; 0.01 0.03]
    ens_func(θ_ens) = run_linear_ensemble(θ_ens, G)
    
    ukiobj = ExKIObj(parameter_names,
    θ_bar, 
    θθ_cov,
    t_mean, # observation
    t_cov,
    alpha)
    
    
    for i in 1:N_iter
        
        params_i = deepcopy(ukiobj.θ_bar[end])
        
        #@info "At iter ", i, " θ: ", params_i
        
        update_ensemble!(ukiobj, ens_func) 
        
    end
    
    return ukiobj
    
end


function Linear_Test(alpha::Float64, θ0_bar::Array{Float64,1}, N_ite::Int64 = 100)
    γ = 2.0
    
    nθ = 2
    θθ0_cov = Array(Diagonal(fill(0.5^2, nθ)))     # standard deviation
    
    
    # over-determined case
    
    # under-determined case
    t_mean = [3.0;]
    
    t_cov = Array(Diagonal(fill(0.1^2, size(t_mean) )))
    G = [1.0 2.0;]
    
    ukiobj = UKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov, G, N_ite, alpha)
    
    
    
    
    return ukiobj
end


# mission : "2params" "Hilbert"
mission = "2params"

fig, ax = PyPlot.subplots(ncols=3, sharex=false, sharey=true, figsize=(18,5))


@info "need to update RExKI file"
# +    r = exki.θ_bar[1]
 
# -    θ_p_bar  = α_reg*θ_bar + (1-α_reg)*exki.θ_bar[1]
# +    r = [0; 0]
# +
# +    θ_p_bar  = α_reg*θ_bar + (1-α_reg)*r

if mission == "2params"
    alphas = [0.1; 0.5; 0.9;]
    for i = 1:3
        alpha = alphas[i]
        N_ite = 50
        if alpha  == 0.9
            N_ite = 500
        end
        
        ukiobj_1 = Linear_Test(alpha, [0.0;   0.0],   N_ite)
        ukiobj_2 = Linear_Test(alpha, [1.0;   1.0],   N_ite)
        ukiobj_3 = Linear_Test(alpha, [1.0;  -1.0],  N_ite)
        
        θ_ref = ukiobj_1.θ_bar[end]
        @info "alpha is ", alpha,  " θ_ref is ", θ_ref 
        # Plot θ error
        
        ites = Array(LinRange(1, N_ite+1, N_ite+1))
        errors = zeros(Float64, (3, N_ite+1))
        for i = 1:N_ite+1
            errors[1, i] = norm(ukiobj_1.θ_bar[i] .- θ_ref)
            errors[2, i] = norm(ukiobj_2.θ_bar[i] .- θ_ref)
            errors[3, i] = norm(ukiobj_3.θ_bar[i] .- θ_ref)
        end
        
        markevery = div(N_ite, 10)
        ax[i].semilogy(ites, errors[1, :], "--o", fillstyle="none", color="C1", markevery=markevery, label=L"m_0 = [0 \, \,0]^T")
        # semilogy(ites, errors[2, :], "--o", fillstyle="none", label="NS (update)")
        ax[i].semilogy(ites, errors[2, :], "--s", fillstyle="none", color="C2", markevery=markevery, label=L"m_0 = [1 \, \, 1]^T")
        # semilogy(ites, errors[6, :], "--o", fillstyle="none", label="OD (update)")
        ax[i].semilogy(ites, errors[3, :], "--^", fillstyle="none", color="C3", markevery=markevery, label=L"m_0 = [1 -1]^T")
        # semilogy(ites, errors[4, :], "--o", fillstyle="none", label="UD (update)")
        
        ax[i].set_xlabel("Iterations")
        
        ax[i].grid("on")
        ax[i].legend()
        
    end
    ax[1].set_ylabel("\$L_2\$ norm error")
    fig.tight_layout()
    fig.savefig("Linear-para-alpha.pdf")
    close(fig)
    
end

@info "finished"




