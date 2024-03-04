using LinearAlgebra
using PyPlot
include("../../Inversion/IPS.jl")
include("../../Inversion/Plot.jl")
include("../../Inversion/Utility.jl")
function residual(method_type::String, m_oo, C_oo, m, C)
    if method_type == "gradient_descent"
        # under-determined case
        return -C_oo\(m - m_oo), 1/2.0*(inv(C) - inv(C_oo))
        
    elseif method_type == "natural_gradient_descent"
        # over-determined case
        return -(m - m_oo), C - C*(C_oo\C) 
        
    elseif method_type == "wasserstein_gradient_descent"
        # over-determined case
        return -C_oo\(m - m_oo), 2I - C_oo\C - C/C_oo 

    else
        error("Problem type : ", problem_type, " has not implemented!")
    end
end

function Continuous_Dynamics(method_type::String, m_oo, C_oo, m_0, C_0, Δt, N_t)

    N_θ = length(m_0)
    m = zeros(N_t+1, N_θ)
    C = zeros(N_t+1, N_θ, N_θ)
    
    m[1, :] = m_0
    C[1, :, :] = C_0

    for i = 1:N_t
        dm, dC = residual(method_type, m_oo, C_oo, m[i, :], C[i, :, :])
        m[i+1, :]    = m[i, :] + dm * Δt
        C[i+1, :, :] = C[i, :, :] + dC * Δt
    end

    return m, C
end

function KL_estimator(m_1,  C_1, m_2 , C_2)
    d = length(m_1)
    temp = C_2\C_1
    return 1/2*(  -log(det(temp)) - d + tr(temp) + (m_2 - m_1)'*(C_2\(m_2 - m_1)) )
end



Random.seed!(42)
######################### TEST #######################################
mutable struct Setup_Param{IT<:Int, VEC, MAT}
    N_θ::IT
    m_oo::VEC
    C_oo::MAT
end

function compute_∇logρ(s_param::Setup_Param, θ::Array{FT, 1}) where {FT<:AbstractFloat}
    C_oo, m_oo = s_param.C_oo, s_param.m_oo
    logρ  = -1/2*(θ - m_oo)'*(C_oo\(θ - m_oo))
    ∇logρ = -C_oo\(θ - m_oo)
    # ∇²Φ = C_oo
    return logρ, ∇logρ
end


N_θ = 2
m_0 = [10.0; 10.0]
C_0 = [1/2 0; 0 2.0]



ϵs = [0.01, 0.1, 1]
fig, ax = PyPlot.subplots(ncols=3, nrows=3, sharex=true, sharey="row", figsize=(12,12))
fig_cont, ax_cont = PyPlot.subplots(ncols=3, nrows=4, sharex="col", sharey=false, figsize=(12,12))

N_ens = 1000
Δt = 0.001
N_t = 15000
ts = LinRange(0, Δt*N_t, N_t+1)
θ0 = Array(rand(MvNormal(m_0, C_0), N_ens)')
ω =  rand(Normal(0, 1), (20, 2))
b = rand(Uniform(0, 2*pi), 20)

N_x, N_y = 1000, 1000
X_cont, Y_cont, Z_cont = zeros(N_x, N_y), zeros(N_x, N_y), zeros(N_x, N_y)
for test_id = 1:length(ϵs)
    ϵ = ϵs[test_id]
    m_oo = [0.0; 0.0]
    C_oo = [1.0 0.0; 0.0 1/ϵ]

    s_param = Setup_Param(N_θ, m_oo, C_oo)
    
    cos_ref = cos_error_estimation_particle(m_oo, C_oo, ω, b ) 
    
    logρ(θ) = -0.5*θ'*(C_oo\θ)
    x_min, x_max = -3, 3
    y_min, y_max = -3/sqrt(ϵ), 3/sqrt(ϵ)
    
    xx_cont = Array(LinRange(x_min, x_max, N_x))
    yy_cont = Array(LinRange(y_min, y_max, N_y))
#     X_cont .= repeat(xx_cont, 1, N_y) 
#     Y_cont .= repeat(yy_cont, 1, N_x)'
#     Z_cont = zeros(N_x, N_y)
    
    for i = 1:N_x
        for j = 1:N_y
            X_cont[i,j] = xx_cont[i]
            Y_cont[i,j] = yy_cont[j]
            Z_cont[i, j] = logρ( [X_cont[i,j], Y_cont[i,j]] )
        end
    end
    Z_cont .= exp.(Z_cont)
    for i = 1:4
        ax_cont[i, test_id].contour(X_cont, Y_cont, Z_cont, 5, colors="grey", alpha=0.5)
    end

    for method in ["Wasserstein", "Stein"]
        for preconditioner in [false, true]
    
            ips_obj = IPS_Run(s_param, compute_∇logρ, θ0, N_ens, Δt, N_t, method, preconditioner)

            ips_errors    = zeros(N_t+1, 3)
            for i = 1:N_t+1
                m_i, C_i = dropdims(mean(ips_obj.θ[i], dims=1), dims=1), construct_cov(ips_obj.θ[i])
                ips_errors[i, 1] = norm(m_i .- m_oo)
                ips_errors[i, 2] = norm(C_i .- C_oo)/norm(C_oo)
                ips_errors[i, 3] = norm(cos_ref - cos_error_estimation_particle(m_i, C_i, ω, b ))/sqrt(length(b))
            end
            @info method, " preconditioner ", preconditioner
            @info L"\lambdaean = ", dropdims(mean(ips_obj.θ[end], dims=1), dims=1), " ", m_oo
            @info "cov = ", construct_cov(ips_obj.θ[end]), " ", C_oo
            
            

            ites = Array(0:N_t)
            markevery = div(N_t, 10)
            label, color, marker = "Fisher-Rao", "C2", "o"
            if (method  == "Wasserstein" || method  == "Stein")
                label = (preconditioner ? "Affine invariant "*method : method) * " GF"
                color = (method  == "Stein" ? "C0" : "C1")
                if method  ==  "Wasserstein" && ~preconditioner
                    color  = "C2"
                end
                if method  ==  "Stein" && ~preconditioner
                    color  = "C3"
                end
                
                marker = (method  == "Stein" ? "*" : "s")
            end
            linestyle = preconditioner ?  "solid" : "dotted"
            
            
            ax[1, test_id].semilogy(ts, ips_errors[:, 1], linestyle=linestyle, color=color, marker=marker, fillstyle="none", label=label, markevery = markevery)
            ax[2, test_id].semilogy(ts, ips_errors[:, 2], linestyle=linestyle, color=color, marker=marker, fillstyle="none", label=label, markevery = markevery)
            ax[3, test_id].semilogy(ts, ips_errors[:, 3], linestyle=linestyle, color=color, marker=marker, fillstyle="none", label=label, markevery = markevery)
            
            
            if (method == "Wasserstein" && ~preconditioner)
                ax_cont[1, test_id].scatter(ips_obj.θ[end][:, 1], ips_obj.θ[end][:, 2], s = 10)
            elseif (method == "Wasserstein" && preconditioner)
                ax_cont[2, test_id].scatter(ips_obj.θ[end][:, 1], ips_obj.θ[end][:, 2], s = 10)
            elseif (method == "Stein" && ~preconditioner)
                ax_cont[3, test_id].scatter(ips_obj.θ[end][:, 1], ips_obj.θ[end][:, 2], s = 10)
            else 
                ax_cont[4, test_id].scatter(ips_obj.θ[end][:, 1], ips_obj.θ[end][:, 2], s = 10)
            end

        end
    end

    ax_cont[1, test_id].set_title(L"\lambda = " * string(ϵ))            
    ax[1, test_id].set_title(L"\lambda = " * string(ϵ))
    ax[1, test_id].set_xlabel("t")
    ax[2, test_id].set_xlabel("t")
    ax[3, test_id].set_xlabel("t")
    
    ax[1, test_id].grid("on")
    ax[2, test_id].grid("on")
    ax[3, test_id].grid("on")
    
    
#     if test_id ==1
#         for i = 1:4
#             ax_cont[i,test_id].set_ylabel(L"\theta_{(2)}")
#         end
#     end
    ax_cont[4,test_id].set_xlabel(L"\theta^{(1)}")
    
end

handles, labels = ax[1,1].get_legend_handles_labels()
fig.legend(handles,labels,loc = "upper center",bbox_to_anchor=(0.5,0.99),ncol=4)
ax[1,1].set_ylabel(L"Error of $\mathbb{E}[\theta]$")
ax[2,1].set_ylabel(L"Rel. error of Cov$[\theta]$")
ax[3,1].set_ylabel(L"Error of $\mathbb{E}[\cos(\omega^T\theta + b)]$")

fig.subplots_adjust(bottom=0.05,top=0.91,left=0.1,right=0.98,hspace=0.2)
fig.savefig("Gaussian_gd_particle_converge.pdf")


ax_cont[1, 1].set_ylabel("Wasserstein GF")
ax_cont[2, 1].set_ylabel("Affine invariant Wasserstein  GF")
ax_cont[3, 1].set_ylabel("Stein  GF")
ax_cont[4, 1].set_ylabel("Affine invariant Stein  GF")
fig_cont.subplots_adjust(bottom=0.05,top=0.91,left=0.1,right=0.98,hspace=0.2)
fig_cont.savefig("Gaussian_gd_particle_contour.pdf")
