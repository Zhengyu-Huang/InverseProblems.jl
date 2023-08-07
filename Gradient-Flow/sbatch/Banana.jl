using Random
using Distributions
using LinearAlgebra
using ForwardDiff
include("../../Inversion/Plot.jl")
include("../../Inversion/RWMCMC.jl")
include("../../Inversion/SMC.jl")
include("../../Inversion/KalmanInversion.jl")
include("../../Inversion/NGD.jl")
include("../../Inversion/IPS.jl")
include("../../Inversion/Utility.jl")


mutable struct Setup_Param{IT<:Int}
    θ_names::Array{String,1}
    N_θ::IT
    N_y::IT
end

function Setup_Param(N_θ::IT, N_y::IT) where {IT<:Int}
    return Setup_Param(["θ"], N_θ, N_y)
end


function Gaussian_density_helper(θ_mean::Array{FT,1}, θθ_cov::Array{FT,2}, θ::Array{FT,1}) where {FT<:AbstractFloat}
    N_θ = size(θ_mean,1)
    
    return exp( -1/2*((θ - θ_mean)'* (θθ_cov\(θ - θ_mean)) )) / ( sqrt(det(θθ_cov)) )

end


function Gaussian_density(θ_mean::Array{FT,1}, θθ_cov::Array{FT,2}, θ::Array{FT,1}) where {FT<:AbstractFloat}
    N_θ = size(θ_mean,1)
    
    return exp( -1/2*((θ - θ_mean)'* (θθ_cov\(θ - θ_mean)) )) / ( (2π)^(N_θ/2)*sqrt(det(θθ_cov)) )

end

function Gaussian_2d(θ_mean::Array{FT,1}, θθ_cov::Array{FT,2}; std=2,  N_p = 100) where {FT<:AbstractFloat}
    
    N_p = 100
    std = 2
    U, S, _ = svd(θθ_cov)
    xy_p  = [cos.(LinRange(0, 2*pi, N_p))  sin.(LinRange(0, 2*pi, N_p))]'

    xy_p = U * (std * sqrt.(S) .* xy_p) + θ_mean*ones(N_p)'

    return xy_p[1, :], xy_p[2, :]
end


function KL_estimator(m, C, func_log_rho_post; N_ens = 100)
    Random.seed!(42)
    
    N_θ = length(m)
    c_weights    =  rand(Normal(0, 1), N_θ, N_ens)
    # shift mean to and covariance to I
    c_weights -= dropdims(mean(c_weights, dims=2), dims=2)*ones(N_ens)'
    U1, S1, _ = svd(c_weights)
    c_weights = (S1/sqrt(N_ens - 1.0) .\U1') *  c_weights 

    chol_C = cholesky(Hermitian(C)).L
    xs = zeros(N_ens, N_θ)
    xs .= (m * ones(N_ens)' + chol_C * c_weights)'
    
    log_rho_posts = zeros(N_ens)
    for i = 1:N_ens
        log_rho_posts[i] = func_log_rho_post(xs[i,:])
    end
    return -1/2*log(det(C)) - sum(log_rho_posts)/N_ens
end


function log_Rosenbrock(θ,  ϵ)
    θ₁, θ₂ = θ
    return -ϵ*(θ₂ - θ₁^2)^2/20 - (1.0 - θ₁)^2/20  #+ log_prior(θ)
end

function compute_Eref(ϵ, ω, b)
    xx = Array(LinRange(-200.0, 200.0, 10^7))

    E1 = sum(exp.(-(1 .- xx).^2/20))
    Eθ₁²  = sum(xx.^2 .* exp.(-(1 .- xx).^2/20))  / E1
    Eθ₁θ₂ = sum(xx.^3 .* exp.(-(1 .- xx).^2/20))  /E1
    Eθ₂²  = sum((xx.^4 .+ 10/ϵ) .* exp.(-(1 .- xx).^2/20))     /E1
    Ecos = zeros(length(b))
    for i = 1:length(b)
        Ecos[i]  = sum(cos.(ω[i,2]*xx.^2 + ω[i,1]*xx .+ b[i]).* exp.(-(1 .- xx).^2/20))/exp(5/ϵ*ω[i,2]^2) /E1
    end

    return [1.0;11.0], [Eθ₁²-1.0 Eθ₁θ₂-11.0; Eθ₁θ₂-11.0 Eθ₂²-121.0], Ecos 
end






Random.seed!(42)
######################### TEST #######################################
N_θ = 2
m_0 = [0.0; 0.0]
σ0 = 2
C_0 = [σ0^2  0.0; 0.0  σ0^2]
# ϵs = [0.01, 0.1, 1, 10, 100] 
ϵs = [0.01, 0.1, 1] 

s_param = Setup_Param(2, 2)
fig, ax = PyPlot.subplots(ncols=3, nrows=3, sharex=true, sharey="row", figsize=(12,12))
fig_cont, ax_cont = PyPlot.subplots(ncols=3, nrows=4, sharex=true, sharey=false, figsize=(12,12))
N_ens = 1000
θ0 = Array(rand(MvNormal(m_0, C_0), N_ens)')

Δt = 0.001
N_t = 15000
ts = LinRange(0, Δt*N_t, N_t+1)
xx = LinRange(-10, 10, 400)
yy = LinRange(-100, 100,  4000)
ω =  rand(Normal(0, 1), (20, 2))
b = rand(Uniform(0, 2*pi), 20)
y_ranges = [[-60,80],[-20,40],[-10, 40],[-2.5, 40],[-2.5, 20]]

for test_id = 1:length(ϵs)

    ϵ = ϵs[test_id]
    
    logρ(θ) = log_Rosenbrock(θ,  ϵs[test_id])
    
    function ∇logρ(s_param::Setup_Param, θ)
        return logρ(θ), ForwardDiff.gradient(logρ, θ)
    end
    
    m_oo, C_oo, cos_ref = compute_Eref(ϵs[test_id], ω, b)
    
    
    # compute posterior distribution by MCMC
    x_min, x_max = -10.0, 10.0
    y_min, y_max = y_ranges[test_id]
    N_x, N_y = 1000, 1000
    xx_cont = Array(LinRange(x_min, x_max, N_x))
    yy_cont = Array(LinRange(y_min, y_max, N_y))
    X_cont,Y_cont = repeat(xx_cont, 1, N_y), repeat(yy_cont, 1, N_x)'
    Z_cont = zeros(N_x, N_y)
    for i = 1:N_x
        for j = 1:N_y
            Z_cont[i, j] = logρ( [X_cont[i,j], Y_cont[i,j]] )
        end
    end
    Z_cont .= exp.(Z_cont)
    for i = 1:4
        ax_cont[i, test_id].contour(X_cont, Y_cont, Z_cont, 5, colors="grey", alpha=0.5)
    end
         
    
    
    

    for method in ["Wasserstein", "Stein"]
        for preconditioner in [false, true]
    
            ips_obj = IPS_Run(s_param, ∇logρ, θ0, N_ens, Δt, N_t, method, preconditioner)

            ips_errors    = zeros(N_t+1, 3)
            for i = 1:N_t+1
                ips_errors[i, 1] = norm(dropdims(mean(ips_obj.θ[i], dims=1), dims=1) .- m_oo)
                ips_errors[i, 2] = norm(construct_cov(ips_obj.θ[i]) .- C_oo)/norm(C_oo)
                ips_errors[i, 3] = norm(cos_ref - cos_error_estimation_particle(ips_obj.θ[i], ω, b ))/sqrt(length(b))
            end
            @info method, " preconditioner ", preconditioner, " ϵ = ", ϵ
            @info "mean = ", dropdims(mean(ips_obj.θ[end], dims=1), dims=1), " ", m_oo
            @info "cov = ", construct_cov(ips_obj, ips_obj.θ[end]), " ", C_oo
            
            

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
            
            @info sum(ips_errors)
            ax[1, test_id].plot(ts, ips_errors[:, 1], linestyle=linestyle, color=color, marker=marker, fillstyle="none", label=label, markevery = markevery)
            ax[2, test_id].plot(ts, ips_errors[:, 2], linestyle=linestyle, color=color, marker=marker, fillstyle="none", label=label, markevery = markevery)
            ax[3, test_id].plot(ts, ips_errors[:, 3], linestyle=linestyle, color=color, marker=marker, fillstyle="none", label=label, markevery = markevery)
            
            
            
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
fig.savefig("Banana_gd_particle_converge.pdf")



ax_cont[1, 1].set_ylabel("Wasserstein GF")
ax_cont[2, 1].set_ylabel("Affine invariant Wasserstein GF")
ax_cont[3, 1].set_ylabel("Stein GF")
ax_cont[4, 1].set_ylabel("Affine invariant Stein GF")
fig_cont.subplots_adjust(bottom=0.05,top=0.96,left=0.1,right=0.98,hspace=0.2)
fig_cont.savefig("Banana_gd_particle_contour.pdf")
