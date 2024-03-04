using Random
using Distributions
using LinearAlgebra
using ForwardDiff
using NPZ
include("../Inversion/Plot.jl")
include("../Inversion/NGD.jl")
include("../Inversion/IPS.jl")
include("Darcy-1D.jl")



Random.seed!(42);

N, L = 128, 1.0
obs_ΔN = 16
d = 2.0
τ = 3.0
N_KL = N_θ = 16

ση = 1.0
σprior = 10.0
darcy = Darcy(N, L, N_KL, obs_ΔN, N_θ, ση, σprior, d, τ)

θ_ref = darcy.θ_ref
k = exp.(darcy.logk)
h = darcy.h_ref

# observation
y = darcy.y_obs

logρ(arg) = logρ_posterior(arg, darcy)
    
function compute_ddΦ(θ) 
    Φ   =  -logρ(θ)
    dΦ  =  -ForwardDiff.gradient(logρ, θ)
    ddΦ =  -ForwardDiff.hessian(logρ, θ)
    return Φ, dΦ, ddΦ
end

function compute_dlogρ(θ) 
    return logρ(θ), ForwardDiff.gradient(logρ, θ)
end


# intitial condition 
σ0 = 1.0
μ0 =  zeros(N_θ) 
Σ0 = Array(Diagonal(fill(σ0^2, N_θ)))
    



s_param = nothing
Φ_func(s_param, θ) = compute_ddΦ(θ) 
sampling_method = "UnscentedTransform"
N_ens = 100

N_t = 5000

compute_gradient = "second-order"
gradient_flow = "Fisher-Rao"
Δt = 0.162 
ngd_obj = NGD_Run(s_param, Φ_func, μ0, Σ0, sampling_method, N_ens,  Δt, N_t, compute_gradient, gradient_flow);
npzwrite("GFR_mean.npy", stack(ngd_obj.θ_mean)) 
npzwrite("GFR_cov.npy", stack(ngd_obj.θθ_cov)) 
@info "finish ", gradient_flow


gradient_flow = "Gradient_descent"
Δt = 0.002 #0.002-0.003
gd_obj = NGD_Run(s_param, Φ_func, μ0, Σ0, sampling_method, N_ens,  Δt, N_t, compute_gradient, gradient_flow);
npzwrite("GD_mean.npy", stack(gd_obj.θ_mean)) 
npzwrite("GD_cov.npy", stack(gd_obj.θθ_cov)) 
@info "finish ", gradient_flow



gradient_flow = "Wasserstein"
Δt = 0.018 #0.018-0.02
wgd_obj = NGD_Run(s_param, Φ_func, μ0, Σ0, sampling_method, N_ens,  Δt, N_t, compute_gradient, gradient_flow);
npzwrite("GW_mean.npy", stack(wgd_obj.θ_mean)) 
npzwrite("GW_cov.npy", stack(wgd_obj.θθ_cov))
@info "finish ", gradient_flow
    
    

Random.seed!(42);

∇logρ(s_param, θ) = compute_dlogρ(θ)

θ0 = Array(rand(MvNormal(μ0, Σ0), N_ens)')

method, preconditioner ="Stein", false
Δt = 0.099 #0.099
@info method, preconditioner
ips_obj = IPS_Run(s_param, ∇logρ, θ0, N_ens, Δt, N_t, method, preconditioner, 10.0)
npzwrite("Stein_theta.npy", stack(ips_obj.θ) )

        
method, preconditioner ="Stein", true
Δt = 0.009
@info method, preconditioner
ips_obj = IPS_Run(s_param, ∇logρ, θ0, N_ens, Δt, N_t, method, preconditioner, 10.0)
npzwrite("Stein_prec_theta.npy", stack(ips_obj.θ) )

# Stein_prec_θ = stack(ips_obj.θ)
# N_ens, N_θ, N_t = size(Stein_prec_θ) 
# Stein_prec_mean = zeros(N_θ, N_t)
# Stein_prec_cov = zeros(N_θ, N_θ, N_t) 
# for i = 1:N_t
#     Stein_prec_mean[:, i] = mean(Stein_prec_θ[:,:,i], dims=1)
#     Stein_prec_cov[:,:, i] = (Stein_prec_θ[:,:,i] - ones(N_ens)*Stein_prec_mean[:, i]')' * (Stein_prec_θ[:,:,i] - ones(N_ens)*Stein_prec_mean[:, i]') / (N_ens - 1)
# end
# Stein_prec_cov[:,:, end]

method, preconditioner ="Wasserstein", false
Δt = 0.030 #0.06 #0.058-0.061
@info method, preconditioner
ips_obj = IPS_Run(s_param, ∇logρ, θ0, N_ens, Δt, N_t, method, preconditioner, 1.0)
npzwrite("Wasserstein_theta.npy", stack(ips_obj.θ) )

method, preconditioner ="Wasserstein", true
Δt = 0.030
@info method, preconditioner
ips_obj = IPS_Run(s_param, ∇logρ, θ0, N_ens, Δt, N_t, method, preconditioner, 1.0)
npzwrite("Wasserstein_prec_theta.npy", stack(ips_obj.θ) ) 
       






