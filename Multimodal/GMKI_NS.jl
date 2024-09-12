using LinearAlgebra
using Distributions
using Random
using SparseArrays
using JLD2

include("../Inversion/Plot.jl")
include("../Inversion/KalmanInversion.jl")
include("../Fluid/Spectral-Navier-Stokes.jl")
include("../Fluid/Spectral-Mesh.jl")


ν = 1.0e-2                                      # viscosity
N, L = 128, 2*pi                                 # resolution and domain size 
ub, vb = 0.0, 2*pi                        # background velocity 
method="Crank-Nicolson"                         # RK4 or Crank-Nicolson
N_t = 2500;                                     # time step
T = 0.5;                                        # final time
obs_ΔNx, obs_ΔNy, obs_ΔNt = 8, 16, 1250         # observation
symmetric = true
σ_0 = sqrt(2)*pi
N_KL = 128
N_θ = 128

seed=22
Random.seed!(seed);

mesh = Spectral_Mesh(N, N, L, L)
s_param = Setup_Param(ν, ub, vb,  
    N, L,  
    method, N_t, T,
    obs_ΔNx, obs_ΔNy, obs_ΔNt; 
    symmetric = symmetric,
    N_ω0_θ = N_θ,
    N_ω0_ref = N_KL,
    f = (x, y) -> (0, cos(4*x)),
    σ = σ_0,
    ω0_seed=seed)
ω0_ref = s_param.ω0_ref
ω0_ref_mirror = -ω0_ref[[1;end:-1:2], :]
# generate observation data
y_noiseless = forward_helper(s_param, ω0_ref; symmetric=true, save_file_name="NS", vmin=-5.0, vmax=5.0);
y_noiseless_mirror = forward_helper(s_param, ω0_ref_mirror; symmetric=true, save_file_name="NS_mirror", vmin=-5.0, vmax=5.0);
@info "y - y_mirror = ", norm(y_noiseless - y_noiseless_mirror)

# compute posterior distribution by GMKI
N_iter =50
update_freq = 1
N_modes = 3
θ0_w  = fill(1.0, N_modes)/N_modes



θ0_mean, θθ0_cov  = zeros(N_modes, N_θ), zeros(N_modes, N_θ, N_θ)
for i = 1:N_modes
    θ0_mean[i, :]    .= rand(Normal(0, σ_0), N_θ) 
    θθ0_cov[i, :, :] .= Array(Diagonal(fill(1.0^2, N_θ)))
end
μ_0 = zeros(Float64, N_θ)  # prior/initial mean 
Σ_0 = Array(Diagonal(fill(σ_0^2, N_θ)))  # prior/initial covariance


σ_η = 0.1
N_y = length(y_noiseless)
y = y_noiseless + rand(Normal(0, σ_η), N_y)
Σ_η = Array(Diagonal(fill(σ_η^2, N_y)))


### Augment the system
aug_y = [y; μ_0]
aug_Σ_η = [Σ_η zeros(Float64, N_y, N_θ); zeros(Float64, N_θ, N_y) Σ_0]  
s_param.N_y = length(aug_y)


γ = 1.0
Δt = γ/(1+γ)
@time gmkiiobj = GMKI_Run(s_param, aug_forward, θ0_w, θ0_mean, θθ0_cov, aug_y, aug_Σ_η, Δt, update_freq, N_iter; unscented_transform="modified-2n+1")
@save "gmkiobj.jld2" gmkiobj



