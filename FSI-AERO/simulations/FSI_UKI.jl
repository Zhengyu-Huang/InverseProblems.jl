using JLD
using LinearAlgebra
include("../calibration/IO.jl")
include("../../Inversion/UKI.jl")


# N_θ = 5 case with 5% Gaussian error
N_θ = 5
noise_level = 0.05

y_noiseless = read_obs("results.1/reference/AGARD.disp.";  NT = 2000,  Δobs = 40)[1][:]

N_y = length(y_noiseless)
# generate noisy observation 
y = copy(y_noiseless)
Random.seed!(42);
for i = 1:length(y)
    noise = rand(Normal(0, noise_level*y[i]))
    y[i] += noise
end
Σ_η = Array(Diagonal(fill(0.1^2, length(y))))

θ_names = ["damage_θ"]
# initial mean and covariance
θ0_mean = zeros(Float64, N_θ)               # mean 
# the exponential map to 0 damage
θθ0_cov = Array(Diagonal(fill(1.0, N_θ)))


N_iter = 20
ites = Array(1:N_iter)

# update evolution covariance to Cn
update_freq = 1
# no regularization
α_reg = 1.0

RESTART = true
if RESTART
    ukiobj = load("ukiobj.jld")["ukiobj"]
else
    ukiobj = UKIObj(θ_names ,
    θ0_mean, 
    θθ0_cov,
    y,
    Σ_η,
    α_reg,
    update_freq;
    modified_uscented_transform = true)
end


# ensemble function
ens_func(θ_ens) = begin
    N_ens,  N_θ = size(θ_ens)
    g_ens = zeros(Float64, N_ens,  N_y)
    
    for i = 1:N_ens
        θ = θ_ens[i, :]
        @info "θ = ", θ
        # generate the inpute file
        generate_mat_file("./agard.fem.composite", θ)
        # run aero-f aero-s
        run(`./run.unsteady`)
        # read displacement field
        obs, _ = read_obs("results.1/AGARD.disp.", NT = 2000, Δobs = 40)
        
        g_ens[i, :] .= obs[:]
    end

    return g_ens
end


for i in ukiobj.iter+1:N_iter
    @info "Iteration : ", i
    update_ensemble!(ukiobj, ens_func) 

    @info "mean norm = ", norm(ukiobj.θ_mean[end]), " cov norm = ", norm(ukiobj.θθ_cov[end])
    save("ukiobj.jld", "ukiobj", ukiobj)
    
end


