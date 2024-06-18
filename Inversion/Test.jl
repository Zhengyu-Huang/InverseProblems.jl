include("GMGD.jl")
########## Test compute_logρ_gm_expectation
# Gaussian
# Elogρ_mean = -N_θ/2 - log |2\pi C| /2
# ∇logρ_mean = 0
# ∇²logρ_mean = -C^-1
N_modes, N_x = 1, 3
x_w = ones(1)
x_mean = reshape([1.0, 2.0, 3.0], 1, N_x)
xx_cov = reshape([2.0 1.0 1.0; 1.0 3.0 2.0; 1.0 2.0 4.0], 1, N_x, N_x)


for compute_sqrt_matrix_type in ["SVD"]#["Cholesky", "SVD"]
    sqrt_xx_cov, inv_sqrt_xx_cov = [], []
    for im = 1:N_modes
        sqrt_cov, inv_sqrt_cov = compute_sqrt_matrix(xx_cov[im,:,:]; type=compute_sqrt_matrix_type) 
        push!(sqrt_xx_cov, sqrt_cov)
        push!(inv_sqrt_xx_cov, inv_sqrt_cov) 
    end

    c_weight = 0.1 

    for quadrature_type in ["cubature_transform_o3", "cubature_transform_o5", "unscented_transform"]
        _, c_weights_GM, mean_weights_GM = generate_quadrature_rule(N_x, quadrature_type; c_weight=c_weight)
        logρ_mean, ∇logρ_mean, ∇²logρ_mean = compute_logρ_gm_expectation(x_w, x_mean, sqrt_xx_cov, inv_sqrt_xx_cov, c_weights_GM, mean_weights_GM)
        
        @assert(abs(logρ_mean[1] + (N_x + log(det(2 * π * xx_cov[1, :, :]))) / 2.0) < 1e-12)
        @assert(norm(∇logρ_mean[1,:]) < 1e-12)
        @assert(norm(∇²logρ_mean[1,:,:] + inv(xx_cov[1,:,:])) < 1e-12)
    end
end







include("QuadratureRule.jl")

x_mean = [3.0, 1.0, 2.0]
xx_cov = [1.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 2.0]
N_x = length(x_mean)

N_f = 2
A₁, b₁, c₁ = [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 3.0], [1.0; 2.0; 3.0], 1.0
A₂, b₂, c₂ = [3.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 2.0], [2.0; 1.0; 1.0], 1.0

args = (A₁,b₁,c₁,A₂,b₂,c₂)
compute_sqrt_matrix_type = "Cholesky"
sqrt_xx_cov, inv_sqrt_xx_cov = compute_sqrt_matrix(xx_cov; type=compute_sqrt_matrix_type)
 
Φᵣ_means, ∇Φᵣ_means, ∇²Φᵣ_means = [], [] ,[]
for (Bayesian_inverse_problem,  quadrature_type) in [(false,   "cubature_transform_o5"), (true,   "unscented_transform")]
    c_weight = 0.1 
    N_ens, c_weights, mean_weights = generate_quadrature_rule(N_x, quadrature_type; c_weight=c_weight)

    xs = construct_ensemble(x_mean, sqrt_xx_cov; c_weights = c_weights)
    if Bayesian_inverse_problem
        V, ∇V, ∇²V = zeros(N_ens, N_f), zeros(N_ens, N_f, N_x), zeros(N_ens, N_f, N_x, N_x)
        for i = 1:N_ens
            V[i,:], ∇V[i,:,:] = func_dF(xs[i,:], args)
        end
    else
        V, ∇V, ∇²V = zeros(N_ens), zeros(N_ens, N_x), zeros(N_ens, N_x, N_x)
        for i = 1:N_ens
            V[i], ∇V[i,:], ∇²V[i,:,:] = func_dPhi_R(xs[i,:], args)
        end
    end

    Φᵣ_mean, ∇Φᵣ_mean, ∇²Φᵣ_mean = Bayesian_inverse_problem ? 
    compute_expectation_BIP(x_mean, inv_sqrt_xx_cov, V, c_weight) : 
    compute_expectation(V, ∇V, ∇²V, mean_weights) 
    push!(Φᵣ_means, Φᵣ_mean) 
    push!(∇Φᵣ_means, ∇Φᵣ_mean) 
    push!(∇²Φᵣ_means, ∇²Φᵣ_mean)
end 

@assert(abs(Φᵣ_means[1] - Φᵣ_means[2]) < 1.0e-8)
@assert(norm(∇Φᵣ_means[1] - ∇Φᵣ_means[2]) < 1.0e-8)
@assert(norm(∇²Φᵣ_means[1] - ∇²Φᵣ_means[2]) < 1.0e-8)
