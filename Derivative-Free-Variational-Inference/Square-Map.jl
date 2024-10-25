function G(θ)
    #return θ
    return [θ'*θ; θ]
end


function F(θ, args)
    y, ση = args
    Gθ = G(θ)
    return (y - Gθ) ./ ση
end


function logrho(θ, args)
    Fθ = F(θ, args)
    return -0.5*norm(Fθ)^2
end


function V(θ, args)
    return -logrho(θ, args), 
           -ForwardDiff.gradient(x -> logrho(x, args), θ), 
           -ForwardDiff.hessian(x -> logrho(x, args), θ)
end



function Gaussian_mixture_VI(func_V, func_F, w0, μ0, Σ0; N_iter = 100, dt = 1.0e-3, quadrature_type_GM = "mean_point", N_ens_GM = -1)

    N_modes, N_θ = size(μ0)
    

    
    T =  N_iter * dt
    N_modes = 1
    x0_w = w0
    x0_mean = μ0
    xx0_cov = Σ0
    sqrt_matrix_type = "Cholesky"
    
    objs = []

    if func_V !== nothing
        gmgdobj = GMGD_Run(
        func_V, 
        T,
        N_iter,
        # Initial condition
        x0_w, x0_mean, xx0_cov;
        sqrt_matrix_type = sqrt_matrix_type,
        # setup for Gaussian mixture part
        quadrature_type_GM = quadrature_type_GM,
        N_ens_GM = N_ens_GM,
        # setup for potential function part
        Bayesian_inverse_problem = false, 
        quadrature_type = "mean_point")
        
        push!(objs, gmgdobj)

    end

    if func_F !== nothing
        N_f = length(func_F(ones(N_θ)))
        gmgdobj_BIP = GMGD_Run(
        func_F, 
        T,
        N_iter,
        # Initial condition
        x0_w, x0_mean, xx0_cov;
        sqrt_matrix_type = sqrt_matrix_type,
        # setup for Gaussian mixture part
        quadrature_type_GM = quadrature_type_GM,
        N_ens_GM = N_ens_GM,
        # setup for potential function part
        Bayesian_inverse_problem = true, 
        N_f = N_f,
        quadrature_type = "unscented_transform",
        c_weight_BIP = 1.0e-3,
        w_min=1e-8)
        
        push!(objs, gmgdobj_BIP)

    end

    return objs
end


