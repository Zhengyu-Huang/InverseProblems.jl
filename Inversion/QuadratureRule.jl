using LinearAlgebra
using ForwardDiff

function generate_quadrature_rule(N_x, quadrature_type; c_weight=sqrt(N_x),  quadrature_order = 3)
    if quadrature_type == "mean_point"
        N_ens = 1
        c_weights    = zeros(N_x, N_ens)
        mean_weights = ones(N_ens)
        
    elseif  quadrature_type == "unscented_transform"
        @info N_x
        N_ens = 2N_x+1
        c_weights = zeros(N_x, N_ens)
        for i = 1:N_x
            c_weights[i, i+1]      =  c_weight
            c_weights[i, N_x+i+1]  = -c_weight
        end
        mean_weights = fill(1/(2.0*c_weight^2), N_ens)
        # warning: when c_weight <= sqrt(N_x), the weight is negative
        # the optimal is sqrt(3), 
        # Julier, S. J., Uhlmann, J. K., & Durrant-Whyte, H. F. (2000). A new method for nonlinear transformation of means and covariances in filters and estimators. 
        mean_weights[1] = 1 - N_x/c_weight^2
    
    elseif quadrature_type == "cubature_transform"
        if quadrature_order == 3
            N_ens = 2N_x
            c_weight = sqrt(N_x)
            c_weights = zeros(N_x, N_ens)
            for i = 1:N_x
                c_weights[i, i]          =  c_weight
                c_weights[i, N_x+i]      =  -c_weight
            end
            mean_weights = ones(N_ens)/N_ens 
        elseif quadrature_order == 5
            # High-degree cubature Kalman filter✩
            # Bin Jia, Ming Xin, Yang Cheng
            N_ens = 2N_x*N_x + 1
            c_weights    = zeros(N_x, N_ens)
            mean_weights = ones(N_ens)

            mean_weights[1] = 2.0/(N_x + 2)

            for i = 1:N_x
                c_weights[i, 1+i]          =  sqrt(N_x+2)
                c_weights[i, 1+N_x+i]      =  -sqrt(N_x+2)
                mean_weights[i+1] = mean_weights[N_x+i+1] = (4 - N_x)/(2*(N_x+2)^2)
            end
            ind = div(N_x*(N_x - 1),2)
            for i = 1: N_x
                for j = i+1:N_x
                    c_weights[i, 2N_x+1+i],      c_weights[j, 2N_x+1+i]        =   sqrt(N_x/2+1),  sqrt(N_x/2+1)
                    c_weights[i, 2N_x+1+ind+i],  c_weights[j, 2N_x+ind+1+i]    =  -sqrt(N_x/2+1), -sqrt(N_x/2+1)
                    c_weights[i, 2N_x+1+2ind+i], c_weights[j, 2N_x+2ind+1+i]   =   sqrt(N_x/2+1), -sqrt(N_x/2+1)
                    c_weights[i, 2N_x+1+3ind+i], c_weights[j, 2N_x+3ind+1+i]   =  -sqrt(N_x/2+1),  sqrt(N_x/2+1)
                    
                    mean_weights[2N_x+1+i]      = 1.0/(N_x+2)^2
                    mean_weights[2N_x+1+ind+i]  = 1.0/(N_x+2)^2
                    mean_weights[2N_x+1+2ind+i] = 1.0/(N_x+2)^2
                    mean_weights[2N_x+1+3ind+i] = 1.0/(N_x+2)^2
                end
            end
            

        else 
            print("cubature tansform with quadrature order ", quadrature_order, " has not implemented.")
        end
    end

    return N_ens, c_weights, mean_weights
end

function compute_sqrt_matrix(C; inverse=false, type="Cholesky")
    @info type
    if type == "Cholesky"
        sqrt_cov = inverse ? inv(cholesky(Hermitian(C)).L) : cholesky(Hermitian(C)).L
    elseif type == "SVD"
        U, D, _ = svd(Hermitian(C))
        sqrt_cov = inverse ? Diagonal(sqrt.(1.0./D))*U'  : U*Diagonal(sqrt.(D))
        
    else
        print("Type ", type, " for computing sqrt matrix has not implemented.")
    end
    return sqrt_cov  
end

function construct_ensemble(x_mean, sqrt_cov; c_weights, quadrature_type = "unscented_transform", N_ens = 10)

    N_x = size(x_mean)

    if quadrature_type == "random_sampling"
        xs = ones(N_ens)*x_mean' + (sqrt_cov * rand(Normal(0, 1),N_x, N_ens))'
    else
        N_ens = size(c_weights,2)
        xs = ones(N_ens)*x_mean' + (sqrt_cov * c_weights)'
    end

    return xs
end



function compute_expectation(x_mean, inv_sqrt_cov, V, ∇V, ∇²V;
                             Bayesian_inverse_problem = false,
                             gradient_computation_order = 0,
                             c_weight=sqrt(size(x_mean)))

    N_x = length(x_mean)
    Φᵣ_mean, ∇Φᵣ_mean, ∇²Φᵣ_mean = 0.0, zeros(N_x), zeros(N_x, N_x)

    if Bayesian_inverse_problem # Φᵣ = FᵀF/2 
        if gradient_computation_order == 0
            α = c_weight
            N_ens, N_f = size(V)
            a = zeros(N_x, N_f)
            b = zeros(N_x, N_f)
            c = zeros(N_f)
            
            c = V[1, :]
            for i = 1:N_x
                a[i, :] = (V[i+1, :] + V[i+N_x+1, :] - 2*V[1, :])/(2*α^2)
                b[i, :] = (V[i+1, :] - V[i+N_x+1, :])/(2*α)
            end
            ATA = a * a'
            BTB = b * b'
            BTA = b * a'
            BTc, ATc = b * c, a * c
            cTc = c' * c
            Φᵣ_mean = 1/2*(sum(ATA) + 2*tr(ATA) + 2*sum(ATc) + tr(BTB) + cTc)
            ∇Φᵣ_mean = inv_sqrt_cov'*(sum(BTA,dims=2) + 2*diag(BTA) + BTc)
            ∇²Φᵣ_mean = inv_sqrt_cov'*( Diagonal(2*dropdims(sum(ATA, dims=2), dims=2) + 4*diag(ATA) + 2*ATc) + BTB)*inv_sqrt_cov
           

        else 
            print("Compute expectation for Bayesian inverse problem. 
                   Gradient computation order ", gradient_computation_order, " has not implemented.")
        end

    else # general sampling problem Φᵣ = V 
        @info size(V), size(∇V), size(∇²V), size(mean_weights)
        Φᵣ_mean   = mean_weights' * V
        ∇Φᵣ_mean  = mean_weights' * ∇V
        ∇²Φᵣ_mean = sum(mean_weights[i] * ∇²V[i, :, :] for i = 1:length(mean_weights))

    end

    return Φᵣ_mean, ∇Φᵣ_mean, ∇²Φᵣ_mean
end




#F₁ = xᵀA₁x + b₁ᵀx₁ + c₁
#F₂ = xᵀA₂x + b₂ᵀx₂ + c₂
#Φᵣ = FᵀF/2

function func_F(x, args)
    A₁,b₁,c₁,A₂,b₂,c₂ = args
    return [x'*A₁*x + b₁'*x + c₁; 
            x'*A₂*x + b₂'*x + c₂]
end

function func_Phi_R(x, args)
    F = func_F(x, args)
    Φᵣ = (F' * F)/2.0
    return Φᵣ
end

function func_dF(x, args)
    return func_F(x, args), 
           ForwardDiff.jacobian(x -> func_F(x, args), x)
end

function func_dPhi_R(x, args)
    return func_Phi_R(x, args), 
           ForwardDiff.gradient(x -> func_Phi_R(x, args), x), 
           ForwardDiff.hessian(x -> func_Phi_R(x, args), x)
end


# x_mean = [1.0, 2.0]
# xx_cov = [2.0 1.0; 1.0 3.0]
x_mean = [0.0, 0.0]
xx_cov = [1.0 0.0; 0.0 2.0]
N_x = length(x_mean)

N_f = 2
# A₁, b₁, c₁ = [1.0 0.0; 0.0 2.0], [1.0; 2.0], 1.0
# A₂, b₂, c₂ = [1.0 1.0; 1.0 2.0], [2.0; 1.0], 3.0
# A₁, b₁, c₁ = [1.0 0.0; 0.0 3.0], [1.0; 2.0], 1.0
# A₂, b₂, c₂ = [1.0 0.0; 0.0 2.0], [2.0; 1.0], 3.0
A₁, b₁, c₁ = [1.0 0.0; 0.0 2.0], [1.0; 2.0], 1.0
A₂, b₂, c₂ = [3.0 0.0; 0.0 1.0], [2.0; 1.0], 1.0

args = (A₁,b₁,c₁,A₂,b₂,c₂)
compute_sqrt_matrix_type = "Cholesky"
sqrt_xx_cov = compute_sqrt_matrix(xx_cov; inverse=false, type=compute_sqrt_matrix_type)
inv_sqrt_xx_cov = compute_sqrt_matrix(xx_cov; inverse=true, type=compute_sqrt_matrix_type)


Bayesian_inverse_problem, gradient_computation_order, quadrature_type, quadrature_order =false, 2, "cubature_transform", 5
#Bayesian_inverse_problem, gradient_computation_order, quadrature_type, quadrature_order =true, 0, "unscented_transform", 2
c_weight = 0.1 
N_ens, c_weights, mean_weights = generate_quadrature_rule(N_x, quadrature_type; c_weight=c_weight,  quadrature_order = quadrature_order)

xs = construct_ensemble(x_mean, sqrt_xx_cov; c_weights = c_weights, quadrature_type = quadrature_type)
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


Φᵣ_mean, ∇Φᵣ_mean, ∇²Φᵣ_mean = compute_expectation(x_mean, inv_sqrt_xx_cov, V, ∇V, ∇²V;
                             Bayesian_inverse_problem = Bayesian_inverse_problem,
                             gradient_computation_order = gradient_computation_order,
                             c_weight=c_weight)


    
