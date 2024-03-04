using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions
using Roots
# generate ensemble
Random.seed!(123)
"""
DiffusionKI{FT<:AbstractFloat, IT<:Int}
Structure that is used in Diffusion based Kalman Inversion (EKI)
#Fields
$(DocStringExtensions.FIELDS)
"""
mutable struct DiffusionKI{FT<:AbstractFloat, IT<:Int}
    "filter_type type"
    filter_type::String
    "a vector of arrays of size N_ensemble x N_parameters containing the parameters (in each EKI iteration a new array of parameters is added)"
    θ::Vector{Array{FT, 2}}
    "vector of observations (length: N_y)"
    y::Array{FT, 1}
    "covariance of the observational noise"
    Σ_η
    "size of θ"
    N_p::IT
    N_θ::IT
    "size of y"
    N_y::IT
    "current iteration number"
    iter::IT
    "time step, t₁=0, t₂, ..., tₙ₊₁=T"
    ts::Array{FT,1}
    "final time"
    T::FT
    "number of iterations"
    N_iter::IT
    "whether use stochastic or deterministic reverse process"
    randomized_update::Bool
    "quadrature parameters"
    "ensemble size"
    N_ens::IT
    c_weights::Array{FT,2}  
    mean_weights::Array{FT,1}  
    cov_weights::Array{FT,1}
    "prior information"
    prior_mean::Array{FT,1}
    prior_cov::Array{FT,2}
end



"""
UKIObj Constructor 
parameter_names::Array{String,1} : parameter name vector
θ0_mean::Array{FT} : prior mean
θθ0_cov::Array{FT, 2} : prior covariance
g_t::Array{FT,1} : observation 
obs_cov::Array{FT, 2} : observation covariance
α_reg::FT : regularization parameter toward θ0 (0 < α_reg <= 1), default should be 1, without regulariazion

unscented_transform : "original-2n+1", "modified-2n+1", "original-n+2", "modified-n+2" 
"""
function DiffusionKI(
    filter_type::String,
    # initial condition
    θ0::Array{FT, 2},
    y::Array{FT, 1},
    Σ_η::Array{FT, 2},
    # prior information
    prior_mean::Array{FT},
    prior_cov::Array{FT,2},
    N_iter::IT,
    T::FT,
    ts::Array{FT,1},     
    randomized_update::Bool;
    exact_init::Bool = true) where {FT<:AbstractFloat, IT<:Int}

    N_p, N_θ = size(θ0)

    if exact_init
        @info "Correct the initial condition, such that the empirical mean is 0, the empirical covariance is I."
        # correct the initial condition, such that the empirical mean is 0, the empirical covariance is I.
        # shift mean to 0
        θ0 -= repeat(mean(θ0, dims=1), N_p, 1)   
        U1, S1, V1t = svd(θ0)
        θ0 = sqrt(N_p)*U1
    end

    θ = Array{FT,2}[]        # array of Array{FT, 2}'s
    push!(θ, θ0)             # insert parameters at end of array (in this case just 1st entry)
    

    # unscented transform 
    if filter_type == "unscented_transform"
        unscented_transform = "modified-2n+1"
        if unscented_transform == "original-2n+1" ||  unscented_transform == "modified-2n+1"
            # ensemble size
            N_ens = 2*N_θ+1

            c_weights = zeros(FT, N_θ, N_ens)
            mean_weights = zeros(FT, N_ens)
            cov_weights = zeros(FT, N_ens)

            κ = 0.0
            β = 2.0
            α = min(sqrt(4/(N_θ + κ)), 1.0)
            λ = α^2*(N_θ + κ) - N_θ

            for i = 1:N_θ
                c_weights[i,i+1]      =   sqrt(N_θ + λ)
                c_weights[i,i+1+N_θ]  =  -sqrt(N_θ + λ)
            end
            mean_weights[1] = λ/(N_θ + λ)
            mean_weights[2:N_ens] .= 1/(2(N_θ + λ))
            cov_weights[1] = λ/(N_θ + λ) + 1 - α^2 + β
            cov_weights[2:N_ens] .= 1/(2(N_θ + λ))

            if unscented_transform == "modified-2n+1"
                mean_weights[1] = 1.0
                mean_weights[2:N_ens] .= 0.0
            end

        elseif unscented_transform == "original-n+2" ||  unscented_transform == "modified-n+2"

            N_ens = N_θ+2
            c_weights = zeros(FT, N_θ, N_ens)
            mean_weights = zeros(FT, N_ens)
            cov_weights = zeros(FT, N_ens)

            # todo cov parameter
            α = N_θ/(4*(N_θ+1))
        
            IM = zeros(FT, N_θ, N_θ+1)
            IM[1,1], IM[1,2] = -1/sqrt(2α), 1/sqrt(2α)
            for i = 2:N_θ
                for j = 1:i
                    IM[i,j] = 1/sqrt(α*i*(i+1))
                end
                IM[i,i+1] = -i/sqrt(α*i*(i+1))
            end
            c_weights[:, 2:end] .= IM

            if unscented_transform == "original-n+2"
                mean_weights .= 1/(N_θ+1)
                mean_weights[1] = 0.0

                cov_weights .= α
                cov_weights[1] = 0.0

            else unscented_transform == "modified-n+2"
                mean_weights .= 0.0
                mean_weights[1] = 1.0
                
                cov_weights .= α
                cov_weights[1] = 0.0
            end
        end


    elseif filter_type == "cubature_transform"
        # ensemble size
        N_ens = 2*N_θ

        c_weights    = zeros(FT, N_θ, N_ens)
        mean_weights = zeros(FT, N_ens)
        cov_weights  = zeros(FT, N_ens)

        for i = 1:N_θ
            c_weights[i,i]      =   sqrt(N_θ)
            c_weights[i,i+N_θ]  =  -sqrt(N_θ)
        end
                
        mean_weights[:] .= 1/N_ens
        cov_weights[:]  .= 1/N_ens
                
    else 
        error("error: filter_type ", filter_type, " has not implemented.")
    end

    iter = 0
    N_y = length(y)
    
    DiffusionKI{FT,IT}(
    filter_type, θ, 
    y, Σ_η, 
    N_p, N_θ, N_y, 
    iter, ts, 
    T, N_iter, 
    randomized_update, 
    N_ens, 
    c_weights,
    mean_weights,
    cov_weights,
    prior_mean,
    prior_cov)

end


function update_ensemble!(diffki::DiffusionKI{FT,IT}, ens_func::Function) where {FT<:AbstractFloat, IT<:Int}
    # update particles Z_{tₖ} at tₖ to Z_{tₖ₊₁} at tₖ₊₁
    # t = T - tₖ
    diffki.iter += 1
    iter = diffki.iter
    tₖ = diffki.ts[iter]
    Δt = diffki.ts[iter+1] - tₖ
    
    # particles at tₖ
    θ = diffki.θ[end]
    θ_p = similar(θ)
    N_p, N_θ = diffki.N_p, diffki.N_θ

    if diffki.randomized_update
        rhs, rhs_stoc = update_ensemble_rhs(θ, tₖ, diffki, ens_func) 
        # update particles Z_{tₖ} at tₖ
        θ_p = θ + Δt*rhs + sqrt(Δt)*rhs_stoc
        
    else
        # RK2 for the deterministic system

        k1, k2 = similar(θ), similar(θ)
        rhs, rhs_stoc = update_ensemble_rhs(θ, tₖ, diffki, ens_func) 
        k1 = Δt*rhs
        rhs, rhs_stoc = update_ensemble_rhs(θ+k1/2.0, tₖ+Δt/2.0, diffki, ens_func) 
        k2 = Δt*rhs
        θ_p = θ + k2


        # k1, k2, k3, k4 = similar(θ), similar(θ), similar(θ), similar(θ)
        # rhs, rhs_stoc = update_ensemble_rhs(θ, tₖ, diffki, ens_func) 
        # k1 = Δt*rhs
        # rhs, rhs_stoc = update_ensemble_rhs(θ+k1/2.0, tₖ+Δt/2.0, diffki, ens_func) 
        # k2 = Δt*rhs
        # rhs, rhs_stoc = update_ensemble_rhs(θ+k2/2.0, tₖ+Δt/2.0, diffki, ens_func) 
        # k3 = Δt*rhs
        # rhs, rhs_stoc = update_ensemble_rhs(θ+k3, tₖ+Δt, diffki, ens_func) 
        # k4 = Δt*rhs
        # θ_p = θ + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0 
    end

    


    
    push!(diffki.θ, θ_p)
    return θ_p
end







# compute mean 
# ρ(θ; z) ∝ exp{ -1/2<Σ_η⁻¹[y-G(θ)], y-G(θ)> - 1/2<Σ₀⁻¹(θ-r₀), (θ-r₀)>  - 1/2<z-λₜθ, z-λₜθ>/σₜ² } 
function compute_mean(diffki::DiffusionKI{FT}, ens_func::Function, θ, λₜ, σₜ) where {FT<:AbstractFloat}
    r₀ = diffki.prior_mean
    Σ₀ = diffki.prior_cov
    Σ̃₀ = inv(inv(Σ₀) + λₜ^2/σₜ^2*I)

    r̃₀ = copy(θ)
    N_p = diffki.N_p
    for j = 1:N_p
        r̃₀[j,:] = Σ̃₀*(Σ₀\r₀ + λₜ/σₜ^2*θ[j,:])
    end

    m = copy(θ)
    y, Σ_η = diffki.y, diffki.Σ_η
    for j = 1:N_p
        EG, CovθG, CovGG = integ_rule(diffki, ens_func, r̃₀[j,:], Σ̃₀)
        m[j,:] = r̃₀[j,:] + CovθG*((CovGG + Σ_η)\(y - EG))
    end

    return m
end

function integ_rule(diffki::DiffusionKI{FT}, ens_func::Function, r::Array{FT,1}, Σ::Array{FT,2}) where {FT<:AbstractFloat}
    N_ens, N_y = diffki.N_ens, diffki.N_y
    ############ Generate sigma points
    θ = construct_sigma_ensemble(diffki.c_weights, r, Σ)

    g = zeros(FT, N_ens, N_y)
    g .= ens_func(θ)
    
    # play the role of symmetrizing the covariance matrix
    EG = construct_mean(diffki.mean_weights, g)
    EθG = construct_cov(diffki.cov_weights, θ, r, g, EG)
    EGG = construct_cov(diffki.cov_weights, g, EG)
    
    return EG, EθG, EGG
end




"""
construct_sigma_ensemble
Construct the sigma ensemble, based on the mean x_mean, and covariance x_cov
"""
function construct_sigma_ensemble(c_weights, x_mean::Array{FT}, x_cov::Array{FT,2}) where {FT<:AbstractFloat}
  
    N_x, N_ens = size(c_weights)
    chol_xx_cov = cholesky(Hermitian(x_cov)).L

    x = zeros(FT, N_ens, N_x)
    for i = 1: N_ens
        x[i,     :] = x_mean + chol_xx_cov * c_weights[:, i]
    end
  
    return x
end


"""
construct_mean x_mean from ensemble x
"""
function construct_mean(mean_weights, x::Array{FT,2}) where {FT<:AbstractFloat}
    N_ens, N_x = size(x)

    x_mean = zeros(FT, N_x)
    
    for i = 1: N_ens
        x_mean += mean_weights[i]*x[i,:]
    end

    return x_mean
end

"""
construct_cov xx_cov from ensemble x and mean x_mean
"""
function construct_cov(cov_weights, x::Array{FT,2}, x_mean::Array{FT}) where {FT<:AbstractFloat}
    N_ens, N_x = size(x)
    
    xx_cov = zeros(FT, N_x, N_x)

    for i = 1: N_ens
        xx_cov .+= cov_weights[i]*(x[i,:] - x_mean)*(x[i,:] - x_mean)'
    end

    return xx_cov
end

"""
construct_cov xy_cov from ensemble x and mean x_mean, ensemble y and mean y_mean
"""
function construct_cov(cov_weights, x::Array{FT,2}, x_mean::Array{FT}, y::Array{FT,2}, y_mean::Array{FT}) where {FT<:AbstractFloat}
    N_ens, N_x = size(x)
    _, N_y = size(y)
    
    xy_cov = zeros(FT, N_x, N_y)

    for i = 1: N_ens
        xy_cov .+= cov_weights[i]*(x[i,:] - x_mean)*(y[i,:] - y_mean)'
    end

    return xy_cov
end


function ensemble(s_param, θ_ens::Array{FT,2}, forward::Function)  where {FT<:AbstractFloat}
    N_ens,  N_θ = size(θ_ens)
    N_y = s_param.N_y
    g_ens = zeros(FT, N_ens,  N_y)
    Threads.@threads for i = 1:N_ens
        θ = θ_ens[i, :]
        g_ens[i, :] .= forward(s_param, θ)
    end
    return g_ens
end

function DiffusionKI_Run(
    s_param, forward::Function, 
    θ0::Array{FT, 2},
    prior_mean::Array{FT, 1}, prior_cov::Array{FT, 2},
    y::Array{FT, 1}, Σ_η::Array{FT, 2},
    T::FT,
    N_iter::IT;
    filter_type::String = "cubature_transform",
    ts::Array{FT,1} = collect(LinRange(0, T, N_iter+1)),
    randomized_update::Bool = false) where {FT<:AbstractFloat, IT<:Int}
    
    diffkiobj = DiffusionKI(filter_type, θ0, y, Σ_η, prior_mean, prior_cov, N_iter, T, ts, randomized_update)
    
    ens_func(θ_ens) = ensemble(s_param, θ_ens, forward)
    
    for i = 1:N_iter

        # DEBUG: comparing mₜ with analytical m for linear inverse problems
        # θ = diffkiobj.θ[end]
        # mₜ_exact = copy(θ)
        # G = s_param.G
        # t = T - diffkiobj.Δt * diffkiobj.iter
        # λₜ = exp(-t)
        # σₜ = sqrt(1 - exp(-2*t))
        # for i = 1:diffkiobj.N_p
        #     Cₜ = inv(G'*(Σ_η\G) + inv(prior_cov) + λₜ^2/σₜ^2*I)
        #     mₜ_exact[i,:] = Cₜ*(prior_cov\prior_mean + G'*(Σ_η\y) + λₜ/σₜ^2*θ[i,:])
        # end


        update_ensemble!(diffkiobj, ens_func) 
        
        # @info "norm(mₜ - mₜ_exact) = ", norm(mₜ - mₜ_exact)

    end
    
    return diffkiobj
end







function update_ensemble_rhs(θ::Array{FT, 2}, tₖ::FT, diffki::DiffusionKI{FT,IT}, ens_func::Function) where {FT<:AbstractFloat, IT<:Int}
    # update particles Z_{tₖ} at tₖ to Z_{tₖ₊₁} at tₖ₊₁
    # t = T - tₖ
    
    t = diffki.T - tₖ
    λₜ = exp(-t)
    σₜ² = 1 - exp(-2*t)
    σₜ = sqrt(σₜ²)
    
    # particles at tₖ
    N_p, N_θ = diffki.N_p, diffki.N_θ
    rhs, rhs_stoc = similar(θ), similar(θ)
    
    mₜ = compute_mean(diffki, ens_func, θ, λₜ, σₜ)       


    if diffki.randomized_update
        noise = rand(Normal(0, 1), (N_p, N_θ))
        for j = 1:N_p
            rhs[j,:] = (θ[j,:] + 2*(λₜ*mₜ[j,:] - θ[j,:])/σₜ²) 
            rhs_stoc[j, :] = sqrt(2.0)*noise[j,:]
        end
    else
        for j = 1:N_p
            rhs[j,:] = (θ[j,:] + (λₜ*mₜ[j,:] - θ[j,:])/σₜ²)
            rhs_stoc[j, :] .= 0.0
        end
    end

    return rhs, rhs_stoc
end




# generate mesh between a and b 
# dx0, dx0*r, dx0*r^2 ... dx0*r^{N-2}  r ≥ 1
# b - a = dx0*(r^{N-1} - 1)/(r - 1)
# a ,.....   b-dx0-dx0*r,  b-dx0, b
function GeoSpace(a, b, N; r = -1.0, dx0= -1.0)

    xx = Array(LinRange(a, b, N))

    if r > 1 || dx0 > 0
        if r > 1 
            dx0 = (b - a)/((r^(N - 1) - 1)/(r - 1))

            dx = dx0
            for i = N-1:-1:2
                xx[i] = xx[i+1] - dx
                dx *= r
            end

        else
            # first use r=1.05 to generate half of the grids
            # then compute r and generate another half of the grids
            f(r)  = (r-1)*(b-a) - dx0*(r^(N-1) - 1)
            Df(r) = (b-a) - dx0*(N-1)*r^(N-2)  
            r = find_zero(f, (1+1e-4, 1.5), Roots.Bisection())
            if r > 1.02
                r = min(r, 1.02)
                dx = dx0
                Nf = div(N, 2)
               
                
                # for i = 2:Nf
                for i = N-1:-1:N-Nf+1
                    xx[i] = xx[i+1] - dx
                    dx *= r
                end
                b = xx[N-Nf+1]
                dx0 = dx
                
                f = (r) -> (r-1)*(b-a)-dx0*(r^(N - Nf) - 1)
                Df = (r) -> (b-a)-dx0*(N-Nf)*r^(N - Nf - 1)
        
        
                r = find_zero(f, (1+1e-4, 2.0), Roots.Bisection())
                # for i = Nf+1:N-1
                for i = N-Nf:-1:2
                    xx[i] = xx[i+1] - dx
                    dx *= r
                end

            else
                dx = dx0
                for i = N-1:-1:2
                    xx[i] = xx[i+1] - dx
                    dx *= r
                end
            end
            
        end 
    
        
    end

    return xx
end    
