using Random
using Statistics
using Distributions
using PyPlot
include("../Inversion/Numerics.jl")
mutable struct Setup_Param{FT<:AbstractFloat, IT<:Int}
    θ_names::Array{String,1}
    N_θ::IT
    N_y::IT


    K::IT
    J::IT
    
    F::FT
    h::FT
    c::FT
    d::FT
    b::FT 
    
    
    T::FT
    N_t::IT
    Δt::FT
    RK_order::IT
    
    obs_type::String
    obs_p::IT
    

    
    
end

function Setup_Param(N_θ::IT, K::IT, J::IT, F::FT, h::FT, c::FT, b::FT, d::FT,
                     T::FT, Δt::FT, RK_order::IT,
                     obs_type::String, obs_p::IT) where {FT<:AbstractFloat, IT<:Int}
    """
    SUPERVISED LEARNING FROM NOISY OBSERVATIONS: COMBINING MACHINE-LEARNING TECHNIQUES WITH DATA ASSIMILATION
    
    K,  J = 8, 32
    F,  h, c, b = 20.0, 1.0, 10.0, 10.0
    d = b

    """
    θ_names = ["θ"]
    
    N_t = IT(T / Δt)
    
    if obs_type == "Statistics"
        kmode = obs_p
        N_y = kmode + div(kmode*(kmode + 1), 2)
    elseif obs_type == "Time-Series"
        Δobs = obs_p
        N_y = K*div(N_t, Δobs)
    else
        error("obs_type ", obs_type, " has not implemented, yet")
    end
    
    Setup_Param(θ_names, N_θ, N_y, K, J, F, h, c, d, b, 
                T, N_t, Δt, RK_order, 
                obs_type, obs_p)
end




function ΦNN(x::FT, θ::Array{FT, 1}) where {FT<:AbstractFloat}
    n = div(length(θ)-1, 3)
    a, b, c, d = θ[1:n], θ[n+1:2n], θ[2n+1:3n], θ[3n+1]
    # σ(a x + b) ⋅ c + e
    return d + c' * tanh.(a*x+b)
end


function Φpoly(x::FT, θ::Array{FT, 1}) where {FT<:AbstractFloat}
    #=
    This set of basis is not stable
    =#
    N_θ = length(θ)
    xs = [x^i for i = 0:N_θ-1]
    return θ' * xs
end


function ΦGP(x::FT, θ::Array{FT, 1}) where {FT<:AbstractFloat}
    #=
    x ∈ [-20, 20]
    kernal k(x, x') = exp(- (x - x')^2/(2l^2))
    
    f = θi k(xi, x)
    =#
    x_l, x_h = -20.0, 20.0
    
    N_θ = length(θ)
    nm = N_θ - 1
    xs = Array(LinRange(x_l, x_h, nm))
    
    l = θ[nm+1]
    
    return θ[1:nm]' * exp.(-(xs .- x).^2/(2l^2))
    
end

function ΦLP(x::FT, θ::Array{FT, 1}) where {FT<:AbstractFloat}
    #=
    x ∈ [x_l, x_h], x_l = -20 x_h = 20, piecewise linear polynomial
    =#
    x_l, x_h = -20.0, 20.0
    
    if x >= x_h 
        return θ[end]
    elseif x <= x_l
        return θ[1]
    end
    
    N_e = length(θ) - 1
    
    Δx = (x_h - x_l)/N_e
    int_val = ceil(IT, (x - x_l)/Δx)
    ξ = (x - x_l - Δx*(int_val-1) )/Δx
    
    return (1 - ξ)*θ[int_val] + ξ*θ[int_val+1] 
end




function ΦQP(x::FT, θ::Array{FT, 1}) where {FT<:AbstractFloat}
    #=
    x ∈ [x_l, x_h], x_l = -20 x_h = 20, piecewise cubic polynomial
    o --- o --- o ...... o --- o
    1     3                    N_θ-1=2ne+1
    2     4                    N_θ  =2ne+2
    =#
    x_l, x_h = -20.0, 20.0
    if x >= x_h 
        return θ[end-1]
    elseif x <= x_l
        return θ[1]
    end
    N_e = div(length(θ)-2 ,2)
    
    Δx = (x_h - x_l)/N_e
    int_val = ceil(Int64, (x - x_l)/Δx)
    
    ξ = (x - x_l - Δx*(int_val-1) )/Δx
    
    @assert(ξ>= 0 && ξ <= 1.0)
    
    N1 = (1 - 3ξ^2 + 2ξ^3)  # 0  -> 1  
    N2 = (ξ - 2ξ^2 + ξ^3)   # 0' -> 1
    N3 = (3ξ^2 - 2ξ^3)      # 1  -> 1
    N4 = (-ξ^2 + ξ^3)       # 1' -> 1
    
    return N1*θ[2*int_val-1] + N2*θ[2*int_val] + N3*θ[2*int_val+1] + N4*θ[2*int_val+2]
end

"""
Q = [X ; Y(k=1) ; Y(k=2) ... ; Y(k=K)]
"""
function Multiscale_Lorenz96_res(s_param::Setup_Param{FT, IT}, Q::Array{FT, 1}) where {FT<:AbstractFloat, IT<:Int}
    J, K = s_param.J, s_param.K
    F, h, c, d, b = s_param.F, s_param.h, s_param.c, s_param.d, s_param.b
    
    ng = 2
    Xg = zeros(FT, K+2*ng)
    Yg = zeros(FT, J+2*ng, K+2*ng)
    
    
    Xg[ng+1:K+ng] .= Q[1:K]
    # Y[j ,k]
    Yg[ng+1:J+ng, ng+1:K+ng] .= reshape(Q[K+1:end], J, K)
    
    # update periodic boundary
    for i = 1:ng
        # ng + 1 - i -> K+ng + 1 - i
        # K+ng + i -> i + ng
        Xg[ng + 1 - i] = Xg[K+ng + 1 - i]
        Xg[K+ng + i] = Xg[i+ng]
        
        Yg[:, ng + 1 - i] = Yg[:, K+ng + 1 - i]
        Yg[:, K+ng + i] = Yg[:, i+ng]
        
        Yg[ng + 1 - i, :] = Yg[J+ng + 1 - i,:]
        Yg[J+ng + i,:] = Yg[i+ng,:]
        
    end
    
    dXg = zeros(FT, K+2*ng)
    dYg = zeros(FT, J+2*ng, K+2*ng)
    
    for k = ng+1:K+ng
        dXg[k] = - Xg[k-1]*(Xg[k-2] - Xg[k+1]) - Xg[k] + F - h*c/d*(sum(Yg[ng+1:J+ng, k]))
        for j = ng+1:J+ng
            dYg[j,k] = -c*b*Yg[j+1,k]*(Yg[j+2,k] - Yg[j-1,k]) - c*Yg[j,k]  + h*c/d*Xg[k]
        end
    end
    
    dQ = similar(Q)
    
    dQ[1:K] .= dXg[ng+1:K+ng]
    dQ[K+1:end] .= dYg[ng+1:J+ng, ng+1:K+ng][:]
    
    return dQ
    
end


"""
Q = [X]
"""
function Model_Lorenz96_res(s_param::Setup_Param{FT, IT}, Q::Array{FT, 1}, θ::Array{FT, 1}, Φ::Function) where {FT<:AbstractFloat, IT<:Int}
    J, K = s_param.J, s_param.K
    F = s_param.F 
    
    # with ghost states for periodicity
    ng = 2
    Xg = zeros(FT, K+2*ng)
    
    Xg[ng+1:K+ng] .= Q[1:K]
    
    # update periodic boundary
    for i = 1:ng
        # ng + 1 - i -> K+ng + 1 - i
        # K+ng + i -> i + ng
        Xg[ng + 1 - i] = Xg[K+ng + 1 - i]
        Xg[K+ng + i] = Xg[i+ng]
    end
    
    dXg = zeros(FT, K+2*ng)
    for k = ng+1:K+ng
        dXg[k] = - Xg[k-1]*(Xg[k-2] - Xg[k+1]) - Xg[k] + F + Φ(Xg[k], θ)
    end
    
    dQ = similar(Q)
    dQ[1:K] .= dXg[ng+1:K+ng]
    
    return dQ
    
end

function run_Lorenz96(s_param::Setup_Param{FT, IT}, Q0::Array{FT,1}, θ=nothing, Φ=nothing) where {FT<:AbstractFloat, IT<:Int}
    RK_order = s_param.RK_order
    N_t, Δt = s_param.N_t, s_param.Δt
    Q = copy(Q0)
    
    if θ === nothing
        f = (t, Q) -> Multiscale_Lorenz96_res(s_param, Q)
    else
        f = (t, Q) -> Model_Lorenz96_res(s_param, Q, θ, Φ)
    end
    
    Qs = Explicit_Solve!(f, Q0, Δt, N_t; order = RK_order)
    
    return Qs
end

#=
type::String = "Statistics" or "Time-Series"
=#
function compute_obs(s_param::Setup_Param{FT, IT}, Qs::Array{FT, 2}) where {FT<:AbstractFloat, IT<:Int}
    
    N_t, Δt = s_param.N_t, s_param.Δt
    K = s_param.K
    obs_type, obs_p = s_param.obs_type, s_param.obs_p
    
    if obs_type == "Time-Series"
        Δobs = obs_p
        obs = Qs[1:K, Δobs:Δobs:N_t][:]
    elseif obs_type == "Statistics"
        # first and second moments of the first k components
        kmode = obs_p
        
        obs = zeros(FT, IT(kmode + kmode*(kmode+1)/2))
        for i = 1:kmode
            obs[i] = mean(Qs[i, :])
        end
        id = kmode
        for i = 1:kmode
            for j = 1:i
                id += 1
                obs[id] = mean(Qs[i, :].*Qs[j, :])
            end
        end
        
    else
        error("Observation type ", obs_type, " is not recognized")
    end
    
    return obs
end


function forward(s_param::Setup_Param{FT,IT}, θ::Array{FT,1}, Q0::Array{Float64, 1}, Φ::Function) where {FT<:AbstractFloat, IT<:Int}
    
    Qs = run_Lorenz96(s_param, Q0, θ, Φ)
  
    obs = compute_obs(s_param, Qs)
    return obs
end




# function Test_Lorenz96(s_param::Setup_Param, multiscale::Bool)
#     T, N_t, Δt = s_param.T, s_param.N_t, s_param.Δt
#     tt = Array(LinRange(0, T, N_t+1))
    
#     K, J = s_param.K, s_param.J
    
    
#     if multiscale
#         Random.seed!(42);
#         Q0 = [rand(Normal(0, 1.0), K) ; rand(Normal(0, 0.01), K*J)]
        
#         #Q0 = Array(LinRange(1.0, K*(J+1), K*(J+1)))
#         data = run_Lorenz96(s_param, Q0)
#         figure(1)
#         plot(tt, data[1, :], label = "slow")
#         plot(tt, data[K+1, :], label = "fast")
#         savefig("Sample_Traj.png"); close("all")
        
#         figure(2) # hist
#         hist(data[1:K,:][:], bins = 100, density = true, histtype = "step")
#         savefig("X_density.png"); close("all")
        
#         figure(3) # modeling term
#         # Xg[k] vs - h*c/d*(sum(Yg[ng+1:J+ng, k]))
#         h, c, d = s_param.h, s_param.c, s_param.d
#         X = (data[1:K, :])[:]
#         Φ_ref = -h*c/d * sum(reshape(data[K+1:end,:], J, K*size(data, 2)), dims=1)
#         scatter(X, Φ_ref, s = 0.1, c="grey")
#         xs = Array(LinRange(-10, 15, 1000))
        
#         fwilks = -(0.262 .+ 1.45*xs - 0.0121*xs.^2 - 0.00713*xs.^3 + 0.000296*xs.^4)
#         plot(xs, fwilks, label="Wilks")
        
#         fAMP = -(0.341 .+ 1.3*xs - 0.0136*xs.^2 - 0.00235*xs.^3)
#         plot(xs, fAMP, label="Arnold")
#         legend()
#         savefig("Closure.png"); close("all")
        
        
#     else
#         Random.seed!(123);
#         Q0 = rand(Normal(0, 1), K)
#         Φ = (x, θ) -> 0.0
#         θ = [0.0]
#         data = run_Lorenz96(s_param, Q0, θ, Φ)
#         plot(tt, data[1,:], label = "slow")
#         savefig("Sample_Traj_Simple.png"); close("all")
        
#         figure(2) # hist
#         hist(data[1:K, :][:], bins = 100, density = true, histtype = "step")
#         savefig("X_density_Simple.png"); close("all")
        
#     end
    
#     return data
# end



# obs_type, obs_p = "Statistics", 8
# T, Δt = 1000.0, 0.005

# s_param = Setup_Param(obs_type, obs_p,  T, Δt) 
# data = Test_Lorenz96(s_param, true)
# data = Test_Lorenz96(s_param, false)


# K,  J = 8, 32
# F,  h, c, b = 20.0, 1.0, 10.0, 10.0
# d = b
# T, Δt, RK_order = 1000.0, 0.005, 4
# obs_type, obs_p = "Statistics", 4

# # Set up cubic Hermite polynomials
# N_θ = 12
# θ0_mean = zeros(Float64, N_θ)  #rand(Normal(0, 1), N_θ)                    # 
# θθ0_cov = Array(Diagonal(fill(1.0, N_θ)))           # standard deviation
# Φ = ΦQP
# N_iter = 20
# α_reg = 1.0
# update_freq = 0


# # Setup simulation parameters
# lorenz96_param = Setup_Param(N_θ, K, J, F, h, c, b, d,
#                       T, Δt, RK_order, 
#                       obs_type, obs_p) 


# # Initialization
# Random.seed!(42);
# Q0_ref = [rand(Normal(0, 1.0), K) ; rand(Normal(0, 0.01), K*J)]
# Q0 = rand(Normal(0, 1.0), K)

# # Run Multiscale Lorenz96
# Qs_ref = run_Lorenz96(lorenz96_param, Q0_ref)




# y =  compute_obs(lorenz96_param, Qs_ref)
# noise_level = 0.05
# Random.seed!(123);
# for i = 1:length(y)
#     noise = rand(Normal(0, noise_level*y[i]))
#     y[i] += noise
# end
# Σ_η = Array(Diagonal(noise_level^2*y.^2)) 




# ukiobj = UKI_Run(lorenz96_param, (s_params,θ)->forward(s_params, θ, Q0, Φ), θ0_mean, θθ0_cov, y, Σ_η, α_reg, update_freq, N_iter)
# # compute the solution with prior
# Qs_prior = run_Lorenz96(lorenz96_param, Q0, θ0_mean, Φ)
# # compute the solution with UKI results
# Qs = run_Lorenz96(lorenz96_param, Q0, ukiobj.θ_mean[end], Φ)



# fig, ax = PyPlot.subplots(figsize=(6,6))
# nbin = 100
# ax.hist(Qs_ref[1:K, :][:], bins = nbin, color="black", density = true, histtype = "step", label="Truth")
# ax.hist(Qs_prior[1:K, :][:], bins = nbin, color="C5", density = true, histtype = "step", label="Prior")
# Qs = run_Lorenz96(phys_params, Q0, ukiobjs[3].θ_mean[end], Φ)
# ax.hist(Qs[1:K, :][:], bins = nbin, color="C3", density = true, histtype = "step", label="UKI")
# ax.set_xlabel("X")
# ax.legend()
# ax.set_title("X empirical density")
# fig.tight_layout()






# fig, ax = PyPlot.subplots(figsize=(6,6))
# h, c, d = phys_params.h, phys_params.c, phys_params.d
# X = (Qs_ref[:,1:K]')[:]
# Φ_ref = -h*c/d * sum(reshape(Qs_ref[:, K+1:end]', J, K*size(Qs_ref, 1)), dims=1)
# ax.scatter(X, Φ_ref, s = 0.1, c="grey")
# xs = Array(LinRange(-10, 15, 1000))
# fwilks = -(0.262 .+ 1.45*xs - 0.0121*xs.^2 - 0.00713*xs.^3 + 0.000296*xs.^4)
# ax.plot(xs, fwilks, c="C4", label="Wilks")
# fAMP = -(0.341 .+ 1.3*xs - 0.0136*xs.^2 - 0.00235*xs.^3)
# ax.plot(xs, fAMP, c="C5", label="Arnold")
# fDA = similar(xs)
# for i = 1:length(xs);  fDA[i] = Φ(xs[i], ukiobj.θ_mean[end]); end
# ax.plot(xs, fDA, c="C3", linestyle="--", marker="o", fillstyle="none", markevery=100, label="UKI")
# ax.set_xlabel("X")
# ax.set_ylabel("ψ(X)")
# ax.set_title(Closure)
# fig.tight_layout()















