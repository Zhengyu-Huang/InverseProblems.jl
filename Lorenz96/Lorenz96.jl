using Random
using Statistics
using Distributions
using PyPlot
mutable struct Params
    K::Int64
    J::Int64
    
    
    
    RK_order::Int64
    T::Float64
    NT::Int64
    ΔT::Float64
    
    obs_type::String
    obs_p::Int64

    n_data::Int64
    
    F::Float64
    h::Float64
    c::Float64
    d::Float64
    b::Float64
    
    
end

function Params(K::Int64 = 8, J::Int64 = 32, RK_order = 4, T = 100, ΔT = 0.005, obs_type = "Statistics", obs_p = 1) 
    F,  h, c, b = 20.0, 1.0, 10.0, 10.0
    # d = b the original Lorenz96, d = J modified Lorenz96 from 
    # "Earth System Modeling 2.0: A Blueprint for Models That Learn From Observations and Targeted High-Resolution Simulations"
    d = b
    
    NT = Int64(T/ΔT)

    if obs_type == "Statistics"
        kmode = obs_p
        n_data = kmode + kmode*(kmode + 1)/2
    elseif obs_type == "Time-Series"
        Δobs = obs_p
        n_data = K*Int64(NT/Δobs)
    else
        error("obs_type ", obs_type, " has not implemented, yet")
    end
    
    Params(K, J, RK_order, T, NT, ΔT, obs_type, obs_p, n_data, F, h, c, d, b)
end

function Rk_Update!(t::Float64, Δt::Float64, f::Function, Q::Array{Float64,1}; order::Int64)
    if order == 1
        k1 = Δt*f(t, Q)
        Q .+= k1
        
    elseif order == 2
        k1 = Δt*f(t, Q)
        k2 = Δt*f(t+Δt, Q+k1)
        Q .+= (k1 + k2)/2.0
        
    elseif order == 4
        k1 = Δt*f(t, Q)
        k2 = Δt*f(t+0.5*Δt, Q+0.5*k1)
        k3 = Δt*f(t+0.5*Δt, Q+0.5*k2)
        k4 = Δt*f(t+Δt, Q+k3)
        Q .+= (k1 + 2*k2 + 2*k3 + k4)/6
        
    end
end

function ΦNN(x::Float64, θ::Array{Float64, 1})
    n = Int64((length(θ)-1)/3)
    a, b, c, d = θ[1:n], θ[n+1:2n], θ[2n+1:3n], θ[3n+1]
    # σ(a x + b) ⋅ c + e
    return d + c' * tanh.(a*x+b)
end

function ΦRFF(x::Float64, θ::Array{Float64, 1})
    #=
    This method is not stable
    =#
    
end

function Φpoly(x::Float64, θ::Array{Float64, 1})
    #=
    This set of basis is not stable
    =#
    nθ = length(θ)
    xx = [x^i for i = 0:nθ-1]
    return θ' * xx
end


function ΦGP(x::Float64, θ::Array{Float64, 1})
    #=
    x ∈ [-20, 20]
    kernal k(x, x') = exp(- (x - x')^2/(2l^2))

    f = θi k(xi, x)
    =#
    x_l, x_h = -20.0, 20.0

    nθ = length(θ)
    nm = nθ - 1
    xx = Array(LinRange(x_l, x_h, nm))

    l = θ[nm+1]

    return θ[1:nm]' * exp.(-(xx .- x).^2/(2l^2))

end

function ΦLP(x::Float64, θ::Array{Float64, 1})
    #=
    x ∈ [x_l, x_h], x_l = -20 x_h = 20, piecewise linear polynomial
    =#
    x_l, x_h = -20.0, 20.0

    if x >= x_h 
        return θ[end]
    elseif x <= x_l
        return θ[1]
    end

    ne = length(θ) - 1

    Δx = (x_h - x_l)/ne
    int_val = ceil(Int64, (x - x_l)/Δx)
    ξ = (x - x_l - Δx*(int_val-1) )/Δx

    return (1 - ξ)*θ[int_val] + ξ*θ[int_val+1] 
end


# function ΦQP(x::Float64, θ::Array{Float64, 1})
#     #=
#     x ∈ [x_l, x_h], x_l = -20 x_h = 20, piecewise quadratic polynomial
#     =#
#     x_l, x_h = -20.0, 20.0
#     if x >= x_h 
#         return θ[end]
#     elseif x <= x_l
#         return θ[1]
#     end
#     ne = Int64((length(θ)-1)/2)
    
#     Δx = (x_h - x_l)/ne
#     int_val = ceil(Int64, (x - x_l)/Δx)
#     @assert(int_val <= ne)
#     ξ = (x - x_l - Δx*(int_val-1) )/Δx
#     if !(ξ>= 0 && ξ <= 1.0)
#         @info x, Δx, int_val, Δx*(int_val-1), ξ 
#     end

#     @assert(ξ>= 0 && ξ <= 1.0)

#     return (1-ξ)*(1-2*ξ)*θ[2*int_val-1]  +  4*ξ*(1-ξ)*θ[2*int_val]  +  ξ*(2*ξ-1)*θ[2*int_val+1]
# end


function ΦQP(x::Float64, θ::Array{Float64, 1})
    #=
    x ∈ [x_l, x_h], x_l = -20 x_h = 20, piecewise cubic polynomial
    o --- o --- o ...... o --- o
    1     3                    nθ-1=2ne+1
    2     4                    nθ  =2ne+2
    =#
    x_l, x_h = -20.0, 20.0
    if x >= x_h 
        return θ[end-1]
    elseif x <= x_l
        return θ[1]
    end
    ne = Int64((length(θ)-2)/2)
    
    Δx = (x_h - x_l)/ne
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
function Lorenz_96(phys_params::Params, Q::Array{Float64, 1})
    J, K = phys_params.J, phys_params.K
    F, h, c, d, b = phys_params.F, phys_params.h, phys_params.c, phys_params.d, phys_params.b
    
    ng = 2
    Xg = zeros(Float64, K+2*ng)
    Yg = zeros(Float64, J+2*ng, K+2*ng)
    
    
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
    
    dXg = zeros(Float64, K+2*ng)
    dYg = zeros(Float64, J+2*ng, K+2*ng)
    
    for k = ng+1:K+ng
        dXg[k] = - Xg[k-1]*(Xg[k-2] - Xg[k+1]) - Xg[k] + F - h*c/d*(sum(Yg[ng+1:J+ng, k]))
        for j = ng+1:J+ng
            dYg[j,k] = -c*b*Yg[j+1,k]*(Yg[j+2,k] - Yg[j-1,k]) - c*Yg[j,k]  + h*c/d*Xg[k]
        end
    end
    # @info c, b
    # @info "dXg ", dXg
    # @info "dYg ", dYg
    
    dQ = similar(Q)
    
    dQ[1:K] .= dXg[ng+1:K+ng]
    dQ[K+1:end] .= dYg[ng+1:J+ng, ng+1:K+ng][:]
    
    return dQ
    
end


"""
Q = [X]
"""
function Lorenz_96(phys_params::Params, Q::Array{Float64, 1}, θ::Array{Float64, 1}, Φ::Function)
    J, K = phys_params.J, phys_params.K
    F = phys_params.F 
    
    ng = 2
    Xg = zeros(Float64, K+2*ng)
    
    
    Xg[ng+1:K+ng] .= Q[1:K]
    
    # update periodic boundary
    for i = 1:ng
        # ng + 1 - i -> K+ng + 1 - i
        # K+ng + i -> i + ng
        Xg[ng + 1 - i] = Xg[K+ng + 1 - i]
        Xg[K+ng + i] = Xg[i+ng]
    end
    
    dXg = zeros(Float64, K+2*ng)
    for k = ng+1:K+ng
        dXg[k] = - Xg[k-1]*(Xg[k-2] - Xg[k+1]) - Xg[k] + F + Φ(Xg[k], θ)
    end
    
    dQ = similar(Q)
    dQ[1:K] .= dXg[ng+1:K+ng]
    
    return dQ
    
end

function Run_Lorenz96(phys_params::Params, Q0, θ=nothing, Φ=nothing)
    RK_order = phys_params.RK_order
    NT, ΔT = phys_params.NT, phys_params.ΔT
    Q = copy(Q0)

    if θ == nothing
        f = (t, Q) -> Lorenz_96(phys_params, Q)
    else
        f = (t, Q) -> Lorenz_96(phys_params, Q, θ, Φ)
    end
    
    data = zeros(Float64, NT, size(Q,1))
    
    for i = 1:NT
        Rk_Update!(i*ΔT, ΔT, f, Q; order=RK_order)
        data[i , :] .= Q

    end
    return data
end

#=
type::String = "Statistics" or "Time-Series"
=#
function Compute_Obs(phys_params::Params, data::Array{Float64, 2})
    NT, ΔT = phys_params.NT, phys_params.ΔT
    K = phys_params.K
    obs_type, obs_p = phys_params.obs_type, phys_params.obs_p

    if obs_type == "Time-Series"
        Δobs = obs_p
        obs = data[Δobs:Δobs:NT, 1:K][:]
    elseif obs_type == "Statistics"
        # first and second moments of the first k components
        kmode = 8

        obs = zeros(Float64, Int64(kmode + kmode*(kmode+1)/2))
        for i = 1:kmode
            obs[i] = mean(data[:,i])
        end
        id = kmode
        for i = 1:kmode
            for j = 1:i
                id += 1
                obs[id] = mean(data[:,i].*data[:,j])
            end
        end

    else
        error("Observation type ", obs_type, " is not recognized")
    end

    return obs
end






function Test_Lorenz96(multiscale::Bool)
    RK_order = 4
    T,  ΔT = 20.0, 0.005
    NT = Int64(T/ΔT)
    tt = Array(LinRange(ΔT, T, NT))
    
    #K, J = 36, 10
    K, J = 8, 32
    phys_params = Params(K, J, RK_order, T,  ΔT)
    
    if multiscale
        Random.seed!(42);
        Q0 = rand(Normal(0, 1), K*(J+1))
        #Q0 = Array(LinRange(1.0, K*(J+1), K*(J+1)))
        data = Run_Lorenz96(phys_params, Q0)
        figure(1)
        plot(tt, data[:,1], label = "slow")
        plot(tt, data[:,K+1], label = "fast")
        savefig("Sample_Traj.png"); close("all")

        figure(2) # hist
        hist(data[:,1:K][:], bins = 1000, density = true, histtype = "step")
        savefig("X_density.png"); close("all")

        figure(3) # modeling term
        # Xg[k] vs - h*c/d*(sum(Yg[ng+1:J+ng, k]))
        h, c, d = phys_params.h, phys_params.c, phys_params.d
        X = (data[:,1:K]')[:]
        Φ_ref = -h*c/d * sum(reshape(data[:, K+1:end]', J, K*size(data, 1)), dims=1)
        scatter(X, Φ_ref, s = 0.1, c="grey")
        xx = Array(LinRange(-10, 15, 1000))

        fwilks = -(0.262 .+ 1.45*xx - 0.0121*xx.^2 - 0.00713*xx.^3 + 0.000296*xx.^4)
        plot(xx, fwilks, label="Wilks")

        fAMP = -(0.341 .+ 1.3*xx - 0.0136*xx.^2 - 0.00235*xx.^3)
        plot(xx, fAMP, label="Arnold")
        legend()
        savefig("Closure.png"); close("all")


    else
        Random.seed!(123);
        Q0 = rand(Normal(0, 1), K)
        Φ = (x, θ) -> 0.0
        θ = [0.0]
        data = Run_Lorenz96(phys_params, Q0, θ, Φ)
        plot(tt, data[:,1], label = "slow")
        
    end
    
    return phys_params, data
end

#phys_params, data = Test_Lorenz96(true)
#phys_params, data = Test_Lorenz96(false)