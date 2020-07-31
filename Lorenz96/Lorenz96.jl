using Random
using Distributions
using PyPlot
mutable struct Params
    K::Int64
    J::Int64
    
    
    
    RK_order::Int64
    T::Float64
    NT::Int64
    ΔT::Float64
    
    Δobs::Int64
    
    F::Float64
    h::Float64
    c::Float64
    d::Float64
    b::Float64
    
    
end

function Params(K::Int64 = 8, J::Int64 = 32, RK_order = 4, T = 100, ΔT = 0.005, Δobs = 1) 
    F,  h, c, b = 20.0, 1.0, 10.0, 10.0
    # d = b the original Lorenz96, d = J modified Lorenz96 from 
    # "Earth System Modeling 2.0: A Blueprint for Models That Learn From Observations and Targeted High-Resolution Simulations"
    d = b
    
    NT = Int64(T/ΔT)
    
    Params(K, J, RK_order, T, NT, ΔT, Δobs, F, h, c, d, b)
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
        dXg[k] = - Xg[k-1]*(Xg[k-2] - Xg[k+1]) - Xg[k] + F - Φ(Xg[k], θ)
    end
    
    dQ = similar(Q)
    dQ[1:K] .= dXg[ng+1:K+ng]
    
    return dQ
    
end

function Run_Lorenz96(phys_params::Params, Q0, θ=nothing, Φ=nothing)
    RK_order = phys_params.RK_order
    NT, Δobs, ΔT = phys_params.NT, phys_params.Δobs, phys_params.ΔT
    Q = copy(Q0)
    if θ == nothing
        f = (t, Q) -> Lorenz_96(phys_params, Q)
    else
        f = (t, Q) -> Lorenz_96(phys_params, Q, θ, Φ)
    end
    
    data = zeros(Float64, Int64(NT/Δobs), size(Q,1))
    
    for i = 1:NT
        Rk_Update!(i*ΔT, ΔT, f, Q; order=RK_order)
        if i%Δobs == 0
            data[Int64(i/Δobs) , :] .= Q
        end
    end
    
    return data
end

function Test_Lorenz96(multiscale::Bool)
    RK_order = 4
    T,  ΔT = 20.0, 0.005
    NT = Int64(T/ΔT)
    tt = Array(LinRange(ΔT, T, NT))
    Δobs = 1
    
    #K, J = 36, 10
    K, J = 8, 32
    phys_params = Params(K, J, RK_order, T,  ΔT, Δobs)
    
    if multiscale
        Random.seed!(13);
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
        Φ = -h*c/d * sum(reshape(data[:, K+1:end]', J, K*size(data, 1)), dims=1)
        scatter(X, Φ, s = 0.1, c="grey")
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

phys_params, data = Test_Lorenz96(true)
#phys_params, data = Test_Lorenz96(false)