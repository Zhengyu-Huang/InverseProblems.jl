# J = x_j[ix]^m + x_{j+1}[ix]^m + ... + x_{k}[ix]^m
# x_1 = μ0
# x_{i+1} = x_i + Δt f(x_i, μ1), i = 1, 2, ... nt
# here μ0 is the initial condition, and μ1 is the parameter [σ=10.0; r=28.0; β=8.0/3.0]
# 

function f(x::Array{Float64,1}, σ::Float64, r::Float64, β::Float64)
    return [σ*(x[2]-x[1]); x[1]*(r-x[3])-x[2]; x[1]*x[2]-β*x[3]] 
end

function df_du(x::Array{Float64,1}, σ::Float64, r::Float64, β::Float64)
    return  [-σ        σ       0.0; 
             (r-x[3])  -1.0    -x[1]; 
              x[2]      x[1]    -β]
end

function df_dμ(x::Array{Float64,1}, σ::Float64, r::Float64, β::Float64)
    return [(x[2]-x[1]) 0.0    0.0; 
             0.0        x[1]   0.0; 
             0.0        0.0   -x[3]] 
    
end

function foward(μ::Array{Float64,1}, Δt::Float64, nt::Int64)
    @assert(length(μ) == 6)
    nx = 3
    x0 = μ[1:nx]
    σ,r, β = μ[nx+1:6]
    
    xs = zeros(nx, nt+1)
    xs[:,1] = x0
    for i=2:nt+1
        xs[:,i] = xs[:,i-1] + Δt*f(xs[:,i-1], σ, r, β)
    end
    
    return xs
end

function compute_J_helper(x::Array{Float64,1})
    # todo hard coded here ix, m
    ix, m = 1, 1
    nx = length(x)
    J = x[ix]^m
    pJ_px = zeros(nx)
    pJ_px[ix] = m*x[ix]^(m-1)
    return J, pJ_px
end

function compute_J(xs::Array{Float64,2}, μ::Array{Float64,1}, j::Int64, k::Int64)
    # J is defined as the sum 
    # J = x_j[ix]^m + x_j+1[ix]^m ... + x_k[ix]^m /(k-j+1)
    # pJ/pμ = 0
    # return J and dJ/dxs
    nμ = 6
    @assert(length(μ) == nμ)

    nx, nt = size(xs, 1), size(xs, 2) - 1
    σ,r, β = μ[nx+1:nμ]
    
    J = 0.0
    pJ_pxs = zeros(nx, nt+1)
    
    for i = j:k
        J_i, pJ_pxs_i = compute_J_helper(xs[:,i])
        J +=  J_i
        pJ_pxs[:,i] = pJ_pxs_i
    end
    
    
    pJ_pxs ./= (k-j+1)
    J /= (k-j+1)
    

    return J, pJ_pxs
    
end

function adjoint(μ::Array{Float64,1}, xs::Array{Float64,2}, pJ_pxs::Array{Float64,2}, Δt::Float64, nt::Int64)
    nμ = 6
    @assert(length(μ) == nμ)
    nx = 3
    x0 = μ[1:nx]
    σ, r, β = μ[nx+1:nμ]


    lambdas = zeros(nx, nt+1)
    lambdas[:,nt+1] .= pJ_pxs[:,nt+1]
    
    for i=nt:-1:1
        lambdas[:,i] = pJ_pxs[:,i] + lambdas[:,i+1] + Δt*df_du(xs[:,i], σ, r, β)'*lambdas[:,i+1]   
    end
    
    dJ_dμ = zeros(nμ)
    for i=2:nt+1
        dJ_dμ[1:nx] .+= df_dμ(xs[:,i-1], σ, r, β)' * lambdas[:,i]*Δt
    end
    
    dJ_dμ[nx+1:nμ] = lambdas[:,1]
    
    #J = ∑_j^k f(xi)/(k-j+1)
    return dJ_dμ
end

function fd_test()
    T = 1
    Δt = 0.01
    nt = Int64(T/Δt)
    σ, r, β = 10.0, 28.0, 8.0/3.0

    #σ, r, β = 1.0, 2.0, 3.0
    
    μ = [-8.67139571762; 4.98065219709; 25; σ; r; β]
    nμ = length(μ)
    #δμ = rand(nμ)
    δμ = [1.0;1.0;1.0;1.0;1.0;1.0]
    ε = 1.0e-4

    xs = foward(μ, Δt, nt)
    J, pJ_pxs = compute_J(xs, μ, 1, nt+1)
    dJ_dμ = adjoint(μ, xs, pJ_pxs, Δt, nt)

    @info μ, J, dJ_dμ 

    μ_p = μ + δμ*ε
    xs_p = foward(μ_p, Δt, nt)
    J_p, _ = compute_J(xs_p, μ_p, 1, nt+1)


    μ_m = μ - δμ*ε
    xs_m = foward(μ_m, Δt, nt)
    J_m, _ = compute_J(xs_m, μ_m, 1, nt+1)
        
    @info "fd error is ", (J_p - J_m)/(2*ε) .- dJ_dμ'*δμ

end

fd_test()