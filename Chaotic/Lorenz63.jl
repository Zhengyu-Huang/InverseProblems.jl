using JLD2
using Statistics
using LinearAlgebra
include("../Inversion/Plot.jl")





function f(x::Array{FT,1}, σ::FT, r::FT, β::FT) where {FT<:AbstractFloat}
    return [σ*(x[2]-x[1]); x[1]*(r-x[3])-x[2]; x[1]*x[2]-β*x[3]] 
end

function df_dx(x::Array{FT,1}, σ::FT, r::FT, β::FT) where {FT<:AbstractFloat}
    return  [-σ        σ       0.0; 
             (r-x[3])  -1.0    -x[1]; 
             x[2]      x[1]    -β]
end

function df_dθ(x::Array{FT,1}, σ::FT, r::FT, β::FT) where {FT<:AbstractFloat}
    return [(x[2]-x[1]) 0.0     0.0; 
            0.0        x[1]     0.0; 
            0.0         0.0    -x[3]] 
end

# integrate Lorenz63 with forward Euler method
function compute_Lorenz63_FE(x0::Array{FT,1}, θ::Array{FT,1}, Δt::FT, N_t::IT) where {FT<:AbstractFloat, IT<:Int}

    N_x = length(x0)
    
    σ, r, β = θ
    
    xs = zeros(N_x, N_t+1)
    xs[:, 1] = x0
    for i = 1:N_t
        xs[:, i+1] = xs[:, i] + Δt*f(xs[:, i], σ, r, β)
    end
    
    return xs
end

function compute_Lorenz63_RK4(x0::Array{FT,1}, θ::Array{FT,1}, Δt::FT, N_t::IT) where {FT<:AbstractFloat, IT<:Int}
    N_x = length(x0)

    σ, r, β = θ
    
    xs = zeros(N_x, N_t+1)
    xs[:,1] = x0
    for i=1:N_t
        k1 = Δt*f(xs[:,i], σ, r, β)
        k2 = Δt*f(xs[:,i] + k1/2, σ, r, β)
        k3 = Δt*f(xs[:,i] + k2/2, σ, r, β)
        k4 = Δt*f(xs[:,i] + k3, σ, r, β)
        
        xs[:,i+1] = xs[:,i] + k1/6 + k2/3 + k3/3 + k4/6
    end
    
    return xs
end

function f_mean_x3(x::Array{FT,1}) where {FT<:AbstractFloat}
    J = x[3]
    pJ_px = FT.([0;0;1])
    return J, pJ_px
end

# Compute objective function J, and dJ/dxs
function compute_J(xs::Array{FT,2}, func::Function, j::IT, k::IT) where {FT<:AbstractFloat, IT<:Int}
    # J is defined as the average 
    # J = ∑_{i=j}^{k} func(x(t_i)) /(k-j+1)
    # return J and dJ/dxs

    
    N_x, N_t = size(xs, 1), size(xs, 2) - 1

    J = 0.0
    ∂J_∂xs = zeros(N_x, N_t+1)
    
    for i = j:k
        J_i, ∂J_∂xs_i = func(xs[:,i])
        J +=  J_i
        ∂J_∂xs[:,i] = ∂J_∂xs_i
    end
    
    
    ∂J_∂xs ./= (k-j+1)
    J /= (k-j+1)
    
    return J, ∂J_∂xs
end

# Compute objective function dJ/dθ
function compute_gradient_adjoint(θ::Array{FT,1}, xs::Array{FT,2}, ∂J_∂xs::Array{FT,2}, Δt::FT) where {FT<:AbstractFloat, IT<:Int}
    

    N_x, N_t = size(xs, 1), size(xs, 2) - 1
    σ, r, β = θ
    N_θ = length(θ)
    λs = zeros(N_x, N_t+1)
    λs[:,N_t+1] .= ∂J_∂xs[:,N_t+1]
    
    for i=N_t:-1:1
        λs[:,i] = ∂J_∂xs[:,i] + λs[:,i+1] + Δt*df_dx(xs[:,i], σ, r, β)'*λs[:,i+1]   
    end
    
    dJ_dθ = zeros(FT, N_θ)
    for i=2:N_t+1
        dJ_dθ .+= df_dθ(xs[:,i-1], σ, r, β)' * λs[:,i]*Δt
    end
    
    #J = ∑_j^k f(xi)/(k-j+1)
    return dJ_dθ
end


function compute_dx3_dr_adjoint(rs = Array(LinRange(0, 50, 1001))) 
    FT = Float64
    Tobs = 20.0
    Tspinup = 30.0
    Δt = 0.001
    
    T = Tobs + Tspinup
    N_t = Int64(T/Δt)
    N_burn_in = Int64(Tspinup/Δt)
    
    N_r = length(rs)
    
    mean_x3s = zeros(FT, N_r)
    dmean_x3_drs = zeros(FT, N_r)

    x0 = [-8.67139571762; 4.98065219709; 25]
    for i = 1:N_r
        θ = [10.0; rs[i]; 8.0/3.0]
        xs = compute_Lorenz63_FE(x0, θ, Δt, N_t)
        
        J, ∂J_∂xs = compute_J(xs, f_mean_x3,  N_burn_in + 1, N_t + 1)
        dJ_dθ = compute_gradient_adjoint(θ, xs, ∂J_∂xs, Δt)
        mean_x3s[i], dmean_x3_drs[i] = J, dJ_dθ[2]
    end
    
    return  rs, mean_x3s, dmean_x3_drs
end




function fd_test()
    T = 20
    Δt = 0.01
    N_t = Int64(T/Δt)
    #σ, r, β = 10.0, 28.0, 8.0/3.0
    # non-chaotic case
    σ, r, β = 1.0, 2.0, 3.0
    
    θ = [σ; r; β]
    N_θ = length(θ)
    δθ = rand(N_θ)
    #δθ = [1.0;1.0;1.0;1.0;1.0;1.0]
    #δθ = [1.0;2.0;3.0;4.0;5.0;6.0]
    ε = 1.0e-4
    x0 = [-8.67139571762; 4.98065219709; 25]

    xs = compute_Lorenz63_FE(x0, θ, Δt, N_t)
    J, ∂J_∂xs = compute_J(xs, f_mean_x3, 1, N_t+1)
    dJ_dθ = compute_gradient_adjoint(θ, xs, ∂J_∂xs, Δt)
    
    θ_p = θ + δθ*ε
    xs_p = compute_Lorenz63_FE(x0, θ_p, Δt, N_t)
    J_p, _ = compute_J(xs_p, f_mean_x3, 1, N_t+1)
    
    
    θ_m = θ - δθ*ε
    xs_m = compute_Lorenz63_FE(x0, θ_m, Δt, N_t)
    J_m, _ = compute_J(xs_m, f_mean_x3, 1, N_t+1)
    
    @info "ε = ", ε, " finite difference error is ", (J_p - J_m)/(2*ε) .- dJ_dθ'*δθ
    
end








function adjoint_plot(rs, mean_x3s, dmean_x3_drs)
    
    
    r_max = 50.0
    N_r = length(mean_x3)
    rs = Array(LinRange(0, r_max, N_r))
    
    PyPlot.figure()
    PyPlot.plot(rs, mean_x3s, "--", fillstyle="none")
    PyPlot.xlabel(L"r")
    PyPlot.ylabel(L"\mathcal{G}(r)")
    PyPlot.grid("on")
    PyPlot.tight_layout()
    
    PyPlot.figure()
    semilogy(rs, abs.(dmean_x3_drs), "or", markersize=1)
    xlabel("\$r\$")
    ylabel(L"|d\mathcal{G}(r)|")
    grid("on")
    tight_layout()
    savefig("Lorenz_dJ.pdf")
    close("all")
    
    #filter the result
    dr = rs[2] - rs[1]
    filtered_x3_arr = copy(mean_x3)
    filtered_dx3_dr_arr = copy(dmean_x3_dr)
    
    σr = sqrt(0.22714463033782045)
    filter_Δ = Int64(ceil(3*σr/dr))
    for i = 1:N_r
        if (i > filter_Δ && i < N_r - filter_Δ)
            filtered_x3 = 0.0
            weights = 0.0
            filtered_dx3_dr = 0.0
            filtered_dx3_dr_numerator = 0.0
            filtered_dx3_dr_denominator = 0.0
            
            for j = i-filter_Δ:i+filter_Δ
                weight = 1/(sqrt(2*pi)*σr)*exp(-(rs[j] - rs[i])^2/(2*σr^2))
                filtered_x3 += mean_x3[j]*weight
                filtered_dx3_dr_numerator += (rs[j] - rs[i])*(mean_x3[j] - mean_x3[i])*weight
                filtered_dx3_dr_denominator += (rs[j] - rs[i])*(rs[j] - rs[i])*weight
                weights += weight

                
            end

            
            
            filtered_x3_arr[i] =  filtered_x3/weights 
            filtered_dx3_dr_arr[i] =  filtered_dx3_dr_numerator/filtered_dx3_dr_denominator
        end
    end
    
    plot(rs[filter_Δ+1:N_r - filter_Δ-1], filtered_x3_arr[filter_Δ+1:N_r - filter_Δ-1], "--", fillstyle="none")
    xlabel(L"r")
    ylabel(L"\mathcal{F}\mathcal{G}(r)")
    grid("on")
    tight_layout()
    savefig("Filtered_Lorenz_J.pdf")
    close("all")
    
    plot(rs[filter_Δ+1:N_r - filter_Δ-1], filtered_dx3_dr_arr[filter_Δ+1:N_r - filter_Δ-1], "or", markersize=1)
    xlabel("\$r\$")
    ylabel(L"\mathcal{F}d\mathcal{G}(r)")
    grid("on")
    tight_layout()
    savefig("Filtered_Lorenz_dJ.pdf")
    close("all")
    
end







