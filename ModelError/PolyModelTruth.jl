using Random
using Distributions
using LinearAlgebra
using PyPlot
include("../RExKI.jl")



# Truth model
function true_G(x::Float64)
    val = 4 + 3x + 2x^2 + x^3 + cos(π * x)
    return val
end

# Misspecified model
function poly_model(x::Float64, θ::Array{Float64, 1})
    val = 0.0
    for i = 1:length(θ)
        val += θ[i]*x^(i-1)
    end
    return val
end


function data_gen(N_θ::Int64, N_y::Int64, σ_η::Float64 = 0.1; seed::Int64 = 123)
    # generate data
    xs_train = Array(LinRange(-2, 2, N_y))
    G = hcat([xs_train.^(i-1) for i = 1:N_θ]...)
    
    ys_train_ref = [true_G(xs_train[i]) for i = 1:length(xs_train)]
    
    y = copy(ys_train_ref)
    Random.seed!(seed); 
    for i = 1:length(y)
        noise = σ_η * rand(Normal(0, 1))
        y[i] += noise
    end
    Σ_η = Array(Diagonal(fill(σ_η^2, N_y)))
    
    return xs_train, y, G, Σ_η
end




function compute_posterior(y::Array{Float64, 1}, G::Array{Float64, 2}, Σ_η::Array{Float64, 2}, 
    m0::Union{Array{Float64, 1} , Nothing}=nothing, Σ0::Union{Array{Float64, 2} , Nothing}=nothing)
    if m0 === nothing
        # compute posterior distribution with an uninformative prior
        m = (G'*(Σ_η\G))\(G'*(Σ_η\y))
        C = inv(G'*(Σ_η\G))
    else
        # compute posterior distribution with prior m0,Σ0
        m = m0 + Σ0*G'*((Σ_η + G*Σ0*G')\(y - G*m0))
        C = Σ0 - Σ0*G'*((Σ_η + G*Σ0*G')\(G*Σ0))
    end
    return m, C
end







function prediction(xs_train, ys_train, xs_test, θ_mean, θθ_cov; color = "red", marker="*")
    
    N_test = length(xs_test)
    
    
    
    ys_test_pred = [poly_model(xs_test[i], θ_mean) for i = 1:N_test]
    ys_test_cov = zeros(Float64, N_test)
    for i = 1:N_test
        g = [xs_test[i]^(j-1) for j = 1:N_θ]
        ys_test_cov[i] = g' * θθ_cov * g
    end
    ys_test_std = sqrt.(ys_test_cov)
    
    plot(xs_test, ys_test_pred, "-", color = color, marker = marker, fillstyle="none", markevery=20, label="Ny = $(length(ys_train))")
    plot(xs_test, ys_test_pred + 3ys_test_std, "--r", color = color)
    plot(xs_test, ys_test_pred - 3ys_test_std, "--r", color = color)
    
    
    
    xlabel("X")
    ylabel("Y")
    legend()
    
end


# Σ_η = 0.1^2 I
N_θ = 4
σ_η = 0.1


N_y = 10 
xs_train, y, G, Σ_η = data_gen(N_θ, N_y, σ_η)
m0, Σ0 = nothing, nothing
m, C = compute_posterior(y, G, Σ_η, m0, Σ0)
xs_test = Array(LinRange(-2, 2, 100))
ys_test_ref = [true_G(xs_test[i]) for i = 1:length(xs_test)]
figure(figsize=(5,4))
plot(xs_test, ys_test_ref, "--o", color="black", fillstyle="none", label = "Reference", markevery=10)
prediction(xs_train, y, xs_test, m, C, color="C2", marker = "s")


N_y = 100
xs_train, y, G, Σ_η = data_gen(N_θ, N_y, σ_η)
m0, Σ0 = nothing, nothing
m, C = compute_posterior(y, G, Σ_η, m0, Σ0)
xs_test = Array(LinRange(-2, 2, 100))
prediction(xs_train, y, xs_test, m, C, color="red")

tight_layout()
savefig("Cubic-0.1.pdf")


# Σ_η = 0.1^2 I
N_θ = 4
σ_η = 0.1


N_y = 10 
xs_train, y, G, Σ_η = data_gen(N_θ, N_y, σ_η)
Σ_η *= 100
m0, Σ0 = nothing, nothing
m, C = compute_posterior(y, G, Σ_η, m0, Σ0)
xs_test = Array(LinRange(-2, 2, 100))
ys_test_ref = [true_G(xs_test[i]) for i = 1:length(xs_test)]
figure(figsize=(5,4))
plot(xs_test, ys_test_ref, "--o", color="black", fillstyle="none", label = "Reference", markevery=10)
prediction(xs_train, y, xs_test, m, C, color="C2", marker = "s")


N_y = 100
xs_train, y, G, Σ_η = data_gen(N_θ, N_y, σ_η)
m0, Σ0 = nothing, nothing
m, C = compute_posterior(y, G, Σ_η, m0, Σ0)
xs_test = Array(LinRange(-2, 2, 100))
prediction(xs_train, y, xs_test, m, C, color="red")

tight_layout()
savefig("Cubic-1.pdf")