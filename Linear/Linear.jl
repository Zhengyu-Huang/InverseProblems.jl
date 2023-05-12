mutable struct Setup_Param{MAT, IT<:Int}
    θ_names::Array{String,1}
    G::MAT
    N_θ::IT
    N_y::IT
end

function Setup_Param(G, N_θ::IT, N_y::IT) where {IT<:Int}
    return Setup_Param(["θ"], G, N_θ, N_y)
end


function forward(s_param::Setup_Param, θ::Array{FT, 1}) where {FT<:AbstractFloat}
    G = s_param.G 
    return G * θ
end

function forward_aug(s_param::Setup_Param, θ::Array{FT, 1}) where {FT<:AbstractFloat}
    G = s_param.G 
    return [G * θ; θ]
end


function compute_Φ(s_param::Setup_Param, θ::Array{FT, 1},  y::Array{FT, 1}, Σ_η::Array{FT, 2}, μ0::Array{FT, 1}, Σ0::Array{FT, 2}) where {FT<:AbstractFloat} 
    Φ   = 1/2*(y - G * θ)*(Σ_η\(y - G * θ)) + 1/2*(μ0 - θ)*(Σ0\(μ0 - θ))
    ∇Φ  = -G * (Σ_η\(y - G * θ)) - Σ0\(μ0 - θ)
    ∇²Φ = G * (Σ_η\G) + inv(Σ0)
    return Φ, ∇Φ, ∇²Φ
end