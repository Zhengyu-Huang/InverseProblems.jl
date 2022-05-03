
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


function ensemble(s_param, θ_ens::Array{FT,3}, forward::Function)  where {FT<:AbstractFloat}
    
    N_modes, N_ens,  N_θ = size(θ_ens)
    N_y = s_param.N_y
    g_ens = zeros(FT, N_modes, N_ens,  N_y)
    
    for im = 1:N_modes
        Threads.@threads for i = 1:N_ens
            θ = θ_ens[im, i, :]
            g_ens[im, i, :] .= forward(s_param, θ)
        end
    end
    
    return g_ens
end