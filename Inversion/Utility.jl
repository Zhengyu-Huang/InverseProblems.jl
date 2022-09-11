
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


# Paper: Gradient Flows for Sampling

function mean_estimation_2d(nf, f, logρ , xx, yy) 
    # compute E f = \sum f(xi, yi) rho(xi, yi) / \sum rho(xi, yi)
    nx, ny = length(xx), length(yy)
    Ef, Eρ = zeros(nx, ny, nf), zeros(nx, ny)
    
    for i = 1:nx
        for j = 1:ny
            Ef[i, j, :] = f([xx[i], yy[j]])
            Eρ[i ,j] = exp(logρ([xx[i], yy[j]]))
        end
    end
    return dropdims(sum(Ef.*Eρ, dims=(1,2)), dims=(1,2))/sum(Eρ)
end


function cos_error_estimation_particle(logρ, xx, yy, ω, b ) 
    # compute E f = \sum f(xi, yi) rho(xi, yi) / \sum rho(xi, yi)
    
    n = length(b)
    nx, ny = length(xx), length(yy)
    Ef, Eρ = zeros(nx, ny, n), zeros(nx, ny)
    
    for i = 1:nx
        for j = 1:ny
            Ef[i, j, :] = cos.(ω * [xx[i], yy[j]] + b)
            Eρ[i ,j] = exp(logρ([xx[i], yy[j]]))
        end
    end
    cos_ref = dropdims(sum(Ef.*Eρ, dims=(1,2)), dims=(1,2))/sum(Eρ)
    
    return cos_ref
end

function cos_error_estimation_particle(m_oo, C_oo, ω, b ) 
    return Diagonal(exp.(-0.5*ω*C_oo*ω')) * cos.(ω*m_oo + b)
end

function cos_error_estimation_particle(ps, ω, b )     
    n = length(b)
    N_ens= size(ps, 1)
    Ef = zeros(n)
    for i = 1:N_ens
        Ef += cos.(ω * ps[i, :] + b)
    end
    Ef = Ef/N_ens
    
    return Ef
end