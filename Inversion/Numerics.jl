function Explicit_Solve!(f::Function, Q0::Array{FT,1}, Δt::FT, N_t::IT; order::IT) where {FT<:AbstractFloat, IT<:Int}
    
    N_q = length(Q0)
    Qs = zeros(FT, N_q, N_t + 1)
    Qs[:, 1] .= Q0 
    t = 0
    if order == 1
        for i = 1:N_t
            k1 = Δt*f(t, Qs[:, i])
            Qs[:, i+1] .= Qs[:, i] + k1
            t += Δt
        end
    elseif order == 2
        for i = 1:N_t
            k1 = Δt*f(t, Qs[:, i])
            k2 = Δt*f(t+Δt, Qs[:, i]+k1)
            Qs[:, i+1] .= Qs[:, i] + (k1 + k2)/2.0
            t += Δt
        end
    elseif order == 4
        for i = 1:N_t
            k1 = Δt*f(t, Qs[:, i])
            k2 = Δt*f(t+0.5*Δt, Qs[:, i]+0.5*k1)
            k3 = Δt*f(t+0.5*Δt, Qs[:, i]+0.5*k2)
            k4 = Δt*f(t+Δt, Qs[:, i]+k3)
            Qs[:, i+1] .= Qs[:, i] + (k1 + 2*k2 + 2*k3 + k4)/6
            t += Δt
        end
    end
    
    return Qs
end