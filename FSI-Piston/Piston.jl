mutable struct Structure{FT<:AbstractFloat}
    """
    m∂t² d + c ∂t d + k d = f_ext
    """   
    Q_h::Array{FT, 1}      # d, u, a at half step n-1/2
    Q::Array{FT, 1}        # d, u, a at step n
    Q_hm1::Array{FT, 1}    # d, u, a at step n-1/2-1
    
    x0::FT
    
    ms::FT
    cs::FT
    ks::FT

    

    #"AEROELASTIC" or "FORCED"
    motion::String

    forced_motion_func::Union{Nothing, Function}

    t::FT
end



# This is for aeroelastic simulation (update Δt = half of the time step)
# Compute Q_h and Q
function A6_First_Half_Step!(structure::Structure, fext::FT, t::FT, Δt::FT) where {FT<:AbstractFloat}
    Δt_h = Δt/2
    ms, cs, ks = structure.ms, structure.cs, structure.ks
    ds, vs, as = structure.Q_h

    structure.Q_hm1 .= structure.Q_h

    # ms as + cs vs + ks ds = f 
    as_h = (fext - ks*ds - cs*vs)/ms
    vs_h   = vs + Δt_h*as_h
    ds_h   = ds + Δt_h*(vs + vs_h)/2

    structure.Q_h = [ds_h, vs_h, as_h]

    # prediction
    structure.Q = [ds_h + Δt_h/2*vs_h + Δt_h/8*(vs_h - vs) , 1.5*vs_h - 0.5*vs, NaN]
    structure.t = structure.t + Δt_h
end


function Send_Embeded_Disp(structure::Structure{FT}, t::FT, Δt::FT) where {FT<:AbstractFloat}
    if structure.motion == "AEROELASTIC"
        ds, vs, _ = structure.Q
    else
        ds, vs, _ = structure.forced_motion_func(t + Δt)
    end

    return [ds + structure.x0, vs]
end


function Structure_Time_Advance!(structure::Structure{FT}, fext::FT, t::FT, Δt::FT) where {FT<:AbstractFloat}

    ms, cs, ks = structure.ms, structure.cs, structure.ks
    
    ds_h,   vs_h,   as_h = structure.Q_h
    ds_hm1, vs_hm1, as_hm1 = structure.Q_hm1

    structure.Q_hm1 .= structure.Q_h

    
    # ms as + cs vs + ks ds = f 
    as_h  = (fext - ks*(ds_hm1 + Δt/2*vs_hm1 + Δt^2/8*as_hm1) - cs*(vs_hm1 + Δt/4*as_hm1) - ms*as_hm1/2)/(ms/2 + cs*Δt/4 + ks*Δt^2/8)
    vs_h  = vs_hm1 + Δt/2*(as_h + as_hm1)
    ds_h  = ds_hm1 + Δt*vs_hm1 + Δt^2/4*(as_hm1 + as_h)


    structure.Q_h =  [ds_h, vs_h, as_h]

    # prediction
    structure.Q = [ds_h + Δt/2*vs_h + Δt/8*(vs_h - vs_hm1) , 1.5*vs_h - 0.5*vs_hm1, NaN]
    structure.t = structure.t + Δt

end


