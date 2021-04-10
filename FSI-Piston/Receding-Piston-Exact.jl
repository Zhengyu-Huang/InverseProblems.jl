using LinearAlgebra
using PyPlot
function f_piston(τ, x, t , a0, L, γ, structure::Function)
    ds, vs, _ = structure(τ)
    f = ((γ + 1)/2*vs - a0)*(t - τ) + ds + L/2 - x
    return f
end

function df_piston(τ, x, t , a0, L, γ, structure::Function)
    ds, vs, as = structure(τ)
    df = -((γ + 1)/2*vs - a0) + (γ + 1)/2*(t - τ)*as + vs
    return df
end
"""
Compute exact solution of the receding piston problem with simple 
wave assumptions for x ∈ [0, L] and the piston is initialized at L/2
"""
function Receding_Piston_Point(x, t, fluid_info, L, structure::Function)
    
    γ, ρ_l, v_l, p_l = fluid_info
    
    ds0, vs0, as0 = structure(0.0)
    ds, vs, as = structure(t)
    x0 = L/2
    a0 = sqrt(γ*p_l/ρ_l)
    K = sqrt(γ*p_l/ρ_l^γ)
    fan_start, fan_end = x0 - t*a0, x0 + t*(vs0*(γ + 1)/2 - a0)

    
    
    ρ , v = 0.0, 0.0
    if (x < fan_start)
        
        ρ , v = ρ_l, v_l
        
    elseif (fan_start <= x && x <= fan_end)
        
        ρ  = ((2a0 - (γ - 1)*(x - L/2)/t)/(K*(γ + 1)))^(2/(γ - 1))
        v   = 2.0*((x - L/2.0)/t + a0)/(γ + 1)
        
    elseif (fan_end <= x && x <= ds + x0)
        
        τ = 0.0
        i = 1
        for i  = 1:100
            fτ =   f_piston(τ, x, t, a0, L, γ, structure)
            dfτ = df_piston(τ, x, t, a0, L, γ, structure)
            Δτ = fτ/dfτ
            
            while (τ - Δτ > t)
                Δτ/= 2.0
            end

            τ -= Δτ
            
            if (abs(fτ) < 1.0e-12 )
                break
            end
        end
        if (i == 100)
            error("Divergence Newtons method")
            
            
            # Bisection
            l, r = 0.0, t
            while true
                τ = (l + r)/2.0
                f = f_piston(τ, x, t, a0, L, γ, structure)
                if (abs(f) < 1.0e-15 )
                    break
                elseif (f > 0)
                    r = tau
                else
                    l = tau
                end
            end
        end
        
        xs, vs, _ = structure(τ)
        v = vs
        ρ  = ((a0 - 0.5*(γ - 1)*vs)/ K)^(2.0/(γ - 1.0))
        
    else
        ρ , v = 0.0, 0.0
    end
    
    p = p_l*(ρ /ρ_l)^γ
    
    return [ρ , v,  p]
    
end
    

function Exact_Solution(t,  mesh_info, fluid_info, structure)
    
    L, N  = mesh_info
    N = Int64(N)
    γ, ρ_l, v_l, p_l = fluid_info
    
    Δx = L/N
    
    xx = LinRange(Δx/2, L - Δx/2, N)
    
    V = zeros(Float64, 3, N)
    
    for i = 1:N
        
        V[:,i] = Receding_Piston_Point(xx[i], t, fluid_info, L, structure)
        
    end
    
    return xx, V

end
            


