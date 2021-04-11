using LinearAlgebra

"""
Convert primitive variables to conservative variables
ρ, v, p   =>   ρ, ρv, E = ρv²/2 + ρCᵥT
"""
function Pri_To_Conser(prim::Array{FT,1}, γ::FT) where {FT<:AbstractFloat}
    ρ, v, p = prim
    return  [ρ; ρ*v; ρ*v*v/2 + p/(γ - 1)]
end


"""
Convert primitive variables to conservative variables
ρ, v, p   =>   ρ, ρv, E = ρv²/2 + ρCᵥT
"""
function Pri_To_Conser!(prim::Array{FT,2}, γ::FT, cons::Array{FT,2}, FS_id::IT = size(prim)[2]+1) where {FT<:AbstractFloat, IT<:Int}
    for i = 1:FS_id - 1
        cons[:, i] .= Pri_To_Conser(prim[:, i], γ)
    end
    
end

"""
Convert conservative variables to primitive variables
ρ, ρv, E   =>   ρ, v, p    
"""
function Conser_To_Pri(cons::Array{FT,1}, γ::FT) where {FT<:AbstractFloat}
    w1, w2, w3 = cons
    return [w1; w2/w1; (w3 - w2*w2/w1/2) * (γ - 1)]
end




"""
Convert conservative variables to primitive variables
ρ, ρv, E   =>   ρ, v, p    
"""
function Conser_To_Pri!(cons::Array{FT,2}, γ::FT, prim::Array{FT,2}, FS_id::IT = size(cons)[2]+1) where {FT<:AbstractFloat, IT<:Int}
    for i = 1:FS_id - 1
        prim[:, i] .= Conser_To_Pri(cons[:, i], γ)
    end
end





"""
Compute 1D Roe flux
"""
function Roe_Flux(prim_l::Array{FT,1}, prim_r::Array{FT,1}, γ::FT) where {FT<:AbstractFloat}
    ρ_l, v_l, p_l = prim_l;
    ρ_r, v_r, p_r = prim_r;
    
    c_l = sqrt(γ * p_l/ ρ_l)
    c_r = sqrt(γ * p_r/ ρ_r)
    w_l = [ρ_l; ρ_l*v_l; ρ_l*v_l*v_l/2 + p_l/(γ - 1)]
    w_r = [ρ_r; ρ_r*v_r; ρ_r*v_r*v_r/2 + p_r/(γ - 1)]
    h_l = v_l*v_l/2.0 + γ * p_l/(ρ_l * (γ - 1));
    h_r = v_r*v_r/2.0 + γ * p_r/(ρ_r * (γ - 1));
    
    # compute the Roe-averaged quatities
    v_rl = (sqrt(ρ_l)*v_l + sqrt(ρ_r)*v_r)/ (sqrt(ρ_r) + sqrt(ρ_l));
    h_rl = (sqrt(ρ_l)*h_l + sqrt(ρ_r)*h_r)/ (sqrt(ρ_r) + sqrt(ρ_l));
    c_rl = sqrt((γ - 1)*(h_rl - v_rl * v_rl/2));
    ρ_rl = sqrt(ρ_r * ρ_l);
    
    
    du = v_r - v_l
    dp = p_r - p_l
    dρ = ρ_r - ρ_l
    du1 = du + dp/(ρ_rl * c_rl);
    du2 = dρ - dp/(c_rl * c_rl);
    du3 = du - dp/(ρ_rl * c_rl);
    
    #compute the Roe-average wave speeds
    λ_1 = v_rl + c_rl;
    λ_2 = v_rl;
    λ_3 = v_rl - c_rl;
    
    #compute the right characteristic vectors
    r_1 =  ρ_rl/(2 * c_rl) * [1 ; v_rl + c_rl;  h_rl + c_rl * v_rl]
    r_2 = [1; v_rl; v_rl * v_rl/2]
    r_3 = -ρ_rl/(2 * c_rl) * [1; v_rl - c_rl; h_rl - c_rl * v_rl]
    
    
    #compute the fluxes
    mach = v_rl/ c_rl
    if (mach <= -1)
        flux = [ρ_r*v_r; ρ_r*v_r*v_r + p_r; ρ_r*v_r*h_r]
    elseif (mach <= 0 && mach >= -1)
        flux = [ρ_r * v_r; ρ_r*v_r^2 + p_r; ρ_r*h_r*v_r] .- r_1*λ_1*du1;
    elseif (mach >= 0 && mach <= 1)
        flux = [ρ_l * v_l; ρ_l*v_l^2 + p_l; ρ_l*h_l*v_l] .+ r_3*λ_3*du3;
    else
        flux = [ρ_l * v_l; ρ_l*v_l^2 + p_l; ρ_l*h_l*v_l]
    end
    
    return flux
end

"""
1D half Riemann solver
cons_l is the fluid state variable
v is the piston or wall speed
fluid_normal is the normal of the fluid interface
if fluid_normal = 1 piston is on the right
if fluid_normal =-1 piston is on the left
"""
function FSRiemann(prim_l::Array{FT,1}, v::FT, fluid_normal::FT, γ::FT)  where {FT<:AbstractFloat}
    
    ρ_l, v_l, p_l = prim_l
    #left facing shock case
    if (v_l*fluid_normal > v*fluid_normal)
        a = 2/((γ + 1)*ρ_l);
        b = p_l*(γ - 1)/(γ + 1)
        phi = a/(v - v_l)^2
        
        p = p_l + (1 + sqrt(4*phi*(p_l + b) + 1))/(2*phi)
        ρ = ρ_l*(p/p_l + (γ - 1)/(γ + 1))/(p/p_l * (γ - 1)/(γ + 1) + 1)
        #left facing rarefactions case
    else
        c_l = sqrt(γ*p_l/ρ_l)
        p = p_l*(-(γ - 1)/(2*c_l)*(v - v_l)*fluid_normal + 1)^(2*γ/(γ - 1))
        ρ = ρ_l*(p/p_l)^(1/γ)
    end
    
    return [ρ; v; p]
end
        
        
        
        
function Limiter_Fun(r::Array{FT,1}) where {FT<:AbstractFloat}
    f = [0., 0., 0.]
    for i = 1:3
        if (r[i] > 0)
            f[i] = 2*r[i]/(r[i] + 1)
        else
            f[i] = 0
        end
    end
    return f
end



"""
Extrapolation or interpolation
"""
function Interpolation(u1, x1::FT, u2, x2::FT, x3::FT) where {FT<:AbstractFloat}
    u3 = (u2 - u1) * (x3 - x2)/(x2 - x1) + u2
    return u3
end

"""
x1               x2       x_si    x_wall
o-------|--------o---------|-------ₜ
prim1         prim2     prim_si   v_wall
"""
function FIVER(x1::FT, prim1::Array{FT, 1}, x2::FT, prim2::Array{FT, 1}, x_si::FT, x_wall::FT, v_wall::FT, γ::FT, Order::IT) where {FT<:AbstractFloat, IT<:Int}
    ##########
    # 1 order: assume the structure is at the cell interface
    ########################
    if (Order == 1)
        
        prim_wall0 = prim2
        prim_wall = FSRiemann(prim_wall0, v_wall, 1.0 , γ)
        prim_si = prim_wall
        
        ###########
        # 2 order:
        ###########################
    else
        
        prim_wall0 = Interpolation(prim1, x1, prim2, x2, x_wall)
        
        prim_wall = FSRiemann(prim_wall0, v_wall, 1.0, γ)
        
        if (x_wall > x_si)
            prim_si = Interpolation(prim2, x2, prim_wall, x_wall, x_si)
        else
            prim_si = Interpolation(prim1, x1, prim_wall, x_wall, x_si)
        end
    end
    
    return prim_si
    
end





################################################
# Fluid Class
###########################################################

mutable struct Euler1D{FT<:AbstractFloat, IT<:Int}
    L::FT
    N::IT
    Δx::FT
    xx::Array{FT,1}
    
    W::Array{FT,2}       # (3, nnodes) array, conservative states at n-1 step  
    V::Array{FT,2}
    
    
    emb::Array{FT,1}
    emb_nm1::Array{FT,1}

    FS_id::IT
    FS_id_nm1::IT

    FIVER_order::IT
    
    γ::FT              # gas constant

    t::FT
end


    
function Euler1D(N::IT, L::FT, emb::Array{FT,1}; FIVER_order::IT = 2, γ::FT=1.4) where {FT<:AbstractFloat, IT<:Int}
    Δx = L/N
    
    
    xx = Array(LinRange(Δx/2, L - Δx/2, N))
    W = zeros(FT, 3, N)
    V = zeros(FT, 3, N)
    xs, vs = emb

    t = FT(0)

    FS_id = FS_id_nm1 = FS_id_cal(xs, Δx)


    return Euler1D(L, N, Δx, xx, W, V, emb, emb, FS_id, FS_id_nm1, FIVER_order, γ, t)

end
    

function Initial!(self::Euler1D{FT, IT}, init_func::Function) where {FT<:AbstractFloat, IT<:Int}
    
    γ = self.γ

    for i = 1:self.FS_id-1
        active_or_not, ρ, v, p = init_func(self.xx[i])
        
        self.V[:,i]  .= ρ, v, p

        cons = Pri_To_Conser([ρ; v; p], γ)
        self.W[:,i] .= cons
    end
end


"""
Compute time step
Δt > 0 constant time step 
otherwise compute maximum wave speed using Primitive states V and use CFL based rule
"""
function Compute_Time_Step(self::Euler1D{FT, IT}, cfl::FT) where {FT<:AbstractFloat, IT<:Int}
    V = self.V
    γ = self.γ
    max_wave_speed = -1.0
    
    for i = 1:self.FS_id_nm1-1
        
            
        wave_speed = abs(V[2,i]) + sqrt(γ* V[3,i]/V[1,i])
        max_wave_speed = max(wave_speed, max_wave_speed)
            
    end
    
    return cfl*self.Δx/max_wave_speed 
end



function Send_Force(self::Euler1D{FT, IT}, t::FT, Δt::FT) where {FT<:AbstractFloat, IT<:Int}
    
    xs, vs = self.emb
    FS_id = self.FS_id
    V, xx = self.V, self.xx
    γ = self.γ

    prim_s = FIVER(xx[FS_id - 2], V[:, FS_id - 2], xx[FS_id - 1], V[:,FS_id - 1], xs, xs, vs, γ, self.FIVER_order)
    
    fext = prim_s[3]

    return fext
end

"""
FS_id is the number of the first inactive node
update the states at FS_id_nm1
"""
function Intersector_Update!(self::Euler1D{FT, IT}, V::Array{FT,2}, W::Array{FT,2})   where {FT<:AbstractFloat, IT<:Int}
    
    xs, vs = self.emb
    FS_id_nm1 = self.FS_id_nm1

    γ = self.γ
    #Using interpolation between inside node and Riemann solution
    xx = self.xx
    x_si = 0.5*(xx[FS_id_nm1] + xx[FS_id_nm1 + 1])
    
    # populate the ghost node at past time
    vv_s = FIVER(xx[FS_id_nm1 - 2], V[:, FS_id_nm1 - 2], xx[FS_id_nm1 - 1], V[:,FS_id_nm1 - 1], x_si, xs, vs, self.γ, self.FIVER_order)
    V[:, FS_id_nm1] .= Interpolation(V[:,FS_id_nm1 - 1],xx[FS_id_nm1 - 1], vv_s, x_si, xx[FS_id_nm1])
    W[:, FS_id_nm1] .= Pri_To_Conser(V[:, FS_id_nm1], γ)
    
end

function Compute_Rhs_Muscl(self::Euler1D{FT, IT}, V::Array{FT,2} ,  RKstep::IT, eps::FT = 1e-8) where {FT<:AbstractFloat, IT<:Int}
    
    N = self.N
    Δx = self.Δx   #assume all the cells have the same size
    xx = self.xx
    γ = self.γ


    rhs  = zeros(Float64, 3, N)
    
    if ( RKstep == 1 )
        FS_id = self.FS_id_nm1
        xs, vs = self.emb_nm1

    else
        FS_id = self.FS_id
        xs, vs = self.emb
    end
    

    # Ghost Fluid Method with Roe Flux to implement wall boundary condition
    prim_l =  [V[1,1],-V[2,1],V[3,1]]
    rhs[:, 1] += Roe_Flux(prim_l, V[:,1], γ)
    
    
    #FS_id is the number of the edge which intersects the structure surface
    for i  = 1: FS_id - 1
        #################################################
        # Left State
        #######################################################
        
        if (i == 1)
            xx_ll = xx[i] - Δx
            xx_l  = xx[i]
            prim_l  = V[:, i]
            
            #Ghost Fluid Method exterpolation
            prim_ll = [prim_l[1]; -prim_l[2]; prim_l[3]]
        else
            
            xx_ll = xx[i - 1]
            xx_l  = xx[i]
            
            prim_ll = V[:,i - 1]
            prim_l  = V[:,i]
        end
        
        ########################################
        # Right State
        ##########################################
        
        @assert( i <= N)
        
        prim_r  = V[:,i + 1]
        prim_rr = V[:,i + 2]
        
        xx_r  = xx[i + 1]
        xx_rr = xx[i + 2]
        
        ######################
        # Left State
        #######################################
        
        if ( i < FS_id - 2 )
            
            L = (prim_r - prim_l)./(prim_l - prim_ll .+ eps);
            phi = Limiter_Fun(L)
            prim_l_recon = prim_l + 0.5* phi .*(prim_l - prim_ll)
            
            R = (prim_r - prim_l)./(prim_rr - prim_r .+ eps);
            phi = Limiter_Fun(R)
            prim_r_recon = prim_r - 0.5* phi .*(prim_rr - prim_r)
            
        elseif (i == FS_id - 2 )
            
            L = (prim_r - prim_l)./(prim_l - prim_ll .+ eps);
            phi = Limiter_Fun(L)
            prim_l_recon = prim_l + 0.5* phi .*(prim_l - prim_ll)
            
            x_si = 0.5*(xx_r + xx_rr)
            prim_s = FIVER(xx_l, prim_l, xx_r, prim_r, x_si, xs, vs, self.γ, self.FIVER_order)
            prim_rr_g = Interpolation(prim_r, xx_r, prim_s, x_si, xx_rr)
            
            R = (prim_r - prim_l)./(prim_rr_g - prim_r .+ eps);
            phi = Limiter_Fun(R)
            prim_r_recon = prim_r - 0.5* phi .*(prim_rr_g - prim_r)
            
        else
            
            x_si = 0.5*(xx_l + xx_r)
            
            prim_l_recon = Interpolation(prim_ll, xx_ll, prim_l, xx_l, x_si)
            prim_r_recon = FIVER(xx_ll, prim_ll, xx_l, prim_l, x_si, xs, vs, self.γ, self.FIVER_order)
            
            # todo 
            prim_l_recon = prim_r_recon
        end
        
        flux = Roe_Flux(prim_l_recon, prim_r_recon, γ)   
        rhs[:, i] -= flux


        # Do not update ghost node
        if i < FS_id - 1
            rhs[:, i + 1] += flux   
        end

        
    end
    return rhs
end



########
# return the fluid structure surrogate interface label
# the first number is the surrogate boundary label of xs_{n-1}
# the second number is the surrogate boundary label of xs_{n}
###############
function FS_id_cal(xs::FT, Δx::FT, tol::FT = 1e-8) where {FT<:AbstractFloat}
    
    FS_id = Int64(ceil((xs - tol)/Δx - 0.5))
    
    return FS_id 
end

function Fluid_Time_Advance!(self::Euler1D{FT, IT}, emb::Array{FT,1}, t::FT, Δt::FT) where {FT<:AbstractFloat, IT<:Int}
    γ = self.γ
    # update embedded surfaces
    self.emb_nm1 = self.emb
    self.emb = emb
    
    FS_id_nm1 = self.FS_id_nm1 = self.FS_id  

    xs, Δx = emb[1], self.Δx
    FS_id = self.FS_id = FS_id_cal(xs, Δx)
    
    @assert(FS_id <= FS_id_nm1 + 1)
    
    W, V = self.W, self.V
    
    #first step k1 = F(w_{n-1}, xs_{n-1}, vs_{n-1})
    rhs = Compute_Rhs_Muscl(self, V,  1)
    
    W_tmp = W + Δt*rhs/Δx
    
    V_tmp = copy(W_tmp)
    Conser_To_Pri!(W_tmp, γ, V_tmp, FS_id_nm1)
    
    if FS_id == FS_id_nm1 + 1
        # node 0, node 1, ... node FS_id - 1 are real cells
        # need GTR or RTG in the procedure
        Intersector_Update!(self, V_tmp, W_tmp)
    end
    
    
    #second step k2 = F(w_{n-1} - dt*k1, xs_n, vs_n)
    rhs = Compute_Rhs_Muscl(self, V_tmp, 2)

    W_tmp .= W_tmp + Δt*rhs/Δx
    
    W .= 0.5*(W_tmp + W)
    
    Conser_To_Pri!(W, γ, V, FS_id)
    
    if (FS_id == FS_id_nm1 + 1)
        # Ghost to Real Change
        Intersector_Update!(self, V, W)
    end
    
    self.t = self.t + Δt
 
end
