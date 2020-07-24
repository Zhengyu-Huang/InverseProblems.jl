using Revise
using PyPlot
using NNFEM

options = Options(1)

mutable struct Params
    # ns element each block
    ns::Int64
    # ns_obs observations on each face of the block
    ns_obs::Int64
    ls::Float64
    porder::Int64
    ngp::Int64

    matlaw::String
    ρ::Float64
    E::Float64
    ν::Float64

    P1::Float64
    P2::Float64
    T::Float64
    NT::Int64
end



"""
Force load on one edge of the plate, 
The edge is [0,L], which is discretized to ne elements, with porder-th order polynomials
type: a string of load type, which can be "constant", or "Gaussian"
args: an array, 
      for Constant load, it has p1 and p2, in tangential and normal direction
      for "Gaussian" load, it has p, x0, and σ, the force is in the normal direction
                  
"""
function ComputeLoad(L, ne, porder, ngp, type,  args)
    @assert(ngp <= 4)
    dx = L/ne

    xx = Array(LinRange(0, L, ne*porder + 1))
    
    Ft, Fn = zeros(Float64, ne*porder + 1), zeros(Float64, ne*porder + 1)

    elem_coords = zeros(Float64, porder + 1, 2)

    # construct pressure load function
    if type == "Constant"
        pt, pn = args
        f_px, f_py = x->pt, x->pn
    elseif type == "Gaussian"
        pn, x0, σ = args
        f_px, f_py = x->0, x-> pn * 1.0/(sqrt(2*pi*σ^2)) * exp.(-0.5*(x .- x0).^2/σ^2)
    else
        error("Force load type is not recogonized ", type)
    end


    # assemble the force

    @show ne
    for e = 1:ne
        if porder == 1
            loc_id = [e, e+1]
        elseif porder == 2
            loc_id = [2*(e-1)+1, 2*(e-1)+3, 2*(e-1)+2]
        else
            error("porder error porder ==", porder)
        end

        loc_xx = xx[loc_id]         #x-coordinates of element nodes 
        elem_coords[:,1] .= loc_xx
        

        #return list 
        weights, hs = get1DElemShapeData( elem_coords, ngp)  
        
        for igp = 1:ngp
            gp_xx = loc_xx' * hs[igp]   #x-coordinates of element Gaussian points
            Ft[loc_id] += f_px(gp_xx) * hs[igp] * weights[igp]
            Fn[loc_id] += f_py(gp_xx) * hs[igp] * weights[igp]

        end
    end

    return Ft, Fn
end






"""
|-------|-------|-------|-------| 
|       |       |       |       |
|       |       |       |       |
|-------|-------|-------|-------| 
|2ns+1  |               |       |
|1 .. ns|               |ns+1 ..|
|-------|               |-------|
(0,0)     (ls,0)

Both nodes and elements are ordered from left to right and bottom to top
"""
function Construct_Mesh(phys_params::Params)
    ns, ls, porder, ngp = phys_params.ns, phys_params.ls, phys_params.porder, phys_params.ngp 
    matlaw, ρ, E, ν = phys_params.matlaw, phys_params.ρ, phys_params.E, phys_params.ν
    P1, P2, T = phys_params.P1, phys_params.P2, phys_params.T

    #########################
    # Material property
    #########################

    prop = Dict("name"=> matlaw, "rho"=>ρ, "E"=>E, "nu"=>ν)

    
    ################################
    # Construct nodes
    ################################
    nnodes = 6*(ns*porder + 1)*(ns*porder + 1) - 5*(ns*porder + 1)
    nodes = zeros(Float64, nnodes, 2)

    Δl = ls/(ns*porder)
    in = 1
    for j = 1:2*ns*porder + 1
        for i = 1:4*ns*porder + 1
    
            x, y = (i-1)*Δl, (j-1)*Δl
            
            if j < ns*porder + 1 && ((i >  ns*porder + 1) && (i < 3*ns*porder + 1))
                continue
            end
            
            nodes[in, :] .= x, y

            in += 1
            
        end
    end
    @assert(in == nnodes+1)
    #################################
    # Construct elements 
    #################################
    

    neles = 6*ns*ns
    elements = []
    for j = 1:2*ns
        for i = 1:4*ns 
        
            if i <=  ns && j <= ns
                nx = 2*ns*porder+2
                n = nx*(j-1)*porder + (i-1)*porder + 1
                
            elseif i > 3*ns && j <= ns
                nx = 2*ns*porder+2
                n = nx*(j-1)*porder + (i-3*ns-1)*porder + ns*porder+2
                if j == ns
                    nx = 4*ns*porder+1
                end
        
            elseif j > ns
                nx = 4*ns*porder+1
                n = nx*(j-ns-1)*porder + (i-1)*porder+1 + (2*ns*porder+2)*(ns*porder)
 
            else
                continue
            end
            
            #element (i,j)
            if porder == 1
                #   4 ---- 3
                #
                #   1 ---- 2
                
                elnodes = [n, n + 1, n + 1 + nx, n + nx]
            elseif porder == 2
                #   4 --7-- 3
                #   8   9   6 
                #   1 --5-- 2
                if i > 3*ns && j == ns

                    nx = [2*ns*porder+2,   4*ns*porder+1]
                    elnodes = [n, n + 2, n + 2 + nx[1]+nx[2],  n + nx[1]+nx[2], n+1, n + 2 + nx[1], n + 1 + nx[1]+nx[2], n + nx[1], n+1+nx[1]]

                    
                else
                    elnodes = [n, n + 2, n + 2 + 2*nx,  n + 2*nx, n+1, n + 2 + nx, n + 1 + 2*nx, n + nx, n+1+nx]
                end
            else
                error("polynomial order error, porder= ", porder)
            end
            
            coords = nodes[elnodes,:]
            push!(elements,SmallStrainContinuum(coords,elnodes, prop, ngp))
        end
    end


    
    ################################################################
    # Construct boundary conditions
    ###############################################################
    
    ndofs = 2
    
    EBC, g = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)
    FBC, fext = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)
    ft = nothing
    gt = nothing

    EBC[collect(1:2*ns*porder+2), :] .= -1 # fix bottom
    FBC[collect(nnodes - 4*ns*porder:nnodes), :] .= -2 # force on the top

    F1, F2 = ComputeLoad(4*ls, 4*ns, porder, ngp, "Constant",  [P1, P2])
    fext[collect(nnodes - 4*ns*porder:nnodes), 1] .= F1
    fext[collect(nnodes - 4*ns*porder:nnodes), 2] .= F2


    dof_to_active = findall(FBC[:].==-2)
    if length(dof_to_active) != 0
        ft = t->fext[:][dof_to_active]*sin(2*pi*t/(T))
    end
    
    
    
    
    
    return nodes, elements, EBC, g, gt, FBC, fext, ft

end

"""
Displacement field on a circle from node 1
"""

function Get_Obs(phys_params::Params)
    ns, ns_obs = phys_params.ns, phys_params.ns_obs
    porder = phys_params.porder

    # choose k nodes on each face, including both ends
    obs_nid = zeros(Int64, 12*ns_obs - 14)

    Δobs = Int64(ns*porder/(ns_obs-1))
    nnodes = 6*(ns*porder + 1)*(ns*porder + 1) - 5*(ns*porder + 1)
    
    obs_nid[1:ns_obs-1] = 1+Δobs*(2*ns*porder+2):  (2*ns*porder+2)*Δobs  : 1 + (2*ns*porder+2)*(ns*porder)
    obs_nid[ns_obs-1: 2*ns_obs-2] =    1 + (2*ns*porder+2)*(ns*porder):  (4*ns*porder+1)*Δobs : nnodes - 4*ns*porder 
    obs_nid[2*ns_obs-2:6*ns_obs-6] =   nnodes - 4*ns*porder :  Δobs  : nnodes  
    obs_nid[6*ns_obs-6:7*ns_obs-7] =    nnodes : -(4*ns*porder+1)*Δobs : 1 + (2*ns*porder+2)*(ns*porder) + (4*ns*porder)
    obs_nid[7*ns_obs-6:8*ns_obs-9] = (2*ns*porder+2)*(Δobs+1) + (2*ns*porder+2)*Δobs*(ns_obs-3)    : -(2*ns*porder+2)*Δobs : (2*ns*porder+2)*(Δobs+1)
    

    obs_nid[8*ns_obs-8:9*ns_obs-11] = 1+Δobs*(2*ns*porder+2)+(ns*porder+1) : (2*ns*porder+2)*Δobs : 1+Δobs*(2*ns*porder+2)+(ns*porder+1) + (2*ns*porder+2)*Δobs*(ns_obs-3)
    
    obs_nid[9*ns_obs-10:11*ns_obs-12] = 1 + (2*ns*porder+2)*(ns*porder) + 3*ns*porder : -Δobs: 1 + (2*ns*porder+2)*(ns*porder) + ns*porder 
    
    obs_nid[11*ns_obs-11:12*ns_obs-14] = 1+Δobs*(2*ns*porder+2)+(ns*porder) + (2*ns*porder+2)*Δobs*(ns_obs-3) : -(2*ns*porder+2)*Δobs: 1+Δobs*(2*ns*porder+2)+(ns*porder)

    return obs_nid

end


function Visual_Block(block::Array{Int64, 2}, state::Array{Float64, 2}, Qoi::Array{Float64, 1}, vmin=nothing, vmax=nothing)
    nbx, nby = size(block)
    X = zeros(Float64, nbx, nby)
    Y = zeros(Float64, nbx, nby)
    C = zeros(Float64, nbx, nby)

    for i = 1:nbx
        for j = 1:nby
            n_id = block[i,j]
            X[i,j] = state[n_id,1] 
            Y[i,j] = state[n_id,2] 
            C[i,j] = Qoi[n_id]
        end
    end

    pcolormesh(X, Y, C, shading ="gouraud", cmap="jet", vmin=vmin, vmax=vmax)
    
end
"""
Block structure mesh visualization
dx, dy: displacement
obs_nid: mark observation points
"""
function Visual(phys_params::Params,
    state::Array{Float64, 2}, Qoi::Array{Float64, 1}, save_file_name="None",  obs_nid=nothing, vmin=nothing, vmax=nothing)

    ns, porder = phys_params.ns, phys_params.porder
    
    
    block = zeros(Int64, ns*porder+1, ns*porder+1)
    for i = 1:ns*porder+1
        start = 1+(i-1)*(2*ns*porder+2)
        block[i, :] .= start: start + ns*porder
    end
    Visual_Block(block, state, Qoi, vmin, vmax)
    

    block = zeros(Int64, ns*porder+1, ns*porder+1)
    for i = 1:ns*porder
        start = ns*porder+2+(i-1)*(2*ns*porder+2)
        block[i, :] .= start:start + ns*porder
    end
    start = 1 + (2*ns*porder+2)*(ns*porder) + 3*ns*porder
    block[ns*porder+1, :] .= start: start + ns*porder
    Visual_Block(block, state, Qoi, vmin, vmax)



    block = zeros(Int64, ns*porder+1, 4*ns*porder+1)
    for i = 1:ns*porder+1
        start = (2*ns*porder+2)*ns*porder +1 + (i-1)*(4*ns*porder+1)
        block[i, :] .= start : start + 4*ns*porder
    end

    Visual_Block(block, state, Qoi, vmin, vmax)
    
    colorbar()
    axis("equal")


    if obs_nid != nothing
        x_obs, y_obs = state[obs_nid, 1], state[obs_nid, 2]
        scatter(x_obs, y_obs, color="black")
    end

    if save_file_name != "None"
        savefig(save_file_name)
        close("all")
    end


end

function Params()
    ns = 10
    ns_obs = 3
    ls = 10.0
    porder = 1
    ngp = 3

    """
    Concrete 
    th = 1m
    ρ  = 2400kg/m^3 = 1kg1/m^3   (2400kg= 1kg1)
    E  = 60×10^9 Pa = 60×10^9 kg/(m s^2) = 2400 kg/(m s1^2) = 1 kg1/(m s1^2)  (s1=2e-4s)
    ν  = 0.2
    ls = 10m     
    """
    matlaw = "PlaneStress"
    ρ = 1.0
    E = 1.0
    ν = 0.2

    

    P1 = 0.0
    P2 = 0.1
    T = 100.0
    NT = 100

    @assert((ns*porder) % (ns_obs-1) == 0)

    Params(ns, ns_obs, ls, porder, ngp, matlaw, ρ, E, ν, P1, P2, T, NT)
end


phys_params = Params()
nodes, elements, EBC, g, gt, FBC, fext, ft = Construct_Mesh(phys_params)

ndofs = 2
domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)

#initial displacement, displacement at previous time step
u = zeros(domain.neqs)
u_p = zeros(domain.neqs)
∂u = zeros(domain.neqs)
∂∂u = zeros(domain.neqs)
globdat = GlobalData(u, u_p, ∂u, ∂∂u, domain.neqs, gt, ft)

assembleMassMatrix!(globdat, domain)
updateStates!(domain, globdat)


obs_nid = Get_Obs(phys_params)
T, NT = phys_params.T, phys_params.NT

Δt = T/NT


#Midpoint rule solver
ρ_oo = 1.0
αm = (2*ρ_oo - 1)/(ρ_oo + 1)
αf = ρ_oo/(ρ_oo + 1)

ts = LinRange(0, T, NT+1)
adaptive_solver_args = Dict("Newmark_rho"=> 0.0, 
                            "Newton_maxiter"=>10, 
                            "Newton_Abs_Err"=>1e-4, 
                            "Newton_Rel_Err"=>1e-6, 
                            "damped_Newton_eta" => 1.0)

globdat, domain, ts = AdaptiveSolver("NewmarkSolver", globdat, domain, T, NT, adaptive_solver_args)



# # error()
if !isdir("$(@__DIR__)/Data")
    mkdir("$(@__DIR__)/Data")
end

if !isdir("$(@__DIR__)/Figs")
    mkdir("$(@__DIR__)/Figs")
end

hist_state = domain.history["state"]
for i = 1:10:NT+1
    disp = reshape(hist_state[i], size(nodes,1), 2)
    state = disp + nodes

    disp_mag = sqrt.(disp[:,1].^2 + disp[:,2].^2)
    vmin, vmax = minimum(disp_mag), maximum(disp_mag)
    Visual(phys_params, state, disp_mag, "disp."*string(i)*".png", nothing, vmin, vmax)
end

# close("all")
# frame = NT
# # length scale(cm), stress scale(GPa), time scale(s)
# scales = [100.0, 1.0/0.000076*1.0e-3, 1.0e-3]
# visσ(domain, nx, ny,  domain.history["stress"][frame], domain.history["state"][frame]; scaling = scales)
# xlabel("X (cm)")
# ylabel("Y (cm)")
# PyPlot.tight_layout()
# savefig("Debug/order$porder/terminal$(tid)_$(force_scale)_$(fiber_size).png")


# close("all")
# Disp = zeros(Float64, NT+1, 5)
# Disp[:,1] = LinRange(0, T, NT+1)
# ux = [reshape(domain.history["state"][i][1:(nx*porder+1)*(ny*porder+1)], nx*porder+1, ny*porder+1)[end,end] for i = 1:length(domain.history["state"])]
# uy = [reshape(domain.history["state"][i][(nx*porder+1)*(ny*porder+1)+1:end], nx*porder+1, ny*porder+1)[end,end] for i = 1:length(domain.history["state"])]
# Disp[:,2], Disp[:,3] = ux, uy
# ux = [reshape(domain.history["state"][i][1:(nx*porder+1)*(ny*porder+1)], nx*porder+1, ny*porder+1)[div((nx*porder+1),2),end] for i = 1:length(domain.history["state"])]
# uy = [reshape(domain.history["state"][i][(nx*porder+1)*(ny*porder+1)+1:end], nx*porder+1, ny*porder+1)[div((nx*porder+1),2),end] for i = 1:length(domain.history["state"])]
# Disp[:,4], Disp[:,5]  = ux, uy
# plot(Disp[:, 1]*scales[3], Disp[:, 2]*scales[1], "--r",  label="Ref")
# plot(Disp[:, 1]*scales[3], Disp[:, 3]*scales[1], "--y")
# plot(Disp[:, 1]*scales[3], Disp[:, 4]*scales[1], "--b")
# plot(Disp[:, 1]*scales[3], Disp[:, 5]*scales[1], "--g")

# savefig("Debug/order$porder/disp$(tid)_$(force_scale)_$(fiber_size).png")