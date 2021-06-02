using Revise
using PyPlot
using LinearAlgebra
using NNFEM

options = Options(1)

"""
Force load on one edge of the plate, 
The edge is [0,L], which is discretized to ne elements, with porder-th order polynomials
type a string of load type, which can be "constant", or "Gaussian"
args an array, 
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



mutable struct Params
    
    # ns element each block
    ns::Int64
    ls::Float64
    porder::Int64
    ngp::Int64
    domain::Domain
    
    # coarse scale
    ns_c::Int64
    porder_c::Int64
    interp_e::Array{Int64, 2}  # node id -> coarse scale element id, local node id
    interp_sdata::Array{Float64, 2}
    ind_θf_to_θc::Array{Int64, 1} 
    domain_c::Domain
    
    # ns_obs observations on each face of the block
    ns_obs::Int64
    n_data::Int64
    
    prop
    # matlaw::String
    # ρ::Float64
    # E::Float64
    # ν::Float64

    
    # force load and time
    P1::Float64 #in x direction
    P2::Float64 #in y direction

    problem::String # Static or Dynamic
    T::Float64
    NT::Int64
    ΔNT::Int64

    N_y::Int64
    θ_names::Array{String, 1}
    
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
function Construct_Mesh(ns::Int64, porder::Int64, ls::Float64, ngp::Int64, prop::Dict, P1::Float64, P2::Float64, problem::String, T::Float64=0.0)
    
    
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
    
    
    F1, F2 = ComputeLoad(4*ls, 4*ns, porder, ngp, "Constant",  [P1, P2])
    fext[collect(nnodes - 4*ns*porder:nnodes), 1] .= F1
    fext[collect(nnodes - 4*ns*porder:nnodes), 2] .= -F2
    
    if problem == "Dynamic"
        FBC[collect(nnodes - 4*ns*porder:nnodes), :] .= -2 # force on the top
        dof_to_active = findall(FBC[:].==-2)
        if length(dof_to_active) != 0
            ft = t->fext[:][dof_to_active]*sin(2*pi*t/(T))
        end
    elseif problem == "Static"
        FBC[collect(nnodes - 4*ns*porder:nnodes), :] .= -1 # force on the top
    else
        error("Problem type ", problem, " has not implemented yet")
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
function Visual_Node(phys_params::Params,
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





"""
3 damage spots
Output should be 0-1, close to 1 means damage
"""  
function Damage_Ref(x::Float64, y::Float64)
    μ = [50.0 50.0; 250.0 160.0; 380.0 100.0]
    
    Σ = zeros(Float64, 3, 2, 2)
    # Σ[1,:,:] = Array([200.0 0.0; 0.0 200.0])
    Σ[1,:,:] = Array([400.0 0.0; 0.0 400.0])
    Σ[2,:,:] = Array([800.0 0.0; 0.0 400.0])
    Σ[3,:,:] = Array([100.0 0.0; 0.0 400.0])
    
    A = [0.8; 0.6; 0.5]

    A = [0.0; 0.9; 0.9]
    
    er = 0.0
    for i = 1:3
        disp = [x; y] - μ[i,:]
        er += A[i]*exp(-0.5*(disp'*(Σ[i,:,:]\disp)))
    end
    # todo check 
    return min(er, 0.9)
end

function Initialize_E!(domain::Domain, prop_ref::Dict{String, Any})
    prop = copy(prop_ref)
    E, ν = prop["E"], prop["nu"]
    elements = domain.elements
    ne = length(elements)

    nodes = domain.nodes
    nnodes = domain.nnodes
    θ_dam = zeros(Float64, nnodes)
    
    for ie = 1:ne

        # update each Gaussian points
        elem = elements[ie]
        mat = elem.mat
        gnodes = getGaussPoints(elem)
        ng = size(gnodes, 1)
        for ig = 1:ng
            x, y = gnodes[ig, :]
            prop["E"] = E * (1.0 - Damage_Ref(x, y))
            mat[ig] = PlaneStress(prop)
        end


        # update θ_dam at each node 
        elnodes = elem.elnodes
        for el in elnodes
            θ_dam[el] = Damage_Ref(nodes[el, 1], nodes[el, 2])
        end
        
    end

    return θ_dam
end


"""
nodal based update length(θ) = nnodes
"""
function Update_E!(domain::Domain, prop_ref::Dict{String, Any}, θ_dam::Array{Float64, 1})
    prop = copy(prop_ref)
    E, ν = prop["E"], prop["nu"]
    elements = domain.elements
    ne = length(elements)
    for ie = 1:ne
        elem = elements[ie]
        mat = elem.mat
        gnodes = getGaussPoints(elem)
        ng = size(gnodes, 1)
        elnodes = elem.elnodes
        hs = elem.hs

        for ig = 1:ng
            θ_intep = θ_dam[elnodes]' * hs[ig]
            prop["E"] = E * (1.0 - θ_intep)
            mat[ig] = PlaneStress(prop)
        end
        
    end

end





function Get_θ_Dam(θ::Float64)
    a = 1.0 
    c = 10.0
    return a*(1-exp(-θ))/(1+c*exp(-θ))
end

function Get_θ(θ_dam::Float64)
    a = 1.0 
    c = 10.0
    return -log((a-θ_dam)/(θ_dam*c + a))
end

# function Get_θ_Dam(θ::Float64)
#     a = 0.9 
#     c = 9.0
#     return θ
# end

# function Get_θ(θ_dam::Float64)
#     a = 0.9 
#     c = 9.0
#     return θ_dam
# end

function Get_θ_Dam(θ::Array{Float64,1})
    θ_trans = similar(θ)
    for i = 1:length(θ)
        θ_trans[i] = Get_θ_Dam(θ[i])
    end
    return θ_trans
end 


function Get_θ(θ_dam::Array{Float64,1})
    θ_trans = similar(θ_dam)
    for i = 1:length(θ_dam)
        θ_trans[i] = Get_θ(θ_dam[i])
    end
    return θ_trans
end 


function Run_Damage(phys_params::Params, θ_type::String, θ = nothing, P1 = phys_params.P1, P2 = phys_params.P2, save_disp_name::String = "None", 
    save_E::String = "None", noise_level::Float64 = -1.0, )
    
    ns, porder, ls, ngp, prop, problem, T = phys_params.ns, phys_params.porder, phys_params.ls, phys_params.ngp, 
                                                    phys_params.prop, phys_params.problem, phys_params.T
    nodes, elements, EBC, g, gt, FBC, fext, ft = Construct_Mesh(ns, porder, ls, ngp, prop, P1, P2, problem, T)
    
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
    
    if θ_type == "Analytic"

        θ_dam = Initialize_E!(domain, prop)

    elseif θ_type == "Piecewise"

        θ_dam = Get_θ_Dam_From_Raw(phys_params.domain_c, phys_params.interp_e, phys_params.interp_sdata, θ)

        # θ_dam = Get_θ_Dam(θ_c)


        Update_E!(domain, prop, θ_dam)
    else
        error("θ_type ", θ_type, " has not implemented yet")
    end

    
    obs_nid = Get_Obs(phys_params)
    
    
    if phys_params.problem == "Dynamic"
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
        
        # get state (displacement) variable 
        hist_state = domain.history["state"]
        
        ΔNT = phys_params.ΔNT
        data = zeros(Float64, length(obs_nid), 2*Int64(NT/ΔNT))
        
        for i = 1:Int64(NT/ΔNT)
            
            disp = reshape(hist_state[i*ΔNT+1], size(nodes,1), 2)
            data[:, 2*i-1:2*i] = disp[obs_nid, :]
            
            if save_disp_name != "None"
                state = disp + nodes
                disp_mag = sqrt.(disp[:,1].^2 + disp[:,2].^2)
                vmin, vmax = minimum(disp_mag), maximum(disp_mag)
                Visual_Node(phys_params, state, disp_mag, save_disp_name*string(i*ΔNT)*".pdf", obs_nid, vmin, vmax)
            end
            
        end
        
    elseif phys_params.problem == "Static"
        StaticSolver(globdat, domain, 1)
        
        # get state (displacement) variable 
        hist_state = domain.history["state"]
        
        data = zeros(Float64, length(obs_nid), 2)
        
        
        disp = reshape(hist_state[end], size(nodes,1), 2)
        data .= disp[obs_nid, :]

        # @show norm(disp), norm(data)
        
        if save_disp_name != "None"
            state = disp + nodes
            disp_mag = sqrt.(disp[:,1].^2 + disp[:,2].^2)
            vmin, vmax = minimum(disp_mag), maximum(disp_mag)
            Visual_Node(phys_params, state, disp_mag, save_disp_name*".pdf", obs_nid, vmin, vmax)
        end
        
    else
        error("phys_params.problem ", phys_params.problem, " has not implemented yet")
    end

    

    if save_E != "None"
        E = phys_params.prop["E"]
        Visual_Node(phys_params, nodes, E*(1.0 .- θ_dam), save_E*".pdf",  nothing, 0, E)
    end


    data = data[:]
    if noise_level > 0.0
        noise = similar(data)
        Random.seed!(123);
        for i = 1:length(data)
            noise[i] = rand(Normal(0, noise_level*abs(data[i])))
            #noise[i] = rand(Uniform(-noise_level*abs(data[i]), noise_level*abs(data[i])))
            
        end
        data .+= noise

        @info data - noise
        @info noise

    end
    return θ_dam, data
  
end


function Params(ns::Int64, ns_obs::Int64, porder::Int64, problem::String, ns_c::Int64, porder_c::Int64, n_test::Int64)
    
    # ns = 4
    # ns_obs = 3
    
    # porder = 1
    # problem = "Static"



    ls = 100.0
    ngp = 3
    

    #number of parameter elements
    # ns_c = 2
    # porder_c = 2
    
    """
    Concrete 
    th = 1m
    ρ  = 2400kg/m^3 = 1kg1/m^3   (2400kg= 1kg1)
    E  = 60×10^9 Pa = 60×10^9 kg/(m s^2) = 2400 kg/(m s1^2) = 1 kg1/(m s1^2)  (s1=2e-4s)
    ν  = 0.2
    ls = 10m     
    """
    prop = Dict("name"=> "PlaneStress", "rho"=>1.0, "E"=>1000.0, "nu"=>0.2)

    
    P1 = 2.0
    P2 = 20.0


    T, NT, ΔNT= 0.0, 0, 0
    n_data = (12*ns_obs - 14) * 2 * n_test
    if problem == "Dynamic"
        T, NT, ΔNT= 100.0, 100, 10
        n_data = (12*ns_obs - 14) * 2*Int64(NT/ΔNT)
    end


    nodes, elements, EBC, g, gt, FBC, fext, ft = Construct_Mesh(ns, porder, ls, ngp, prop, P1, P2, "Static")
    ndofs = 2
    domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)

    nodes, elements, EBC, g, gt, FBC, fext, ft = Construct_Mesh(ns_c, porder_c, ls, ngp, prop, P1, P2, "Static")
    ndofs = 2
    domain_c = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
    
    

    interp_e, interp_sdata, ind_θf_to_θc = Params_Interp(ns, porder, ns_c, porder_c)

    Params(ns, ls, porder, ngp, domain,
           ns_c, porder_c, interp_e, interp_sdata, ind_θf_to_θc, domain_c,
           ns_obs, n_data,
           prop,
           P1, P2, 
           problem, 
           T, NT, ΔNT, n_data, ["damage_ω"])
end

function Params_Interp(ns, porder, ns_c, porder_c)

    nnodes = 6*(ns*porder + 1)*(ns*porder + 1) - 5*(ns*porder + 1)
    sub_ele = Int64((ns*porder)/(ns_c*porder_c))
    # coarse element has ele_c segments or ele_c + 1 nodes in each direction
    ele_c = sub_ele*porder_c
    # mesh node id -> coarse element id, local node id
    interp_e = zeros(Int64, nnodes, 2)
    
    # indices (the the first node) - 1 for each row
    # and at each bridge pier there are 2 columns
    row_incr = zeros(Int64, 2*ns*porder + 1, 2)
    for i = 1:ns*porder + 1
        row_incr[i,1] = (i-1)*2*(ns*porder + 1)
        row_incr[i,2] = (i-1)*2*(ns*porder + 1) + (ns*porder + 1)
    end
    i = ns*porder + 1
    row_incr[i,2] = (i-1)*2*(ns*porder + 1) + 3*ns*porder
    for i = ns*porder + 2: 2*ns*porder + 1
        row_incr[i,1] = row_incr[i-1] + (4*ns*porder + 1)
    end


    # indices (the the first node) - 1 for each row
    # and at each bridge pier there are 2 columns
    row_incr_c = zeros(Int64, 2*ns_c*porder_c + 1, 2)
    for i = 1:ns_c*porder_c + 1
        row_incr_c[i,1] = (i-1)*2*(ns_c*porder_c + 1)
        row_incr_c[i,2] = (i-1)*2*(ns_c*porder_c + 1) + (ns_c*porder_c + 1)
    end
    i = ns_c*porder_c + 1
    row_incr_c[i,2] = (i-1)*2*(ns_c*porder_c + 1) + 3*ns_c*porder_c
    for i = ns_c*porder_c + 2: 2*ns_c*porder_c + 1
        row_incr_c[i,1] = row_incr_c[i-1] + (4*ns_c*porder_c + 1)
    end


    """
    nodes from left to right bottom to top
    """
    neles = 6*ns_c*ns_c
    e = 1
    # loop element id for coarse mesh
    for iy_c = 1:2*ns_c    
        for ix_c = 1:4*ns_c 
            if ix_c > ns_c && ix_c <= 3*ns_c && iy_c <= ns_c
                continue
            end

            # node id for fine mesh
            iy =  (iy_c - 1)*ele_c + 1

            if ix_c > 3*ns_c && iy_c <= ns_c
                i_block = 2
                ix =(ix_c - 3*ns_c - 1)*ele_c + 1
            else
                i_block = 1
                ix =(ix_c - 1)*ele_c + 1
            end
            
            # loop coarse mesh element nodes
            for sub_iy = 1 : ele_c + 1
                for sub_ix = 1:ele_c + 1
                   
                    n = ix + sub_ix - 1 + row_incr[sub_iy - 1 + iy, i_block]
                    interp_e[n, :] .= e, sub_ix + (sub_iy - 1)*(ele_c + 1)
                    #@info "node ", n, e, sub_ix + (sub_iy - 1)*(ele_c + 1), sub_ix, sub_iy

                end
            end
            
            e += 1
        end
    end

    # node n is at element e = interp_e[n, 1]
    # x(n) = x[e.elnodes] * sdata[n, :]
    sdata = zeros(Float64, (ele_c + 1)^2, (porder_c+1)^2)
    for sub_iy = 1 : ele_c + 1
        for sub_ix = 1:ele_c + 1
        
            n = sub_ix + (sub_iy - 1)*(ele_c + 1)
            ξ = 2*[(sub_ix-1)/(ele_c) ; (sub_iy-1)/(ele_c)] .- 1.0
            sData = (porder_c == 1 ? getShapeQuad4(ξ) : getShapeQuad9(ξ))
            sdata[n, :] = sData[:, 1]
        end
    end


    ##########
    nnodes_c = 6*(ns_c*porder_c + 1)*(ns_c*porder_c + 1) - 5*(ns_c*porder_c + 1)
    ind_θf_to_θc = zeros(Int64, nnodes_c)
    for iy_c = 1:2*ns_c*porder_c+1
        iy = (iy_c-1)*sub_ele + 1
        for ix_c = 1:4*ns_c*porder_c+1
            ix  = (ix_c-1)*sub_ele + 1

            if iy_c < ns_c*porder_c+1 && ix_c > ns_c*porder_c+1 && ix_c < 3*ns_c*porder_c+1
                continue
            elseif iy_c < ns_c*porder_c+1 && ix_c >= 3*ns_c*porder_c+1
                ind = row_incr[iy, 2] 
                ind_c = row_incr_c[iy_c, 2] 
                
                n_c = ix_c + ind_c - (3*ns_c*porder_c)
                n_f = ix + ind - (3*ns*porder)
            else
                ind = row_incr[iy, 1]
                ind_c = row_incr_c[iy_c, 1]
                n_c = ix_c + ind_c
                n_f = ix + ind
            end
            ind_θf_to_θc[n_c] = n_f
        end
    end

    # @info ind_θf_to_θc
    # @info interp_e
    # @info sub_ele
    # error("stop")
    return interp_e, sdata, ind_θf_to_θc
end

"""
Interpolate from θ_c on coarse grid to θ_f on fine grid
"""
function Interp_θc_to_θf(domain_c::Domain, interp_e::Array{Int64, 2}, interp_sdata::Array{Float64, 2}, θ_c::Array{Float64,1})
    nnodes = size(interp_e, 1)

    elem_c = domain_c.elements

    @assert(length(θ_c) == size(domain_c.nodes, 1))
    
    θ_f = zeros(Float64, nnodes)
    for i = 1:nnodes 
        e, n  = interp_e[i, :]

        el_nodes = getNodes(elem_c[e])

        θ_f[i] = interp_sdata[n,:]' * θ_c[el_nodes]
    end
    return θ_f
end


"""
Interpolate from θ_c on coarse grid to θ_f on fine grid
"""
function Get_Raw_From_θ_Dam(ind_θf_to_θc::Array{Int64, 1},  θ_f::Array{Float64,1})
    θ_c = θ_f[ind_θf_to_θc]
    θ = Get_θ(θ_c)
    return θ
end


function Get_θ_Dam_From_Raw(domain_c::Domain, interp_e::Array{Int64, 2}, interp_sdata::Array{Float64, 2}, θ::Array{Float64,1})
    θ_c = Get_θ_Dam(θ)
    θ_f = Interp_θc_to_θf(domain_c, interp_e, interp_sdata, θ_c)
end 

function Interp_Test()
    function Quad(nodes::Array{Float64, 2})
        a, b, c, d, e, f = 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
        val = zeros(Float64, size(nodes,1))
        for i = 1:size(nodes,1)
            x, y = nodes[i, :]
            #val[i] = a*x^2 + b*x*y + c*y^2 + d*x + e*y +f
            val[i] = ((x - 200.0)^2 + (y - 120.0)^2)/20000.0
        end
        return val
    end

    ns, ns_obs, porder, problem, ns_c, porder_c = 4, 2, 1, "Static", 2, 2
    phys_params = Params(ns, ns_obs, porder, problem, ns_c, porder_c)



    θ_f = Quad(phys_params.domain.nodes)
    θ_c = Quad(phys_params.domain_c.nodes)
    θ_fc = Interp_θc_to_θf(phys_params.domain_c, phys_params.interp_e, phys_params.interp_sdata, θ_c)
    θ_cf = θ_fc[phys_params.ind_θf_to_θc]
    @info "norm(θ_f - θ_fc)", norm(θ_f - θ_fc)
    @info "norm(θ_c - θ_cf)", norm(θ_c - θ_cf)


end

function Damage_Test()
    ns, ns_obs, porder, problem, ns_c, porder_c = 4, 2, 2, "Static", 2, 2
    phys_params = Params(ns, ns_obs, porder, problem, ns_c, porder_c)

    θ_dam, data = Run_Damage(phys_params, "Analytic", nothing, phys_params.P1, phys_params.P2, "Figs/disp", "Figs/YoungsModule", -1.0)

end


######################

# ns, ns_obs, porder, problem, ns_c, porder_c = 2, 3, 2, "Static", 1, 1
# phys_params = Params(ns, ns_obs, porder, problem, ns_c, porder_c)

# nθ = size(phys_params.domain_c.nodes, 1)
    
# θ_dam, data = Run_Damage(phys_params, "Analytic", nothing, "Figs/disp", "Figs/YoungsModule", -1.0)
# θ_c = zeros(Float64, nθ)
# θ_dam, data = Run_Damage(phys_params, "Piecewise", θ_c, "Figs/disp", "Figs/YoungsModule", -1.0)


#Interp_Test()
#Damage_Test()


function Kernel_function(domain_c, σ::Float64, s0::Float64, τ::Float64)
    nodes = domain_c.nodes
    nθ = size(nodes, 1)
    Cov = zeros(Float64, nθ, nθ)
    
    # σ² * exp(-||x-y||ᵗ/(2*s0²))
    for i = 1:nθ
        for j = 1:nθ
            Cov[i, j] = σ * exp(- ((nodes[i, 1] - nodes[j, 1])^2 + (nodes[i, 2] - nodes[j, 2])^2)^(τ/2) /(2*s0))
        end
    end

    Cov_svd = svd(Cov)
    
    return Cov, Cov_svd.U, Cov_svd.S
end