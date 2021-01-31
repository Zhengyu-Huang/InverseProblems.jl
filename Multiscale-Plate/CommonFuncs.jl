using Revise
using PyCall
using Test 
using MAT
using NNFEM
using PyPlot
using JLD2
using LinearAlgebra
using Distributions, Random
using ForwardDiff
using DelimitedFiles
mpl = pyimport("tikzplotlib")

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
font0 = Dict(
        "font.size" => 16,
        "axes.labelsize" => 16,
        "xtick.labelsize" => 16,
        "ytick.labelsize" => 16,
        "legend.fontsize" => 16,
)
merge!(rcParams, font0)



"""
Feyel, Frédéric, and Jean-Louis Chaboche. 
"FE2 multiscale approach for modelling the elastoviscoplastic behaviour of long fibre SiC/Ti composite materials." 
Computer methods in applied mechanics and engineering 183.3-4 (2000): 309-330.

Geometry:
The plate size is 30 mm by 5mm 
These fibres are made of SiC with diameter about 0.65mm,
The volume fraction is about 33%. 

Domain: 60k by 10k
fibers: k by k
fiber number 200

Property:
Fibers are made of SiC, which are assumed to be isotropic and elastic, with
https://www.azom.com/properties.aspx?ArticleID=42
ρ = 3200 kg/m^3  E = 400GPa   ν = 0.35
The matrix is made of titanium, which are assumed to be elasto-plastic titanium material,  
ρ = 4500 kg/m^3;  E = 100GPa	 K=10e+9  ν =0.2   σY=970 MPa

length scale cm
"""
function generateEleType(nxc, nyc, fiber_size, fiber_fraction, fiber_distribution)
    if fiber_distribution == "Uniform"
        nx,ny = nxc*fiber_size , nyc*fiber_size
        # a,fiber_size,a,fiber_size,a void ratio is about fiber_size^2/(a+fiber_size)^2
        ele_type_x =  zeros(Int, nx,1)
        ele_type_y =  zeros(Int, 1,ny)

        a = round(Int, fiber_size*(1.0/sqrt(fiber_fraction)-1))
        if a <2 
            print("Mesh for the multiscale simulaion is too coarse")
            error()
        end
        rx = nx%(a + fiber_size)
        counter = a
        for i = round(Int, (rx+a)/2.0)+1:nx
            if counter%(a+fiber_size) > a-1
                ele_type_x[i] = 1
            end
            counter += 1
        end 

        ry = ny%(a + fiber_size)
        counter = a
        for i = round(Int, (ry+a)/2.0)+1:ny
            if counter%(a+fiber_size) > a-1
                ele_type_y[i] = 1
            end
            counter += 1
        end 
        ele_type = ele_type_x * ele_type_y

    else
        Random.seed!(123)
        d = Binomial(1, fiber_fraction)
        ele_type_c =  rand(d, nxc, nyc)
        ele_type =  kron(ele_type_c, ones(Int, fiber_size, fiber_size))
    end

    return ele_type
end

# function BoundaryCondition(tid, nx, ny, Lx = 1.0, Ly = 0.5)
#     nnodes, neles = (nx + 1)*(ny + 1), nx*ny
#     Lx, Ly = 1.0, 0.5
#     x = np.linspace(0.0, Lx, nx + 1)
#     y = np.linspace(0.0, Ly, ny + 1)


#     X, Y = np.meshgrid(x, y)
#     nodes = zeros(nnodes,2)
#     nodes[:,1], nodes[:,2] = X'[:], Y'[:]
#     ndofs = 2

#     EBC, g = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)
#     FBC, fext = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)
#     gt = nothing

#     # setting EBC:
#     # tid = 1XX   |==|--> 
#     # tid = 2XX   
#     if div(tid,100)==1
#         EBC[collect(1:nx+1), :] .= -1 # fix bottom
#         FBC[collect((nx+1)*ny+1:(nx+1)*(ny+1)), :] .= -2 # force on the top
#     elseif div(tid,100)==2
#         EBC[collect(1:nx+1:(nx+1)*(ny+1)), :] .= -1 # fix left
#         FBC[collect(nx+1:nx+1:(nx+1)*(ny+1)), :] .= -2 # force on the right        
#     elseif div(tid,100)==3
#         EBC[collect(1:nx+1:(nx+1)*(ny+1)), :] .= -1 # fix left
#         FBC[collect(2:nx+1), :] .= -2 # force on the bottom
#     end

#     F1 = 2e2 /force_scale #gcm/ms^2 compress/pull
#     F2 = 2e1 /force_scale #gcm/ms^2 bend 
#     #Bending or Pulling
#     if tid==100
#         fext[collect((nx+1)*ny+1:(nx+1)*(ny+1)), 1] .= 0
#         fext[collect((nx+1)*ny+1:(nx+1)*(ny+1)), 2] .= F1
#         fext[[(nx+1)*ny+1,(nx+1)*(ny+1)], :] /= 2.0
#     elseif tid==101
#         fext[collect((nx+1)*ny+1:(nx+1)*(ny+1)), 1] .= 0
#         fext[collect((nx+1)*ny+1:(nx+1)*(ny+1)), 2] .= -2*F1
#         fext[[(nx+1)*ny+1,(nx+1)*(ny+1)], :] /= 2.0    

#     elseif tid == 200
#         fext[collect(nx+1:nx+1:(nx+1)*(ny+1)), 1] .= F1
#         fext[collect(nx+1:nx+1:(nx+1)*(ny+1)), 2] .= 0
#         # fext[]
#         fext[[nx+1; (nx+1)*(ny+1)], :] /= 2.0
#     elseif tid == 201
#         fext[collect(nx+1:nx+1:(nx+1)*(ny+1)), 1] .= -F1
#         fext[collect(nx+1:nx+1:(nx+1)*(ny+1)), 2] .= 0
#         fext[[nx+1; (nx+1)*(ny+1)], 2] /= 2.0
#     elseif tid == 202
#         fext[collect(nx+1:nx+1:(nx+1)*(ny+1)), 1] .= 0
#         fext[collect(nx+1:nx+1:(nx+1)*(ny+1)), 2] .= F2
#         fext[[nx+1; (nx+1)*(ny+1)], :] /= 2.0
#     elseif tid == 203
#         fext[collect(nx+1:nx+1:(nx+1)*(ny+1)), 1] .= F1/sqrt(2)
#         fext[collect(nx+1:nx+1:(nx+1)*(ny+1)), 2] .= F2/sqrt(2)
#         fext[[nx+1;(nx+1)*(ny+1)], 2] /= 2.0

#     elseif tid == 300
#         # data_type == "Bending"
#         # force parameter
#         function gauss(L, n, x0; σ=0.2)
#             x = collect(LinRange(0, L, n+1))
#             return 1.0/(sqrt(2*pi*σ^2)) * exp.(-0.5*(x .- x0).^2/σ^2)
#         end

#         fext[collect(2:nx+1), 1] .= 0.0
#         fext[collect(2:nx+1), 2] .= F2 * gauss(Lx, nx, Lx*5.0/6.0)[2:end]
#         fext[[2;nx+1], :] /= 2.0
#     else
#         error("tid = $tid is not understood")
#     end

#     dof_to_active = findall(FBC[:].==-2)
#     ft = t->fext[:][dof_to_active]*sin(π*t/(2T))
#     # @show ft(T)
#     return nodes, EBC, g, gt, FBC, fext, ft
# end




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

function meshgrid(x::Array{Float64,1}, y::Array{Float64,1})
    nx, ny = length(x), length(y)
    X = zeros(Float64, ny, nx)
    Y = zeros(Float64, ny, nx)
    for i = 1:ny
        X[i, :] .= x
    end
    for i = 1:nx
        Y[:, i] .= y
    end

    return X, Y

end

function BoundaryCondition(tid, nx, ny, porder=2, Lx = 1.0, Ly = 0.5; force_scale=5.0)
    nnodes, neles = (nx*porder + 1)*(ny*porder + 1), nx*ny
    
    x = Array(LinRange(0.0, Lx, nx*porder + 1))
    y = Array(LinRange(0.0, Ly, ny*porder + 1))


    X, Y = meshgrid(x, y)
    nodes = zeros(nnodes,2)
    nodes[:,1], nodes[:,2] = X'[:], Y'[:]
    ndofs = 2

    EBC, g = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)
    FBC, fext = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)
    gt = nothing

    #todo only change 203

    # setting EBC:
    # tid = 1XX   |==|--> 
    # tid = 2XX   
    if div(tid,100)==1
        EBC[collect(1:nx*porder+1), :] .= -1 # fix bottom
        FBC[collect((nx*porder+1)*ny*porder+1:(nx*porder+1)*(ny*porder+1)), :] .= -2 # force on the top
    elseif div(tid,100)==2
        EBC[collect(1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), :] .= -1 # fix left
        FBC[collect(nx*porder+1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), :] .= -2 # force on the right        
    elseif div(tid,100)==3
        EBC[collect(1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), :] .= -1 # fix left
        FBC[collect(2:nx*porder+1), :] .= -2 # force on the bottom
    end

    # Uniform conservative pressure load 
    # F1 = 2e2 /force_scale #gcm/ms^2 compress/pull
    # F2 = 2e1 /force_scale #gcm/ms^2 bend 

    P1 = 80000 /force_scale #gcm/ms^2 compress/pull
    P2 = 8000 /force_scale #gcm/ms^2 bend 
    P3 = 30000/force_scale
    @show P1, P2
    ngp = 3
    #Bending or Pulling
    # 
    if tid==100
        F1, F2 = ComputeLoad(Lx, nx, porder, ngp, "Constant",  [0, P1])
        fext[collect((nx*porder+1)*ny*porder+1:(nx*porder+1)*(ny*porder+1)), 1] .= F1
        fext[collect((nx*porder+1)*ny*porder+1:(nx*porder+1)*(ny*porder+1)), 2] .= F2
        
    elseif tid==101
        F1, F2 = ComputeLoad(Lx, nx, porder, ngp, "Constant",  [0, -P1])
        fext[collect((nx*porder+1)*ny*porder+1:(nx*porder+1)*(ny*porder+1)), 1] .= F1
        fext[collect((nx*porder+1)*ny*porder+1:(nx*porder+1)*(ny*porder+1)), 2] .= F2

    elseif tid==102
        F1, F2 = ComputeLoad(Lx, nx, porder, ngp, "Constant",  [P3, 0])
        fext[collect((nx*porder+1)*ny*porder+1:(nx*porder+1)*(ny*porder+1)), 1] .= F1
        fext[collect((nx*porder+1)*ny*porder+1:(nx*porder+1)*(ny*porder+1)), 2] .= F2 

    elseif tid==103
        F1, F2 = ComputeLoad(Lx, nx, porder, ngp, "Constant",  [P3/sqrt(2.0), P1/sqrt(2.0)])
        fext[collect((nx*porder+1)*ny*porder+1:(nx*porder+1)*(ny*porder+1)), 1] .= F1
        fext[collect((nx*porder+1)*ny*porder+1:(nx*porder+1)*(ny*porder+1)), 2] .= F2 

    elseif tid==104
        F1, F2 = ComputeLoad(Lx, nx, porder, ngp, "Constant",  [-P3, 0])
        fext[collect((nx*porder+1)*ny*porder+1:(nx*porder+1)*(ny*porder+1)), 1] .= F1
        fext[collect((nx*porder+1)*ny*porder+1:(nx*porder+1)*(ny*porder+1)), 2] .= F2 
    
    elseif tid==105
        F1, F2 = ComputeLoad(Lx, nx, porder, ngp, "Constant",  [0, 0.75*P1])
        fext[collect((nx*porder+1)*ny*porder+1:(nx*porder+1)*(ny*porder+1)), 1] .= F1
        fext[collect((nx*porder+1)*ny*porder+1:(nx*porder+1)*(ny*porder+1)), 2] .= F2

    elseif tid==106
        F1, F2 = ComputeLoad(Lx, nx, porder, ngp, "Constant",  [0.75*P3, 0])
        fext[collect((nx*porder+1)*ny*porder+1:(nx*porder+1)*(ny*porder+1)), 1] .= F1
        fext[collect((nx*porder+1)*ny*porder+1:(nx*porder+1)*(ny*porder+1)), 2] .= F2 


    elseif tid == 200
        F1, F2 = ComputeLoad(Ly, ny, porder, ngp, "Constant",  [P1, 0])
        fext[collect(nx*porder+1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), 1] .= F1
        fext[collect(nx*porder+1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), 2] .= F2
    
    elseif tid == 201
        F1, F2 = ComputeLoad(Ly, ny, porder, ngp, "Constant",  [-P1, 0])
        fext[collect(nx*porder+1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), 1] .= F1
        fext[collect(nx*porder+1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), 2] .= F2

    elseif tid == 202
        F1, F2 = ComputeLoad(Ly, ny, porder, ngp, "Constant",  [0, P2])
        fext[collect(nx*porder+1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), 1] .= F1
        fext[collect(nx*porder+1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), 2] .= F2

    elseif tid == 203
        F1, F2 = ComputeLoad(Ly, ny, porder, ngp, "Constant",  [P1/sqrt(2), P2/sqrt(2)])
        fext[collect(nx*porder+1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), 1] .= F1
        fext[collect(nx*porder+1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), 2] .= F2

    elseif tid == 204
        F1, F2 = ComputeLoad(Ly, ny, porder, ngp, "Constant",  [0, -P2])
        fext[collect(nx*porder+1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), 1] .= F1
        fext[collect(nx*porder+1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), 2] .= F2

    elseif tid == 205
        F1, F2 = ComputeLoad(Ly, ny, porder, ngp, "Constant",  [0.75*P1, 0])
        fext[collect(nx*porder+1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), 1] .= F1
        fext[collect(nx*porder+1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), 2] .= F2

    elseif tid == 206
        F1, F2 = ComputeLoad(Ly, ny, porder, ngp, "Constant",  [0, 0.75*P2])
        fext[collect(nx*porder+1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), 1] .= F1
        fext[collect(nx*porder+1:nx*porder+1:(nx*porder+1)*(ny*porder+1)), 2] .= F2

    elseif tid == 300

        F1, F2 = ComputeLoad(Lx, nx, porder, ngp, "Gaussian",  [P2, Lx*5.0/6.0, 0.2])
        fext[collect(2:nx*porder+1), 1] .= F1[2:end]
        fext[collect(2:nx*porder+1), 2] .= F2[2:end]
    else
        error("tid = $tid is not understood")
    end

    dof_to_active = findall(FBC[:].==-2)
    ft = t->fext[:][dof_to_active]*sin(π*t/(T))
    # @show ft(T)


    npoints = (nx+1)*(ny+1)
    node_to_point = zeros(Int64, nnodes) .- 1
    pointid = 1
    for j = 1:2:(ny*porder + 1)
        for i = 1:2:(nx*porder + 1)
            node_to_point[(j-1)*(nx*porder + 1) + i] = pointid
            pointid += 1
        end
    end

    return nodes, EBC, g, gt, FBC, fext, ft, npoints, node_to_point
end

function Get_Obs(domain, nx::Int64, ny::Int64, porder::Int64)
    # observation is the top right and top left
    p_ids = [(nx*porder+1)*(ny*porder+1); (nx*porder+1)*(ny*porder) + 1]
    n_frame = length(domain.history["state"])
    obs = zeros(Float64, n_frame-1, 2*length(p_ids))
    for it = 2:n_frame
        for ip = 1:length(p_ids)
            obs[it-1, 2*ip-1] = domain.history["state"][it][p_ids[ip]]  # x displacement
            obs[it-1, 2*ip]   = domain.history["state"][it][p_ids[ip] + (nx*porder+1)*(ny*porder+1)]  # y displacement
        end
    end

    return obs
end

function Run_Homogenized(θ::Array{Float64,1}, θ_scale::Array{Float64,1}, ρ::Float64, tid::Int64, force_scale::Float64; 
    T::Float64 = 200.0, NT::Int64 = 200, nx::Int64 = 10, ny::Int64 = 5, porder::Int64 = 2)

    # "E"=> 1e+6, "nu"=> 0.2,"sigmaY"=>0.97e+4, "K"=>1e+5
    E, nu, sigmaY, K = θ .* θ_scale
    
    
    prop = Dict("name"=> "PlaneStressPlasticity","rho"=> ρ, "E"=> E, "nu"=> nu, "sigmaY"=>sigmaY, "K"=>K)
    
    
    # nx_f, ny_f = 12, 4
    # homogenized computaional domain
    # number of elements in each directions
    nodes, EBC, g, gt, FBC, fext, ft = BoundaryCondition(tid, nx, ny, porder; force_scale=force_scale)
    ndofs=2
    elements = []
    for j = 1:ny
        for i = 1:nx 
            n = (nx*porder+1)*(j-1)*porder + (i-1)porder+1
            #element (i,j)
            if porder == 1
                #   4 ---- 3
                #
                #   1 ---- 2
                elnodes = [n, n + 1, n + 1 + (nx + 1), n + (nx + 1)]
            elseif porder == 2
                #   4 --7-- 3
                #   8   9   6 
                #   1 --5-- 2
                elnodes = [n, n + 2, n + 2 + 2*(2*nx+1),  n + 2*(2*nx+1), n+1, n + 2 + (2*nx+1), n + 1 + 2*(2*nx+1), n + (2*nx+1), n+1+(2*nx+1)]
            else
                error("polynomial order error, porder= ", porder)
            end
            coords = nodes[elnodes,:]
            push!(elements,SmallStrainContinuum(coords,elnodes, prop, 3))
        end
    end
    
    

    domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
    state = zeros(domain.neqs)
    ∂u = zeros(domain.neqs)
    globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)
    
    assembleMassMatrix!(globdat, domain)
    updateStates!(domain, globdat)
    
    
    ts = LinRange(0, T, NT+1)
    adaptive_solver_args = Dict("Newmark_rho"=> 0.0, 
    "Newton_maxiter"=>10, 
    "Newton_Abs_Err"=>1e-4, 
    "Newton_Rel_Err"=>1e-6, 
    "damped_Newton_eta" => 1.0)
    
    globdat, domain, ts = AdaptiveSolver("NewmarkSolver", globdat, domain, T, NT, adaptive_solver_args)
    
    
    
    obs = Get_Obs(domain, nx, ny, porder)

    return domain, obs
    
end
