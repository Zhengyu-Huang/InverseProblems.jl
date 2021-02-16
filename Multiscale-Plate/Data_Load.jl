force_scale = 5.0
fiber_size = 5
T = 200.0
NT = 200
nx, ny = 10, 5
porder = 2
fiber_fraction = 0.25
nxf, nyf =80*fiber_size,40*fiber_size
Lx, Ly = 1.0, 0.5

include("CommonFuncs.jl")



function Reference(tid::Int64, nx::Int64, ny::Int64, porder::Int64)
    
    full_state_history, full_fext_history = read_data("$(@__DIR__)/Data/order$porder/$(tid)_$(force_scale)_$(fiber_size).dat")
    p_ids = [(nx*porder+1)*(ny*porder+1); (nx*porder+1)*(ny*porder) + div(nx*porder, 2) + 1]
    obs = Get_Obs(full_state_history, nx, ny, porder, p_ids)
    @save "Data/order$porder/obs$(tid)_$(force_scale)_$(fiber_size).jld2" obs
end

function BuildDomain(T::Float64, nx::Int64, ny::Int64, Lx::Float64, Ly::Float64, tid::Int64, porder::Int64, force_scale::Float64, prop::Dict{String, Any})
    nodes, EBC, g, gt, FBC, fext, ft = BoundaryCondition(tid, T, nx, ny,porder, Lx, Ly; force_scale=force_scale)
    
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
    
    
    
    domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext), ft, gt
end

function PlotStress(tid::Int64, T::Float64, frame::Int64, nxf::Int64, nyf::Int64, fiber_size::Int64, porder::Int64, 
    Lx::Float64, Ly::Float64, force_scale::Float64)
    
    
    
    #visualize exact solution 
    close("all")
    prop_dummy = Dict("name"=> "PlaneStress","rho"=> 4.5*(1 - fiber_fraction) + 3.2*fiber_fraction, "E"=> 1e+6, "nu"=> 0.2)  #dummy
    
    domain, _, _ = BuildDomain(T,  nxf, nyf, Lx, Ly, tid, porder, force_scale, prop_dummy)
    full_state_history, full_fext_history = read_data("$(@__DIR__)/Data/order$porder/$(tid)_$(force_scale)_$(fiber_size).dat")
    strain, stress = read_strain_stress("Data/order$porder/$(tid)_$(force_scale)_$(fiber_size).dat")
    
    # stress size is NT
    # full_state_history size is NT+1
    vmin = 0; vmax = 2.5
    vmin, vmax = visσ(domain, nxf, nyf,  stress[frame], full_state_history[frame+1],vmin, vmax; scaling = scales)
    @show vmin, vmax
    xlabel("X (cm)")
    ylabel("Y (cm)")
    PyPlot.tight_layout()
    savefig("plate_multiscale_stress_reference$(tid).png")
    
    
    
end


# Reference(203, nxf, nyf, porder)
# Reference(300, nxf, nyf, porder)

PlotStress(100, T, div(NT,2), nxf , nyf , fiber_size , porder ,  Lx , Ly , force_scale )
PlotStress(102, T, div(NT,2), nxf , nyf , fiber_size , porder ,  Lx , Ly , force_scale )
#####################################################




