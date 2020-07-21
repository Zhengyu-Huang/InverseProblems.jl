include("SpectralNS.jl")
using LinearAlgebra

"""
This is a Taylor Green Vortex example
"""
function TGV_Sol(xx, yy, ν, ub, vb, t)
    """
    x, y ∈ [0,2π]×[0,2π]
    """
    nx, ny = size(xx, 1), size(yy, 1)
    ω = zeros(Float64, nx, ny)
    u = zeros(Float64, nx, ny)
    v = zeros(Float64, nx, ny)
    p = zeros(Float64, nx, ny)
    for i = 1:nx
        for j = 1:ny
            x, y = xx[i], yy[j]
            F = exp(-2*ν*t)

            
            u[i,j] = ub  - cos(x - ub*t)*sin(y - vb*t)*F
            v[i,j] = vb  + sin(x - ub*t)*cos(y - vb*t)*F

            #todo check pressure
            p[i,j] =      -(cos(2*x - 2*ub*t) + cos(2*y - 2*vb*t))*F^2/4

            ω[i,j] =     2*cos(x - ub*t)*cos(y - vb*t)*F
        end
    end

    return ω, u, v, p
end



# ν=1.0e-1;   # viscosity
# nx=128;     # resolution in x
# ny=128;     # resolution in y
# Δt=5.0e-3;    # time step
# T=1.0;      # final time
# method="Crank-Nicolson" # RK4 or Crank-Nicolson
# Lx, Ly = 2*pi, 2*pi
# ub, vb = 1.0,1.0

ν=1.0e-1;   # viscosity
nx=128;     # resolution in x
ny=128;     # resolution in y
Δt=5.0e-3;    # time step
T=1.0;      # final time
method="RK4" # RK4 or Crank-Nicolson
Lx, Ly = 2*pi, 2*pi
ub, vb = 1.0,1.0

mesh = Spectral_Mesh(nx, ny, Lx, Ly)
Δx, Δy, xx, yy = mesh.Δx, mesh.Δy, mesh.xx, mesh.yy
ω0, u0, v0, p0 = TGV_Sol(xx, yy, ν, ub, vb, 0.0)

fx = zeros(Float64, nx, ny)
fy = zeros(Float64, nx, ny)

#solver = SpectralNS_Solver(mesh, ν, fx, fy, ω0, ub, vb) 
solver = SpectralNS_Solver(mesh, ν, fx, fy, u0, v0)  
  
Δt_max = Stable_Δt(mesh, ν, solver.u, solver.v)

nt = Int64(T/Δt)
for i = 1:nt
    Solve!(solver, Δt, method)
end

Update_Grid_Vars!(solver)


ω_sol, u_sol, v_sol = TGV_Sol(xx, yy, ν, ub, vb, T)

@info "\n||e_ω||_2/||ω||_2 =", norm(solver.ω - ω_sol)/norm(ω_sol), "\n||e_u||_2/||u||_2 =", norm(solver.u - u_sol)/norm(u_sol), "\n||v_ω||_2/||v||_2 =", norm(solver.v - v_sol)/norm(v_sol)


# Visual(mesh, solver.u, "u", "u.png")
# Visual(mesh, solver.v, "v", "v.png")
# Visual(mesh, solver.ω, "ω", "vor.png")
Visual(mesh, solver.u, "u", "u_num.png")

Visual(mesh, u_sol, "u", "u.png")
Visual(mesh, v_sol, "v", "v.png")
Visual(mesh, ω_sol, "ω", "vor.png")