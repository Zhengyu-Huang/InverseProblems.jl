include("SpectralNS.jl")


function TGV_Sol(xx, yy, ν, t)
    """
    x, y ∈ [0,2π]×[0,2π]
    """
    nx, ny = size(xx, 1), size(yy, 1)
    ω = zeros(Float64, nx, ny)
    u = zeros(Float64, nx, ny)
    v = zeros(Float64, nx, ny)
    for i = 1:nx
        for j = 1:ny
            x, y = xx[i], yy[j]
            F = exp(-2*ν*t)

            ω[i,j] = -2*cos(x)*cos(y)*F
            u[i,j] = cos(x)*sin(y)*F
            v[i,j] = -sin(x)*cos(y)*F
        end
    end

    return ω, u, v
end


# ν=1.0e-3;   # viscosity
# nx=128;     # resolution in x
# ny=128;     # resolution in y
# Δt=1.0e-1;    # time step
# T=1000.0;  # final time


ν=1.0e-2;   # viscosity
nx=8;     # resolution in x
ny=8;     # resolution in y
Δt=1.0;    # time step
T=10.0;  # final time

Lx, Ly = 2*pi, 2*pi

mesh = Spectral_Mesh(nx, ny, Lx, Ly)

ω0 = zeros(Float64, nx, ny)
Δx, Δy, xx, yy = mesh.Δx, mesh.Δy, mesh.xx, mesh.yy
for i = 1:nx
    for j = 1:ny
        x, y = xx[i], yy[j]
        #ω0[i,j] = -2*cos(x)*cos(y)
        ω0[i,j] = exp(-((i*Δx-pi).^2+(j*Δy-pi+pi/4).^2)/(0.2))+exp(-((i*Δx-pi).^2+(j*Δy-pi-pi/4).^2)/(0.2))-0.5*exp(-((i*Δx-pi-pi/4).^2+(j*Δy-pi-pi/4).^2)/(0.4));
    end
end

f = zeros(Float64, nx, ny)


solver = SpectralNS_Solver(mesh, ν, f, ω0)   
nt = Int64(T/Δt)
for i = 1:nt
    Solve!(solver, Δt)
end

Update_Grid_Vars!(solver)


# ω_sol, u_sol, v_sol = TGV_Sol(xx, yy, ν, T)

# @info "\n||e_ω||_2/||ω||_2 =", norm(solver.ω - ω_sol)/norm(ω_sol), "\n||e_u||_2/||u||_2 =", norm(solver.u - u_sol)/norm(u_sol), "\n||v_ω||_2/||v||_2 =", norm(solver.v - v_sol)/norm(v_sol)


Visual(mesh, solver.u, "u", "u.png")
Visual(mesh, solver.u, "v", "v.png")
Visual(mesh, solver.ω, "ω", "vor.png")
