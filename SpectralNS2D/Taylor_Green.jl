include("SpectralNS.jl")

ν=1.0e-3;   # viscosity
nx=128;     # resolution in x
ny=128;     # resolution in y
Δt=1e-1;    # time step
TF=1000.0;  # final time

Lx, Ly = 2*pi, 2*pi

mesh = Spectral_Mesh(nx, ny, Lx, Ly)

ω0 = zeros(Float64, nx, ny)
Δx, Δy, xx, yy = mesh.Δx, mesh.Δy, mesh.xx, mesh.yy
for i = 1:nx
    for j = 1:ny
        x, y = xx[i], yy[j]
        ω0[i,j] = -2*cos(x)*cos(y)

    end
end

f = zeros(Float64, nx, ny)


solver = SpectralNS_Solver(mesh, ν::Float64, f, ω0)   

Solve!(solver, Δt)