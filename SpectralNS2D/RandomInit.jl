include("SpectralNS.jl")
using LinearAlgebra




function Force(mesh::Spectral_Mesh)
    # F = [-∂Φ/∂y, ∂Φ/∂x] and Φ = cos((5x + 5y))
    nx, ny = mesh.nx, mesh.ny
    Δx, Δy, xx, yy = mesh.Δx, mesh.Δy, mesh.xx, mesh.yy

    fx, fy = zeros(Float64, nx, ny), zeros(Float64, nx, ny)
    for i = 1:nx
        for j = 1:ny
            x, y = xx[i], yy[j]
            fx[i,j] =  5*sin(5*(x+y))
            fy[i,j] = -5*sin(5*(x+y))
        end
    end
     
    return fx, fy
end


function Initial_ω0(mesh::Spectral_Mesh, θ_2d::Array{Float64,2})
    # correspond to (0,0), (0,1), ..., (0,K)
    #               (1,0), (1,1), ..., (1,K)
    #                 ......
    #               (K,0), (K,1), ..., (K,K)

    # initial condition is ∑ θ_{k} cos(k ∙ x)

    nx, ny = mesh.nx, mesh.ny
    kx_max, ky_max = size(θ_2d) .- 1
    @assert(kx_max < nx/3 && ky_max < ny/3)

    ω0 = zeros(Float64, nx, ny)
    ω0_hat = zeros(ComplexF64, nx, ny)

    ω0_hat[1:kx_max+1, 1:ky_max+1]                 .= nx*ny * θ_2d/2.0
    ω0_hat[nx:-1:nx-kx_max+1, ny:-1:ny-ky_max+1]   .= nx*ny * θ_2d[2:kx_max+1, 2:ky_max+1]/2.0
    ω0_hat[1, ny:-1:ny-ky_max+1]                   .= nx*ny * θ_2d[1, 2:ky_max+1]/2.0
    ω0_hat[nx:-1:nx-kx_max+1, 1]                   .= nx*ny * θ_2d[2:kx_max+1, 1]/2.0

    ω0_hat[1, 1] = nx*ny * θ_2d[1,1]

    Trans_Spectral_To_Grid!(mesh, ω0_hat, ω0)
    kxx, kyy = mesh.kxx, mesh.kyy

    Δx, Δy = mesh.Δx, mesh.Δy
    ω0_test = zeros(ComplexF64, nx, ny)
    for ix = 0:nx-1
        for iy = 0:ny-1
            x, y = ix*Δx, iy*Δy
            for kx = 0:kx_max 
                for ky = 0:ky_max 
                    ω0_test[ix+1,iy+1] += θ_2d[kx+1, ky+1]*cos(kx*x + ky*y)
                end
            end

        end
    end

    @info "Diff is :", norm(ω0 - ω0_test)



    return ω0
end


function RandomInit_Main(θ_2d::Array{Float64,2}, nx::Int64, ny::Int64, Lx::Float64, Ly::Float64, 
              ν::Float64, Δt::Float64, T::Float64, method::String="Crank-Nicolson")

    mesh = Spectral_Mesh(nx, ny, Lx, Ly)

    ω0 = Initial_ω0(mesh, θ_2d)
    
    fx,fy = Force(mesh)

    solver = SpectralNS_Solver(mesh, ν, fx, fy, ω0)    
    Δt_max = Stable_Δt(mesh, ν, solver.u, solver.v)

    nt = Int64(T/Δt)
    for i = 1:nt
        Solve!(solver, Δt, method)
    end

    Update_Grid_Vars!(solver)
end


ν=1.0e-2;   # viscosity
nx=64;     # resolution in x
ny=64;     # resolution in y
Δt=1.0e-3;    # time step
T=1.0;      # final time
method="Crank-Nicolson" # RK4 or Crank-Nicolson
Lx, Ly = 2*pi, 2*pi
