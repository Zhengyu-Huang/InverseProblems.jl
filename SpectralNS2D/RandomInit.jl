include("SpectralNS.jl")
using LinearAlgebra
using Distributions
using Random



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
    return ω0
end

function Initial_ω0_KL(mesh::Spectral_Mesh)
    # consider C = -Δ^{-1}
    # C u = λ u  => -u = λ Δ u
    # u = e^{ik⋅x}, λ Δ u = -λ |k|^2 u = - u => λ = 1/|k|^2
    # basis are cos(k⋅x)/2π^2 and sin(k⋅x)/2π^2, k!=(0,0), zero mean.
    # ω = ∑ ak cos(k⋅x)/2π^2|k|^2 + ∑ bk sin(k⋅x)/2π^2|k|^2,    &  kx + ky > 0 or (kx + ky = 0 and kx > 0) 
    #   = ∑ (ak/(4π^2|k|^2) - i bk/(4π^2|k|^2) )e^{ik⋅x} + (ak/(4π^2|k|^2) + i bk/(4π^2|k|^2) )e^{-ik⋅x}
    
    nx, ny = mesh.nx, mesh.ny
    kxx, kyy = mesh.kxx, mesh.kyy

    Random.seed!(123);
    abk = rand(Normal(0, 1), nx, ny, 2)

    ω0_hat = zeros(ComplexF64, nx, ny)
    ω0 = zeros(Float64, nx, ny)
    for ix = 1:nx
        for iy = 1:ny
            kx, ky = kxx[ix], kyy[iy]
            if (kx + ky > 0 || (kx + ky == 0 && kx > 0)) 
                ak, bk = abk[ix, iy, 1], abk[ix, iy, 2]
                ω0_hat[ix, iy] = (ak - bk*im)/(4*pi^2*(kx^2+ky^2))

                # 1 => 1, i => n-i+2
                ω0_hat[(ix==1 ? 1 : nx-ix+2), (iy==1 ? 1 : ny-iy+2)] = (ak + bk*im)/(4*pi^2*(kx^2+ky^2))
                
            end
        end
    end
    ω0_hat .*= mesh.alias_filter
    Trans_Spectral_To_Grid!(mesh, nx*ny * ω0_hat, ω0)
end

function RandomInit_Main(θ_2d::Array{Float64,2}, nx::Int64, ny::Int64, Lx::Float64, Ly::Float64, 
    ν::Float64, Δt::Float64, T::Float64, method::String="Crank-Nicolson")
    
    mesh = Spectral_Mesh(nx, ny, Lx, Ly)
    
    #ω0 = Initial_ω0(mesh, θ_2d)
    ω0 = Initial_ω0_KL(mesh)

    
    
    
    fx,fy = Force(mesh)
    
    
    
    solver = SpectralNS_Solver(mesh, ν, fx, fy, ω0)  
    Visual(mesh, ω0, "ω", "vor0.png")
    Visual(mesh, solver.u, "u", "u0.png")
    Visual(mesh, solver.v, "v", "v0.png")
    Δt_max = Stable_Δt(mesh, ν, solver.u, solver.v)

    nt = Int64(T/Δt)

    # initialize observation 
    Δd_x, Δd_y, Δd_t = 16, 16, 200
    d_x = Array(1:Δd_x:nx)
    d_y = Array(1:Δd_y:ny)
    d_t = Array(Δd_t:Δd_t:nt)
    
    data = zeros(Float64, size(d_x,1), size(d_y,1), size(d_t,1))

    
    
    for i = 1:nt
        Solve!(solver, Δt, method)
        if i%Δd_t == 0
            Update_Grid_Vars!(solver)
            Visual(mesh, solver.ω, "ω", "vor."*string(i)*".png")
            data[:, :, Int64(i/Δd_t)] = solver.ω[d_x, d_y]
        end
    end
    
    return data
    
end


ν=1.0e-3;   # viscosity
nx=128;     # resolution in x
ny=128;     # resolution in y
Δt=1.0e-4;    # time step
T=0.1;      # final time
method="Crank-Nicolson" # RK4 or Crank-Nicolson
Lx, Ly = 2*pi, 2*pi


Random.seed!(123);
trunc_K = 9
θ_2d = rand(Normal(0, 1), trunc_K+1, trunc_K+1)
RandomInit_Main(θ_2d, nx, ny, Lx, Ly, ν, Δt, T)