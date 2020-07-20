include("SpectralNS.jl")
using LinearAlgebra
using Distributions
using Random

struct Params
    ν::Float64
    nx::Int64
    ny::Int64
    Lx::Float64
    Ly::Float64
    method::String
    nt::Int64
    T::Float64

    #observation
    Δd_x::Int64
    Δd_y::Int64 
    Δd_t::Int64 

    n_data::Int64

    #parameter standard deviation
    σ::Float64
end

function Params()
    ν = 1.0e-2   # viscosity
    nx, ny = 64, 64   # resolution in x
         # resolution in y
    Lx, Ly = 2*pi, 2*pi
    method="Crank-Nicolson" # RK4 or Crank-Nicolson
    nt = 500;    # time step
    T = 0.5;      # final time
    #observation
    Δd_x, Δd_y, Δd_t = 8, 8, 100

    n_data = (div(nx-1,Δd_x)+1)*(div(ny-1,Δd_y)+1)*(div(nt, Δd_t))
    #parameter standard deviation
    σ = 2*pi^2
    Params(ν, nx, ny, Lx, Ly, method, nt, T, Δd_x, Δd_y, Δd_t, n_data, σ)
end


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


function Initial_ω0_KL(mesh::Spectral_Mesh, params::Params)
    # consider C = -Δ^{-1}
    # C u = λ u  => -u = λ Δ u
    # u = e^{ik⋅x}, λ Δ u = -λ |k|^2 u = - u => λ = 1/|k|^2
    # basis are cos(k⋅x)/2π^2 and sin(k⋅x)/2π^2, k!=(0,0), zero mean.
    # ω = ∑ ak cos(k⋅x)/2π^2|k|^2 + ∑ bk sin(k⋅x)/2π^2|k|^2,    &  kx + ky > 0 or (kx + ky = 0 and kx > 0) 
    #   = ∑ (ak/(4π^2|k|^2) - i bk/(4π^2|k|^2) )e^{ik⋅x} + (ak/(4π^2|k|^2) + i bk/(4π^2|k|^2) )e^{-ik⋅x}

    # Initialize with all fourier modes
    
    nx, ny = mesh.nx, mesh.ny
    kxx, kyy = mesh.kxx, mesh.kyy
    
    Random.seed!(123);
    abk = rand(Normal(0, params.σ), nx, ny, 2)
    
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
    
    ω0
end



function Compute_Seq_Pairs(trunc_KL::Int64)
    seq_pairs = zeros(Int64, trunc_KL, 2)
    trunc_Nx = trunc(Int64, sqrt(2*trunc_KL)) + 1
    
    seq_pairs = zeros(Int64, (trunc_Nx+1)^2 - 1, 2)
    seq_pairs_mag = zeros(Int64, (trunc_Nx+1)^2 - 1)
    
    seq_pairs_i = 0
    for kx = -trunc_Nx:trunc_Nx
        for ky = -trunc_Nx:trunc_Nx
            if (kx + ky > 0 || (kx + ky == 0 && kx > 0)) 
                seq_pairs_i += 1
                seq_pairs[seq_pairs_i, :] .= kx, ky
                seq_pairs_mag[seq_pairs_i] = kx^2 + ky^2
            end
        end
    end
    
    seq_pairs = seq_pairs[sortperm(seq_pairs_mag), :]
    return seq_pairs[1:trunc_KL, :]
end

function Initial_ω0_KL(mesh::Spectral_Mesh, abk::Array{Float64,2}, seq_pairs::Array{Int64,2})
    # consider C = -Δ^{-1}
    # C u = λ u  => -u = λ Δ u
    # u = e^{ik⋅x}, λ Δ u = -λ |k|^2 u = - u => λ = 1/|k|^2
    # basis are cos(k⋅x)/2π^2 and sin(k⋅x)/2π^2, k!=(0,0), zero mean.
    # ω = ∑ ak cos(k⋅x)/2π^2|k|^2 + ∑ bk sin(k⋅x)/2π^2|k|^2,    &  kx + ky > 0 or (kx + ky = 0 and kx > 0) 
    #   = ∑ (ak/(4π^2|k|^2) - i bk/(4π^2|k|^2) )e^{ik⋅x} + (ak/(4π^2|k|^2) + i bk/(4π^2|k|^2) )e^{-ik⋅x}

    # Initialize with first trunc_KL fourier modes
    
    nx, ny = mesh.nx, mesh.ny
    kxx, kyy = mesh.kxx, mesh.kyy
    
    ω0_hat = zeros(ComplexF64, nx, ny)
    ω0 = zeros(Float64, nx, ny)
    trunc_KL = size(abk,1)
    for i = 1:trunc_KL
        kx, ky = seq_pairs[i,:]
        ak, bk = abk[i,:]
        ix = (kx >= 0 ? kx + 1 : nx + kx + 1) 
        iy = (ky >= 0 ? ky + 1 : ny + ky + 1) 
        
        ω0_hat[ix, iy] = (ak - bk*im)/(4*pi^2*(kx^2+ky^2))
        
        # 1 => 1, i => n-i+2
        ω0_hat[(ix==1 ? 1 : nx-ix+2), (iy==1 ? 1 : ny-iy+2)] = (ak + bk*im)/(4*pi^2*(kx^2+ky^2))
    end
    ω0_hat .*= mesh.alias_filter
    Trans_Spectral_To_Grid!(mesh, nx*ny * ω0_hat, ω0)

    ω0
end


function Generate_Data(params::Params)


    ν = params.ν
    nx, ny = params.nx, params.ny
    Lx, Ly = params.Lx, params.Ly
    nt, T = params.nt, params.T
    method = params.method 
    Δd_x, Δd_y, Δd_t = params.Δd_x, params.Δd_y, params.Δd_t
  
    
    mesh = Spectral_Mesh(nx, ny, Lx, Ly)
    
    
    ω0 = Initial_ω0_KL(mesh, params)
    
        
    fx,fy = Force(mesh)
    

    solver = SpectralNS_Solver(mesh, ν, fx, fy, ω0)  
    Visual(mesh, ω0, "ω", "vor0.png")
    Visual(mesh, solver.u, "u", "u0.png")
    Visual(mesh, solver.v, "v", "v0.png")
    Δt_max = Stable_Δt(mesh, ν, solver.u, solver.v)
    
    Δt = T/nt
    
    # initialize observation 
    
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
    
    return ω0, data
    
end


function RandomInit_Main(abk::Array{Float64,2}, seq_paris::Array{Int64,2}, params::Params)

    ν = params.ν
    nx, ny = params.nx, params.ny
    Lx, Ly = params.Lx, params.Ly
    nt, T = params.nt, params.T
    method = params.method 
    Δd_x, Δd_y, Δd_t = params.Δd_x, params.Δd_y, params.Δd_t
    


    mesh = Spectral_Mesh(nx, ny, Lx, Ly)
    ω0 = Initial_ω0_KL(mesh, abk, seq_pairs)        
    fx,fy = Force(mesh)
    

    solver = SpectralNS_Solver(mesh, ν, fx, fy, ω0)  

    Δt_max = Stable_Δt(mesh, ν, solver.u, solver.v)
    
    Δt = T/nt
    
    # initialize observation 
    d_x = Array(1:Δd_x:nx)
    d_y = Array(1:Δd_y:ny)
    d_t = Array(Δd_t:Δd_t:nt)
    
    data = zeros(Float64, size(d_x,1), size(d_y,1), size(d_t,1))
    
    
    for i = 1:nt
        Solve!(solver, Δt, method)
        if i%Δd_t == 0
            Update_Grid_Vars!(solver)
            data[:, :, Int64(i/Δd_t)] = solver.ω[d_x, d_y]
        end
    end
    
    return data
    
end


phys_params = Params()

# data
ω0, t_mean =  Generate_Data(phys_params)
t_cov = Array(Diagonal(fill(1.0, phys_params.n_data))) 


