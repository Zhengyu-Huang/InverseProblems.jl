include("SpectralNS.jl")
using LinearAlgebra
using Distributions
using Random

mutable struct Params
    ν::Float64
    ub::Float64
    vb::Float64
    
    
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






# For fourier modes, consider the zero-mean real field
# u = ∑ f_k e^{ik⋅x}, we have  f_{k} = conj(f_{-k})
# we can also use cos(k⋅x)/2π^2 and sin(k⋅x)/2π^2 as basis (L2 norm is 1)
# with (kx + ky > 0 || (kx + ky == 0 && kx > 0)) 
function Mode_Helper(kx, ky)
    return (kx + ky > 0 || (kx + ky == 0 && kx > 0)) 
end


# Generate random numbers and initialize ω0 with all fourier modes
function Initial_ω0_KL(mesh::Spectral_Mesh, Nθ::Int64, seq_pairs, params::Params)
    # consider C = -Δ^{-1}
    # C u = λ u  => -u = λ Δ u
    # u = e^{ik⋅x}, λ Δ u = -λ |k|^2 u = - u => λ = 1/|k|^2
    # basis are cos(k⋅x)/√2π and sin(k⋅x)/√2π, k!=(0,0), zero mean.
    # ω = ∑ ak cos(k⋅x)/√2π|k|^2 + ∑ bk sin(k⋅x)/√2π|k|^2,    &  kx + ky > 0 or (kx + ky = 0 and kx > 0) 
    #   = ∑ (ak/(2√2π|k|^2) - i bk/(2√2π|k|^2) )e^{ik⋅x} + (ak/(2√2π|k|^2) + i bk/(2√2π^2|k|^2) )e^{-ik⋅x}
    
    # the basis is ordered as 0,1,...,nx/2-1, -nx/2, -nx/2+1, ... -1
    
    Random.seed!(123);
    θ0 = rand(Normal(0, params.σ), Nθ)
    
    ω0 = Initial_ω0_KL(mesh, θ0, seq_pairs)
    
    return ω0, θ0
    
end



# For fourier modes, consider the zero-mean real field
# u = ∑ f_k e^{ik⋅x}, we have  f_{k} = conj(f_{-k})
# we can also use cos(k⋅x)/2π^2 and sin(k⋅x)/2π^2 as basis (L2 norm is 1)
# with (kx + ky > 0 || (kx + ky == 0 && kx > 0)) 
# And these fourier modes are sorted by 1/|k|^2 and the first trunc_KL modes 
# are computed in terms of (kx, ky) pair
function Compute_Seq_Pairs(trunc_KL::Int64)
    seq_pairs = zeros(Int64, trunc_KL, 2)
    trunc_Nx = trunc(Int64, sqrt(2*trunc_KL)) + 1
    
    seq_pairs = zeros(Int64, div(((2trunc_Nx+1)^2 - 1), 2), 2)
    seq_pairs_mag = zeros(Int64, div(((2trunc_Nx+1)^2 - 1),2))
    
    seq_pairs_i = 0
    for kx = -trunc_Nx:trunc_Nx
        for ky = -trunc_Nx:trunc_Nx
            if Mode_Helper(kx, ky)
                
                seq_pairs_i += 1
                seq_pairs[seq_pairs_i, :] .= kx, ky
                seq_pairs_mag[seq_pairs_i] = kx^2 + ky^2
            end
        end
    end
    
    seq_pairs = seq_pairs[sortperm(seq_pairs_mag), :]
    return seq_pairs[1:trunc_KL, :]
end



# we can also use cos(k⋅x)/2π^2 and sin(k⋅x)/2π^2 as basis (L2 norm is 1) for zero-mean real fields
# with (kx + ky > 0 || (kx + ky == 0 && kx > 0)) 
# θ = [ak;bk]
# ω0 = ∑_k ak cos(k⋅x)/2π^2|k|^2 + bk sin(k⋅x)/2π^2|k|^2
function Initial_ω0_KL(mesh::Spectral_Mesh, θ::Array{Float64,1}, seq_pairs::Array{Int64,2})
    # consider C = -Δ^{-2}
    # C u = λ u  => -u = λ Δ u
    # u = e^{ik⋅x}, λ Δ^2 u = -λ |k|^4 u = - u => λ = 1/|k|^4
    # basis are cos(k⋅x)/2π^2 and sin(k⋅x)/2π^2, k!=(0,0), zero mean.
    # ω = ∑ ak cos(k⋅x)/2π^2|k|^2 + ∑ bk sin(k⋅x)/2π^2|k|^2,    &  kx + ky > 0 or (kx + ky = 0 and kx > 0) 
    #   = ∑ (ak/(4π^2|k|^2) - i bk/(4π^2|k|^2) )e^{ik⋅x} + (ak/(4π^2|k|^2) + i bk/(4π^2|k|^2) )e^{-ik⋅x}
    
    
    
    nx, ny = mesh.nx, mesh.ny
    kxx, kyy = mesh.kxx, mesh.kyy
    
    ω0_hat = zeros(ComplexF64, nx, ny)
    ω0 = zeros(Float64, nx, ny)
    abk = reshape(θ, Int64(length(θ)/2), 2)
    
    trunc_KL = size(abk,1)
    for i = 1:trunc_KL
        kx, ky = seq_pairs[i,:]
        if Mode_Helper(kx, ky)
            ak, bk = abk[i,:]
            ix = (kx >= 0 ? kx + 1 : nx + kx + 1) 
            iy = (ky >= 0 ? ky + 1 : ny + ky + 1) 
            
            ω0_hat[ix, iy] = (ak - bk*im)/(2*sqrt(2)*pi*(kx^2+ky^2))
            
            # 1 => 1, i => n-i+2
            ω0_hat[(ix==1 ? 1 : nx-ix+2), (iy==1 ? 1 : ny-iy+2)] = (ak + bk*im)/(2*sqrt(2)*pi*(kx^2+ky^2))
        end
    end
    ω0_hat .*= mesh.alias_filter
    Trans_Spectral_To_Grid!(mesh, nx*ny * ω0_hat, ω0)
    
    
    # ω0_test = zeros(Float64, nx, ny)
    # X, Y = zeros(Float64, nx, ny), zeros(Float64, nx, ny)
    # Δx, Δy = mesh.Δx, mesh.Δy
    # for ix = 1:nx
    #     for iy = 1:ny
    #         X[ix, iy] = (ix-1)*Δx
    #         Y[ix, iy] = (iy-1)*Δy
    #     end
    # end
    
    # for i = 1:trunc_KL
    #     kx, ky = seq_pairs[i,:]
    #     ak, bk = abk[i,:]
    #     @info kx, ky, nx, ny
    
    #     @assert((abs(kx) < nx/3 && abs(ky) < ny/3) && (kx + ky > 0 || (kx + ky == 0 && kx > 0))) 
    #     ω0_test .+= (ak * cos.(kx*X + ky*Y) + bk * sin.(kx*X + ky*Y))/(2*pi^2*(kx^2 + ky^2))
    # end
    # @info  "Error", norm(ω0 - ω0_test), norm(ω0) , norm(ω0_test)
    # error("Stop")
    
    
    ω0
end


# we can also use cos(k⋅x)/2π^2 and sin(k⋅x)/2π^2 as basis (L2 norm is 1) for zero-mean real fields
# with (kx + ky > 0 || (kx + ky == 0 && kx > 0)) 
# θ = [ak;bk]
# ω0 = ∑_k ak cos(k⋅x)/2π^2|k|^2 + bk sin(k⋅x)/2π^2|k|^2
function Initial_ω0_cov_KL(mesh::Spectral_Mesh, θ::Array{Float64,1}, cov_θ::Array{Float64,2}, seq_pairs::Array{Int64,2})
    # consider C = -Δ^{-2}
    # C u = λ u  => -u = λ Δ u
    # u = e^{ik⋅x}, λ Δ^2 u = -λ |k|^4 u = - u => λ = 1/|k|^4
    # basis are cos(k⋅x)/2π^2 and sin(k⋅x)/2π^2, k!=(0,0), zero mean.
    # ω = ∑ ak cos(k⋅x)/2π^2|k|^2 + ∑ bk sin(k⋅x)/2π^2|k|^2,    &  kx + ky > 0 or (kx + ky = 0 and kx > 0) 
    #   = ∑ (ak/(4π^2|k|^2) - i bk/(4π^2|k|^2) )e^{ik⋅x} + (ak/(4π^2|k|^2) + i bk/(4π^2|k|^2) )e^{-ik⋅x}
    
    nx, ny = mesh.nx, mesh.ny
    
    # ω0_test = zeros(Float64, nx, ny)
    X, Y = zeros(Float64, nx, ny), zeros(Float64, nx, ny)
    Δx, Δy = mesh.Δx, mesh.Δy
    for ix = 1:nx
        for iy = 1:ny
            X[ix, iy] = (ix-1)*Δx
            Y[ix, iy] = (iy-1)*Δy
        end
    end
    
    # ω0_1d =  C θ
    trunc_KL = Int64(length(θ)/2)
    C = zeros(nx*ny , length(θ))
    for i = 1:trunc_KL
        kx, ky = seq_pairs[i,:]
        
        @assert((abs(kx) < nx/3 && abs(ky) < ny/3) && (kx + ky > 0 || (kx + ky == 0 && kx > 0))) 
        C[:, i]             .= cos.(kx*X + ky*Y)[:] /(sqrt(2)*pi*(kx^2 + ky^2))
        C[:, i + trunc_KL]  .= sin.(kx*X + ky*Y)[:] /(sqrt(2)*pi*(kx^2 + ky^2))
    end
    ω0 = reshape(C * θ, nx, ny)
    
    cov_ω0 = C*cov_θ*C'
    
    std_ω0 = reshape(sqrt.(diag(cov_ω0)), nx, ny)
    
    ω0, std_ω0
end


# generate truth (all Fourier modes) and observations
# observation are at frame 0, Δd_t, 2Δd_t, ... nt
# with sparse points at Array(1:Δd_x:nx) × Array(1:Δd_y:ny)
# add N(0, std=εy) to y,  here ε = noise_level
function Generate_Data(params::Params, seq_pairs, noise_level::Float64 = -1.0, save_file_name::String = "None", Nθ::Int64=100)
    
    
    ν = params.ν
    ub, vb = params.ub, params.vb
    nx, ny = params.nx, params.ny
    Lx, Ly = params.Lx, params.Ly
    nt, T = params.nt, params.T
    method = params.method 
    Δd_x, Δd_y, Δd_t = params.Δd_x, params.Δd_y, params.Δd_t
    
    
    mesh = Spectral_Mesh(nx, ny, Lx, Ly)
    ω0, θ0 = Initial_ω0_KL(mesh, Nθ, seq_pairs, params)
    
    data = Foward_Helper(params, ω0, save_file_name)
    
    if noise_level > 0.0
        Random.seed!(666);
        for i = 1:length(data)
            
            noise = rand(Normal(0, noise_level*abs(data[i])))
            
            data[i] += noise
        end
    end
    
    
    return ω0, θ0, data
    
end


# generate truth (all Fourier modes) and observations
# observation are at frame 0, Δd_t, 2Δd_t, ... nt
# with sparse points at Array(1:Δd_x:nx) × Array(1:Δd_y:ny)
# add N(0, std=εy) to y,  here ε = noise_level
function Generate_Data_Noiseless(params::Params, seq_pairs, save_file_name::String = "None", Nθ::Int64=100)
    
    
    ν = params.ν
    ub, vb = params.ub, params.vb
    nx, ny = params.nx, params.ny
    Lx, Ly = params.Lx, params.Ly
    nt, T = params.nt, params.T
    method = params.method 
    Δd_x, Δd_y, Δd_t = params.Δd_x, params.Δd_y, params.Δd_t
    
    
    mesh = Spectral_Mesh(nx, ny, Lx, Ly)
    ω0, θ0 = Initial_ω0_KL(mesh, Nθ, seq_pairs, params)
    
    data = Foward_Helper(params, ω0, save_file_name)
    
    
    return ω0, θ0, data
    
end


# generate truth (all Fourier modes) and observations
# observation are at frame 0, Δd_t, 2Δd_t, ... nt
# with sparse points at Array(1:Δd_x:nx) × Array(1:Δd_y:ny)
function Foward_Helper(params::Params, ω0::Array{Float64,2}, save_file_name::String="None")
    
    
    ν = params.ν
    ub, vb = params.ub, params.vb
    nx, ny = params.nx, params.ny
    Lx, Ly = params.Lx, params.Ly
    nt, T = params.nt, params.T
    method = params.method 
    Δd_x, Δd_y, Δd_t = params.Δd_x, params.Δd_y, params.Δd_t
    
    
    mesh = Spectral_Mesh(nx, ny, Lx, Ly)
    
    fx,fy = Force(mesh)
    
    solver = SpectralNS_Solver(mesh, ν, fx, fy, ω0, ub, vb)  
    Δt_max = Stable_Δt(mesh, ν, solver.u, solver.v)
    
    Δt = T/nt
    # initialize observation 
    
    d_x = Array(1:Δd_x:nx)
    d_y = Array(1:Δd_y:ny)
    d_t = Array(Δd_t:Δd_t:nt)
    
    data = zeros(Float64, size(d_x,1), size(d_y,1), size(d_t,1))
    
    plot_range = 5
    #data[:,:,1] = ω0[d_x, d_y]
    if save_file_name != "None"
        Visual(mesh, ω0, "ω", save_file_name*"0.pdf", -plot_range, plot_range)
    end
    
    for i = 1:nt
        Solve!(solver, Δt, method)
        if i%Δd_t == 0
            Update_Grid_Vars!(solver)
            if save_file_name != "None"
                Visual_Obs(mesh, solver.ω, Δd_x, Δd_y, "ω", save_file_name*string(i)*".pdf", -plot_range, plot_range)
            end
            data[:, :, Int64(i/Δd_t)] = solver.ω[d_x, d_y]
        end
    end
    
    return data[:]
end


# project  ω0 on the first nθ0 dominant modes
# namely, ∑ ak cos(k⋅x) + bk sin(k⋅x) with least square fitting
function Construct_θ0(params::Params, ω0::Array{Float64, 2}, nθ0::Int64, seq_pairs::Array{Int64,2})
    @assert(nθ0%2 == 0)
    nθ = 2*size(seq_pairs,1)
    abk0 = zeros(Float64, Int64(nθ/2), 2)
    
    ν = params.ν
    nx, ny = params.nx, params.ny
    Lx, Ly = params.Lx, params.Ly
    nt, T = params.nt, params.T
    method = params.method 
    Δd_x, Δd_y, Δd_t = params.Δd_x, params.Δd_y, params.Δd_t
    
    
    mesh = Spectral_Mesh(nx, ny, Lx, Ly)
    
    Δt = T/nt
    
    # initialize observation 
    
    d_x = Array(1:Δd_x:nx)
    d_y = Array(1:Δd_y:ny)
    
    data0 = ω0[d_x, d_y]
    
    
    ndata = length(data0)
    # fit the dominant modes from data0 by solving A x = data0
    # ω = ∑ ak cos(k⋅x)/2π^2|k|^2 + ∑ bk sin(k⋅x)/2π^2|k|^2,    &  kx + ky > 0 or (kx + ky = 0 and kx > 0) 
    A = zeros(Float64, ndata, nθ0)
    X, Y = zeros(Float64, nx, ny), zeros(Float64, nx, ny)
    Δx, Δy = mesh.Δx, mesh.Δy
    for ix = 1:nx
        for iy = 1:ny
            X[ix, iy] = (ix-1)*Δx
            Y[ix, iy] = (iy-1)*Δy
        end
    end
    X, Y = X[d_x, d_y], Y[d_x, d_y]
    
    for i = 1:Int64(nθ0/2)
        kx, ky = seq_pairs[i,:]
        
        
        A[:,i]              = (cos.(kx*X + ky*Y)/(2*pi^2*(kx^2 + ky^2)))[:]
        A[:,i+Int64(nθ0/2)] = (sin.(kx*X + ky*Y)/(2*pi^2*(kx^2 + ky^2)))[:]
        
    end
    x = A\data0[:]
    
    
    
    abk0[1:Int64(nθ0/2), :] = reshape(x, Int64(nθ0/2), 2)
    θ0 = abk0[:]
    
    ω0_fit = Initial_ω0_KL(mesh, θ0, seq_pairs)
    
    Visual(mesh, ω0_fit, "ω", "vor0.fitted.png")
    
    return θ0
    
end



function RandomInit_Main(θ::Array{Float64,1}, seq_pairs::Array{Int64,2}, params::Params)
    # θ = [ak; bk] and abk = [ak, bk], we have θ = abk[:] and abk = reshape(θ, nθ/2, 2)
    
    ν = params.ν
    ub, vb = params.ub, params.vb
    nx, ny = params.nx, params.ny
    Lx, Ly = params.Lx, params.Ly
    nt, T = params.nt, params.T
    method = params.method 
    Δd_x, Δd_y, Δd_t = params.Δd_x, params.Δd_y, params.Δd_t
    
    
    mesh = Spectral_Mesh(nx, ny, Lx, Ly)
    ω0 = Initial_ω0_KL(mesh, θ, seq_pairs)   
    
    data = Foward_Helper(params, ω0)
    
    return data
end

function Params()
    ν = 1.0e-2   # viscosity
    nx, ny = 128, 128  # resolution in x
    
    #ub, vb = 1.0, 1.0
    ub, vb = 2*pi, 2*pi
    # resolution in y
    Lx, Ly = 2*pi, 2*pi
    method="Crank-Nicolson" # RK4 or Crank-Nicolson
    nt = 2500;    # time step
    T = 0.5;      # final time
    #observation
    Δd_x, Δd_y, Δd_t = 16, 16, 1250
    
    
    n_data = (div(nx-1,Δd_x)+1)*(div(ny-1,Δd_y)+1)*(div(nt, Δd_t))
    #parameter standard deviation
    σ = sqrt(2)*pi
    Params(ν, ub, vb, nx, ny, Lx, Ly, method, nt, T, Δd_x, Δd_y, Δd_t, n_data, σ)
end

function Force(mesh::Spectral_Mesh)
    # F = [-∂Φ/∂y, ∂Φ/∂x] and Φ = cos((5x + 5y))
    nx, ny = mesh.nx, mesh.ny
    Δx, Δy, xx, yy = mesh.Δx, mesh.Δy, mesh.xx, mesh.yy
    fx, fy = zeros(Float64, nx, ny), zeros(Float64, nx, ny)
    for i = 1:nx
        for j = 1:ny
            x, y = xx[i], yy[j]
            fx[i,j] =  2sin(4*y) 
            fy[i,j] =  0
        end
    end
    
    return fx, fy
end


if abspath(PROGRAM_FILE) == @__FILE__
    phys_params = Params()
    # data
    ω0, t_mean =  Generate_Data(phys_params)
    # learn 100 modes
    na = 50        
    seq_pairs = Compute_Seq_Pairs(na)
    
    nx, ny, Δd_x, Δd_y = phys_params.nx, phys_params.ny, phys_params.Δd_x, phys_params.Δd_y
    
    ndata0 = (div(nx-1,Δd_x)+1)*(div(ny-1,Δd_y)+1)
    θ0 = Construct_θ0(phys_params, ω0, div(ndata0,2), seq_pairs)
end