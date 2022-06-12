using LinearAlgebra
using Random
using Distributions
using FFTW
using PyPlot

mutable struct Spectral_Mesh
    N_x::Int64
    
    Lx::Float64
    Δx::Float64
    xx::Array{Float64, 1}
  
    kxx::Array{Float64, 1}
    alias_filter::Array{Float64, 1}
    d_x::Array{ComplexF64, 1}
    laplacian_eigs::Array{Float64, 1}
    
    # container
    u::Array{Float64, 1}
    u_x::Array{Float64, 1}
    u_hat::Array{ComplexF64, 1}
    u_x_hat::Array{ComplexF64, 1}
end




function Spectral_Mesh(N_x::Int64,  Lx::Float64)
    @assert(N_x%2 == 0)
    
    Δx = Lx/N_x
    xx = LinRange(0, Lx-Δx, N_x)
    

    kxx, alias_filter = Spectral_Init(N_x)
    d_x = Apply_Gradient_Init(N_x, Lx, kxx) 
    laplacian_eigs = Apply_Laplacian_Init(N_x, Lx, kxx) 
    
    
    # container
    u= zeros(Float64, N_x)
    u_x= zeros(Float64, N_x)
    u_hat= zeros(ComplexF64, N_x)
    u_x_hat= zeros(ComplexF64, N_x)
    
    Spectral_Mesh(N_x, Lx, Δx, xx, kxx, alias_filter, 
                  d_x, laplacian_eigs,
                  u, u_x, u_hat,  u_x_hat)
end

"""
Compute mode numbers kxx and kyy
kxx = [0,1,...,[N_x/2]-1, -[N_x/2], -[N_x/2]+1, ... -1]   
and alias filter based O
"""
function Spectral_Init(N_x::Int64)
    kxx = mod.((1:N_x) .- ceil(Int64, N_x/2+1),N_x) .- floor(Int64, N_x/2)
    alias_filter = zeros(Float64, N_x)
    for i = 1:N_x
        if (abs(kxx[i]) < N_x/3) 
            alias_filter[i] = 1.0
        end
    end
    return kxx, alias_filter
end




function Trans_Spectral_To_Grid!(mesh::Spectral_Mesh, u_hat::Array{ComplexF64,1},  u::Array{Float64,1})
    """
    All indices start with 0
    
    K = {(kx, ky) | kx ∈ 0,1,...,[N_x/2]-1, -[N_x/2], -[N_x/2]+1, ... -1   ,   ky ∈ 0,1,...,[N_y/2]-1, -[N_y/2], -[N_y/2]+1, ... -1
    x ∈ 0, Δx, 2Δx, ..., Lx - Δx   ,   y ∈ 0, Δy, 2Δy, ..., Ly - Δy  here Δx, Δy = Lx/N_x, Ly/N_y
    
    P(x, y) =  1/(nxny)∑_{kx, ky}  F[kx,ky]  e^{i (2π/Lx kx x + 2π/Ly ky y)}
    
    P[jx, jy] = P(jxΔx, jyΔy) =  1/(nxny)∑_{kx, ky}  F[kx,ky]  e^{i (2π kx jx/N_x + 2π ky jy/N_y)}
    
    
    @test
    u_hat = [1 2 3 4; 1.1 2.2 1.3 2.4; 2.1 3.2 4.1 1.2]
    N_x, N_y = size(u_hat)
    u = zeros(ComplexF64, N_x, N_y)
    for jx = 0:N_x-1
        for jy = 0:N_y-1
            for kx = 0:N_x-1
                for ky = 0:N_y-1
                    u[jx+1, jy+1] += u_hat[kx+1,ky+1] *  exp((2π*kx*jx/N_x + 2.0*π*ky*jy/N_y)*im) /(N_x*N_y)
                end
            end
        end
    end
    ufft = ifft(u_hat)
    @info u - ufft
    """
    
    u .= real(ifft(u_hat)) #fourier for the first dimension
    
end



function Trans_Grid_To_Spectral!(mesh::Spectral_Mesh, u::Array{Float64,1}, u_hat::Array{ComplexF64,1})
    
    """
    K = {(kx, ky) | kx ∈ 0,1,...,N_x/2-1, -N_x/2, -N_x/2+1, ... -1   ,   ky ∈ 0,1,...,N_y/2-1, -N_y/2, -N_y/2+1, ... -1}
    
    P(x, y) = 1/(nx⋅ny) ∑_{kx, ky}  F[kx,ky]  e^{i (2π/Lx kx x + 2π/Ly ky y)}
    
    F[kx, ky] = (nx⋅ny)/(Lx⋅Ly) ∫ P(x,y)  e^{-i (2π/Lx kx x + 2π/Ly ky y)}
              = (nx⋅ny)/(Lx⋅Ly) ∑ P[jx,jy]  e^{-i (2π kx jx/N_x + 2π ky jy/N_y)} ΔxΔy
              = ∑ P[jx,jy]  e^{-i (2π kx jx/N_x + 2π ky jy/N_y)}
    
    @test
    u = [1 2 3 4; 1.1 2.2 1.3 2.4; 2.1 3.2 4.1 1.2]
    N_x, N_y = size(u)
    u_hat = zeros(ComplexF64, N_x, N_y)
    for jx = 0:N_x-1
        for jy = 0:N_y-1
            for kx = 0:N_x-1
                for ky = 0:N_y-1
                    u_hat[jx+1, jy+1] += u[kx+1,ky+1] *  exp(-(2π*kx*jx/N_x + 2.0*π*ky*jy/N_y)*im)
                end
            end
        end
    end
    u_hat2 = fft(u)
    @info u_hat - u_hat2
    """
    
    
    u_hat .= mesh.alias_filter.* fft(u)
    
    
end

function Apply_Laplacian_Init(N_x::Int64, Lx::Float64, kxx::Array{Int64, 1}) 
    """
    See Apply_Laplacian!
    """
    laplacian_eig = zeros(Float64, N_x)
    for i = 1:N_x
        kx = kxx[i]
        laplacian_eig[i] = -(2*pi*kx/Lx)^2 
    end
    return laplacian_eig
end

function Apply_Laplacian!(mesh::Spectral_Mesh, u_hat::Array{ComplexF64,1}, Δu_hat::Array{ComplexF64,1}; order=1) 
    """
    Δ (ω_hat[kx,ky]  e^{i 2π/Lx kx x}) = -(2πkx/Lx)² ω_hat
    """
    eig = mesh.laplacian_eigs.^order
    Δu_hat .= eig .* u_hat
end

function Apply_Gradient_Init(N_x::Int64,  Lx::Float64,  kxx::Array{Int64, 1}) 
    """
    ∂f/∂x_hat = alpha_x f_hat
    """
    d_x = zeros(ComplexF64, N_x)
    
    for i = 1:N_x
        kx = kxx[i]
        d_x[i] = (2*pi/Lx * kx)*im
    end

    return d_x
end


"""
Compute gradients u_x_hat, ω_y, from spectral input u_hat
u_x_hat = d_x u_hat
"""
function Apply_Gradient!(mesh::Spectral_Mesh, u_hat::Array{ComplexF64,1}, u_x_hat::Array{ComplexF64,1})
    d_x = mesh.d_x
    
    u_x_hat .= d_x .* u_hat
end

"""
δu_hat = hat { u ∂u } = hat { 0.5 ∂(uu) } = hat {  ∂(uu/0.5) } = d_x hat{uu/0.5}
"""
function Compute_Horizontal_Advection!(mesh::Spectral_Mesh, u_hat::Array{ComplexF64,1}, δu_hat::Array{ComplexF64, 1})
    u = mesh.u
    # contanier
    u2, u2_hat = mesh.u_x, mesh.u_x_hat

    Trans_Spectral_To_Grid!(mesh, u_hat, u)
    # Apply_Gradient!(mesh, u_hat, u_x)
    # Trans_Grid_To_Spectral!(mesh, u.*u_x,  δu_hat)
    u2 .= u.*u/2.0
    Trans_Grid_To_Spectral!(mesh, u2,  u2_hat)
    Apply_Gradient!(mesh, u2_hat, δu_hat)
    
end



function Visual(mesh::Spectral_Mesh, u::Array{Float64,1}, var_name::String,
    save_file_name::String="None", vmin=nothing, vmax=nothing)
    
    N_x, N_y = mesh.N_x, mesh.N_y
    xx, yy = mesh.xx, mesh.yy
    X,Y = repeat(xx, 1, N_y), repeat(yy, 1, N_x)'
    
    figure()
    pcolormesh(X, Y, u, shading= "gouraud", cmap="jet", vmin=vmin, vmax =vmax)
    xlabel("X")
    ylabel("Y")
    colorbar()
    
    if save_file_name != "None"
        tight_layout()
        savefig(save_file_name)
    end
end




"""
Generalized Kuramoto-Sivashinksy equation with periodic boundary condition
 
∂u/∂t + u∂u + ν₂∂²u + ν₄∂⁴u = 0
u = 1/(N_x)∑_{kx}  u_hat[kx,ky]  e^{i 2π kx x/Lx }
∂u_hat/∂t + FFT[(u⋅∇)u] + [ν₂(i2π kx/Lx)² + ν₄(i2π ky/Ly)⁴]u_hat  = 0

The domain is [0,Lx]
"""
mutable struct Spectral_KS_Solver
    mesh::Spectral_Mesh

    ν₂::Float64
    ν₄::Float64

    u::Array{Float64, 1}
    u_hat::Array{ComplexF64, 1}
    u_hat_old::Array{ComplexF64, 1}

    Δu_hat::Array{ComplexF64, 1}
    δu_hat::Array{ComplexF64, 1}

    k1::Array{ComplexF64, 1}
    k2::Array{ComplexF64, 1}
    k3::Array{ComplexF64, 1}
    k4::Array{ComplexF64, 1}
end


# constructor of the Spectral_KS_Solver
#
# There are two forcing intialization approaches
# * initialize fx and fy components, user shoul make sure they are zero-mean and periodic
# * initialize ∇×f, curl_f
#
# There are velocity(vorticity) intialization approaches
# * initialize u0 and v0 components, user should make sure they are incompressible div ⋅ (u0 , v0) = 0
# * initialize ω0 and mean backgroud velocity ub and vb
#
function Spectral_KS_Solver(mesh::Spectral_Mesh, u0::Array{Float64, 1}, ν₂::Float64 = 1.0, ν₄::Float64 = 1.0)    
    N_x = mesh.N_x
    
    u = zeros(Float64, N_x)
    u_hat = zeros(ComplexF64, N_x)
    u_hat_old = zeros(ComplexF64, N_x)
    
    
    # initialization
    u .= u0
    Trans_Grid_To_Spectral!(mesh, u, u_hat)
    u_hat_old .= u_hat


    δu_hat = zeros(ComplexF64, N_x)
    Δu_hat = zeros(ComplexF64, N_x)
    
    k1 = zeros(ComplexF64, N_x)
    k2 = zeros(ComplexF64, N_x)
    k3 = zeros(ComplexF64, N_x)
    k4 = zeros(ComplexF64, N_x)
    
    Spectral_KS_Solver(mesh, ν₂, ν₄, u, u_hat, u_hat_old, Δu_hat, δu_hat, k1, k2, k3, k4)
end


# # compute convective and diffusive stable time step
# # TODO 
# function Stable_Δt(mesh::Spectral_Mesh, u::Array{Float64,1}, ν₂::Float64, ν₄::Float64)
#     Δx = mesh.Δx
#     u_max = maximum(abs.(u))

#     Δt = min(Δx/u_max, Δx^2/(2*ν₂), (Δx^2/(2*ν₄))^2)

#     return Δt
# end


# ∂u/∂t + u∂u + ν₂∂²u + ν₄∂⁴u = 0
# ∂u_hat/∂t =  - FFT[(u⋅∇)u] - [ν₂(i2π kx/Lx)² + ν₄(i2π ky/Ly)⁴]u_hat  = 0
# Compute the right hand-side
function Explicit_Residual!(self::Spectral_KS_Solver, u_hat::Array{ComplexF64, 1}, δu_hat::Array{ComplexF64, 1})
    mesh = self.mesh

    Compute_Horizontal_Advection!(mesh, u_hat, δu_hat)

    δu_hat .*= -1.0 

    Δu_hat = self.Δu_hat
    Apply_Laplacian!(mesh, u_hat, Δu_hat; order = 1)
    δu_hat .-= self.ν₂ * Δu_hat
    Apply_Laplacian!(mesh, u_hat, Δu_hat; order = 2)
    @info norm(real.(u_hat)), norm(imag.(u_hat))
    δu_hat .-= self.ν₄ * Δu_hat
end

# Also the Crank-Nicolson
# ∂u/∂t + u∂u + ν₂∂²u + ν₄∂⁴u = 0
# ∂u_hat/∂t =  - FFT[(u⋅∇)u] - [ν₂(i2π kx/Lx)² + ν₄(i2π ky/Ly)⁴]u_hat  = 0
# [1 - [ν₂(i2π kx/Lx)² + ν₄(i2π ky/Ly)⁴]/2] (u_hat(n+1)-u_hat(n))/Δt = -FFT[(u⋅∇)u] - [ν₂(i2π kx/Lx)² + ν₄(i2π ky/Ly)⁴]u_hat(n)/2 
# compute (ω_hat(n+1)-ω_hat(n))/Δt
function Semi_Implicit_Residual!(self::Spectral_KS_Solver, u_hat::Array{ComplexF64, 1}, u_hat_old::Array{ComplexF64, 1}, 
    Δt::Float64, δu_hat::Array{ComplexF64, 1})

    mesh = self.mesh

    Compute_Horizontal_Advection!(mesh, u_hat, δu_hat)
    Δu_hat = self.Δu_hat
    Compute_Horizontal_Advection!(mesh, u_hat_old, Δu_hat)

    δu_hat .= -(3.0/2.0*δu_hat - 0.5*Δu_hat) 

    Δu_hat = self.Δu_hat
    Apply_Laplacian!(mesh, u_hat, Δu_hat; order = 1)
    δu_hat .-= self.ν₂ * Δu_hat

    Apply_Laplacian!(mesh, u_hat, Δu_hat; order = 2)
    δu_hat .-= self.ν₄ * Δu_hat
    
    δu_hat ./= ( 1.0 .+ 0.5*Δt*(self.ν₂ * mesh.laplacian_eigs + self.ν₄ * mesh.laplacian_eigs.^2) )

end




function Solve!(self::Spectral_KS_Solver, Δt::Float64, method::String)
    u, u_hat, u_hat_old, δu_hat = self.u, self.u_hat, self.u_hat_old, self.δu_hat

    if method == "Crank-Nicolson-Adam-Bashforth"
        Semi_Implicit_Residual!(self, u_hat, u_hat_old, Δt, δu_hat)

    elseif method == "RK4"
        k1, k2, k3, k4 = self.k1, self.k2, self.k3, self.k4 
        Explicit_Residual!(self, u_hat,  k1)
        Explicit_Residual!(self, u_hat + Δt/2.0*k1, k2)
        Explicit_Residual!(self, u_hat + Δt/2.0*k2, k3)
        Explicit_Residual!(self, u_hat + Δt*k3, k4)
        δu_hat = (k1 + 2*k2 + 2*k3 + k4)/6.0

    end
    u_hat_old .= u_hat
    u_hat .+= Δt*δu_hat
    
    # clean
    Trans_Spectral_To_Grid!(mesh, u_hat, u)
    Trans_Grid_To_Spectral!(mesh, u, u_hat)
end



# ####################################

# ν₂=1.0;     # viscosity
# ν₄=1.0;     # viscosity
# N_x=1024;       # resolution in x
# Δt=0.05;    # time step
# T=1000.0;        # final time
# method="Crank-Nicolson-Adam-Bashforth"  # RK4 or Crank-Nicolson
# Lx = 100.0

# mesh = Spectral_Mesh(N_x, Lx)
# Δx, xx = mesh.Δx, mesh.xx

# u0 = cos.((2 * pi * xx) / Lx) + 0.1 * cos.((4 * pi * xx) / Lx)
# solver = Spectral_KS_Solver(mesh, u0, ν₂, ν₄)  

# N_t = Int64(T/Δt)
# save_every = 1
# u_all = zeros(div(N_t, save_every), N_x)

# for i = 1:N_t
#     Solve!(solver, Δt, method)
#     if i%save_every == 0
#         u_all[i, :] .= solver.u
#     end
# end

# fig, ax = PyPlot.subplots(nrows=1, ncols=1, sharex=false, sharey=false, figsize=(6,6))
# ax.plot(xx, u0, label="T = 0")
# ax.plot(xx, solver.u, label="T = "*string(T))
# ax.legend()



# T, X = repeat(Δt*Array(1:div(N_t, save_every)), 1, N_x), repeat(xx, 1, div(N_t, save_every))' 
# fig, ax = PyPlot.subplots(nrows=1, ncols=1, sharex=false, sharey=false, figsize=(6,6))
# cs = ax.contourf(X, T, u_all, cmap="viridis")
# fig.colorbar(cs)


# auto correlation


# spectral

