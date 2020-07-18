include("Spectral_Mesh.jl")

mutable struct SpectralNS_Solver
    mesh::Spectral_Mesh

    ν::Float64

    f::Array{Float64, 2}
    curl_f_hat::Array{ComplexF64, 2}

    ω_hat::Array{ComplexF64, 2}
    u_hat::Array{ComplexF64, 2}
    v_hat::Array{ComplexF64, 2}

    ω::Array{Float64, 2}
    u::Array{Float64, 2}
    v::Array{Float64, 2}

    Δω_hat::Array{ComplexF64, 2}
    δω_hat::Array{ComplexF64, 2}
end


#initialize with velocity field todo check compressibility 
function SpectralNS_Solver(mesh::Spectral_Mesh, ν::Float64, f::Array{Float64, 2}, u0::Array{Float64, 2}, v0::Array{Float64, 2})    
    nx, ny = mesh.nx, mesh.ny

    curl_f_hat = zeros(ComplexF64, nx, ny)
    Apply_Curl!(mesh, f, curl_f_hat) 

    u .= u0
    v .= v0

    u_hat = zeros(ComplexF64, nx, ny)
    v_hat = zeros(ComplexF64, nx, ny)
    Trans_Grid_To_Spectral!(mesh, u, u_hat)
    Trans_Grid_To_Spectral!(mesh, v, v_hat)



    ω_hat = zeros(ComplexF64, nx, ny)
    ω .= zeros(Float64, nx, ny)
    Vor_From_UV_Spectral!(mesh, u_hat, v_hat, ω_hat, ω)
    
    δω_hat = zeros(ComplexF64, nx, ny)
    Δω_hat = zeros(ComplexF64, nx, ny)

    SpectralNS_Solver(mesh, ν, f, curl_f_hat, ω_hat, u_hat, v_hat, ω, u, v, Δω_hat, δω_hat)
end

#initialize with vorticity field
function SpectralNS_Solver(mesh::Spectral_Mesh, ν::Float64, f::Array{Float64, 2}, ω0::Array{Float64, 2})    
    nx, ny = mesh.nx, mesh.ny

    curl_f_hat = zeros(ComplexF64, nx, ny)
    Apply_Curl!(mesh, f, curl_f_hat) 

    ω = zeros(Float64, nx, ny)
    ω .= ω0
    ω_hat = zeros(ComplexF64, nx, ny)
    Trans_Grid_To_Spectral!(mesh, ω, ω_hat)

    

    u_hat = zeros(ComplexF64, nx, ny)
    v_hat = zeros(ComplexF64, nx, ny)
    UV_Spectral_From_Vor!(mesh, ω_hat, u_hat, v_hat)

    u = zeros(Float64, nx, ny)
    v = zeros(Float64, nx, ny)
    Trans_Spectral_To_Grid!(mesh, u_hat, u)
    Trans_Spectral_To_Grid!(mesh, v_hat, v)

    δω_hat = zeros(ComplexF64, nx, ny)
    Δω_hat = zeros(ComplexF64, nx, ny)

    SpectralNS_Solver(mesh, ν, f, curl_f_hat, ω_hat, u_hat, v_hat, ω, u, v, Δω_hat, δω_hat)
end


function Update_Force!(self::SpectralNS_Solver, f::Array{Float64, 2})
    mesh = self.mesh
    
    self.f .= f
   
    Apply_Curl!(mesh, f, self.curl_f_hat) 
end


function Explicit_Residual!(self::SpectralNS_Solver, ω_hat::Array{ComplexF64, 2}, δω_hat::Array{ComplexF64, 2})
    # ∂ω/∂t - (U⋅∇)ω - νΔω  = ∇×f
    # ∂ω_hat/∂t = -FFT[(U⋅∇)ω] + ν(- ((2πkx/Lx)² + (2πky/Ly)²) )ω_hat  + curl_f_hat
    
    mesh = self.mesh
    δω_hat .= self.curl_f_hat
    Add_Horizontal_Advection!(mesh, ω_hat, δω_hat)

    Δω_hat = self.Δω_hat

    Apply_Laplacian!(mesh, ω_hat, Δω_hat)

    δω_hat .+= self.ν * Δω_hat

    
end

"""
Navier stokes equation with double periodicity
U = (u,v) and p denote velocity and pressure 
∂U/∂t + U⋅∇U - νΔU + ∇p = f
∇⋅U = 0

Let ω = ∇×U, we have the vorticity equation
∂ω/∂t + (U⋅∇)ω - νΔω  = ∇×f

ω = 1/(nxny)∑_{kx, ky}  ω_hat[kx,ky]  e^{i (2π/Lx kx x + 2π/Ly ky y)}

∂ω_hat/∂t + FFT[(U⋅∇)ω] + ν( (2πkx/Lx)² + (2πky/Ly)² )ω_hat  = 2π/Lx kx fy_hat - 2π/Ly ky fx_hat


The domain is [0,Lx][0,Ly]
"""
function Semi_Implicit_Residual!(self::SpectralNS_Solver, ω_hat::Array{ComplexF64, 2}, Δt::Float64, δω_hat::Array{ComplexF64, 2})
    # Also the Crank-Nicolson
    # (ω_hat(n+1) - ω_hat(n))/Δt  = FFT[(U⋅∇)ω] + ν(- ((2πkx/Lx)² + (2πky/Ly)²) )(ω_hat(n)+ω_hat(n+1))/2  + curl_f_hat

    # [1 + νΔt(((2πkx/Lx)² + (2πky/Ly)²))/2] (ω_hat(n+1)-ω_hat(n))/Δt = FFT[(U⋅∇)ω] + ν(-((2πkx/Lx)² + (2πky/Ly)²))ω_hat(n)  + curl_f_hat

    mesh = self.mesh
    δω_hat .= self.curl_f_hat
    Add_Horizontal_Advection!(mesh, ω_hat, δω_hat)
    Δω_hat = self.Δω_hat
    Apply_Laplacian!(mesh, ω_hat, Δω_hat)

    δω_hat .+= self.ν * Δω_hat

    δω_hat ./= (1.0 .- 0.5*self.ν*Δt * mesh.laplacian_eigs)
end


function Solve!(self::SpectralNS_Solver, Δt::Float64)
    ω_hat, δω_hat = self.ω_hat, self.δω_hat
    Semi_Implicit_Residual!(self, ω_hat, Δt, δω_hat)
    ω_hat .+= Δt*δω_hat
end




