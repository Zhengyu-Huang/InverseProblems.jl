using FFTW
using PyPlot

mutable struct Spectral_Mesh
    nx::Int64
    ny::Int64
    
    Lx::Float64
    Ly::Float64

    Δx::Float64
    Δy::Float64
    
    xx::Array{Float64, 1}
    yy::Array{Float64, 1}

    kxx::Array{Float64, 1}
    kyy::Array{Float64, 1}
    alias_filter::Array{Float64, 2}
    
    d_x::Array{ComplexF64, 2}
    d_y::Array{ComplexF64, 2}
    
    alpha_x::Array{ComplexF64, 2}
    alpha_y::Array{ComplexF64, 2}
    
    laplacian_eigs::Array{Float64, 2}

    
    
    # container
    u::Array{Float64, 2}
    v::Array{Float64, 2}
    ω_x::Array{Float64, 2}
    ω_y::Array{Float64, 2}
    u_hat::Array{ComplexF64, 2}
    v_hat::Array{ComplexF64, 2}
    ω_x_hat::Array{ComplexF64, 2}
    ω_y_hat::Array{ComplexF64, 2}
end




function Spectral_Mesh(nx::Int64, ny::Int64, Lx::Float64, Ly::Float64)
    @assert(nx%2 == 0 && ny%2 == 0)
    
    Δx = Lx/nx
    xx = LinRange(0, Lx-Δx, nx)
    
    Δy = Ly/ny
    yy = LinRange(0, Ly-Δy, ny)

    kxx, kyy, alias_filter = Spectral_Init(nx, ny)
    
    
    d_x, d_y = Apply_Gradient_Init(nx, ny, Lx, Ly, kxx, kyy) 
    
    alpha_x, alpha_y = UV_Grid_Init(nx, ny, Lx, Ly, kxx, kyy) 
    
    laplacian_eigs = Apply_Laplacian_Init(nx, ny, Lx, Ly, kxx, kyy) 

    
    # container
    u= zeros(Float64, nx, ny)
    v= zeros(Float64, nx, ny)
    ω_x= zeros(Float64, nx, ny)
    ω_y= zeros(Float64, nx, ny)
    u_hat= zeros(ComplexF64, nx, ny)
    v_hat= zeros(ComplexF64, nx, ny)
    ω_x_hat= zeros(ComplexF64, nx, ny)
    ω_y_hat= zeros(ComplexF64, nx, ny)
    
    
    Spectral_Mesh(nx, ny, Lx, Ly, Δx, Δy, xx, yy, kxx, kyy, alias_filter, 
                  d_x, d_y, alpha_x, alpha_y, laplacian_eigs, 
                  u, v, ω_x, ω_y, u_hat, v_hat, ω_x_hat, ω_y_hat)
end


function Spectral_Init(nx::Int64, ny::Int64)
    kxx = mod.((1:nx) .- ceil(Int64, nx/2+1),nx) .- floor(Int64, nx/2)
    kyy = mod.((1:ny) .- ceil(Int64, ny/2+1),ny) .- floor(Int64, ny/2)

    alias_filter = ones(Float64, nx, ny)
    for i = 1:nx
        for j = 1:ny
            if kxx[i] >= 2*nx/3  || kyy[j] > 2*ny/3 
                alias_filter[i, j] = 0.0
            end
        end
    end
    return kxx, kyy, alias_filter
end




function Trans_Spectral_To_Grid!(mesh::Spectral_Mesh, u_hat::Array{ComplexF64,2},  u::Array{Float64,2})
    """
    All indices start with 0
    
    K = {(kx, ky) | kx ∈ 0,1,...,[nx/2]-1, -[nx/2], -[nx/2]+1, ... -1   ,   ky ∈ 0,1,...,[ny/2]-1, -[ny/2], -[ny/2]+1, ... -1
    x ∈ 0, Δx, 2Δx, ..., Lx - Δx   ,   y ∈ 0, Δy, 2Δy, ..., Ly - Δy  here Δx, Δy = Lx/nx, Ly/ny
    
    P(x, y) =  1/(nxny)∑_{kx, ky}  F[kx,ky]  e^{i (2π/Lx kx x + 2π/Ly ky y)}
    
    P[jx, jy] = P(jxΔx, jyΔy) =  1/(nxny)∑_{kx, ky}  F[kx,ky]  e^{i (2π kx jx/nx + 2π ky jy/ny)}
    
    
    @test
    u_hat = [1 2 3 4; 1.1 2.2 1.3 2.4; 2.1 3.2 4.1 1.2]
    nx, ny = size(u_hat)
    u = zeros(ComplexF64, nx, ny)
    for jx = 0:nx-1
        for jy = 0:ny-1
            for kx = 0:nx-1
                for ky = 0:ny-1
                    u[jx+1, jy+1] += u_hat[kx+1,ky+1] *  exp((2π*kx*jx/nx + 2.0*π*ky*jy/ny)*im) /(nx*ny)
                end
            end
        end
    end
    ufft = ifft(u_hat)
    @info u - ufft
    """
    
    u .= real(ifft(u_hat)) #fourier for the first dimension
    
end



function Trans_Grid_To_Spectral!(mesh::Spectral_Mesh, u::Array{Float64,2}, u_hat::Array{ComplexF64,2})
    
    """
    K = {(kx, ky) | kx ∈ 0,1,...,nx/2-1, -nx/2, -nx/2+1, ... -1   ,   ky ∈ 0,1,...,ny/2-1, -ny/2, -ny/2+1, ... -1}
    
    P(x, y) = 1/(nx⋅ny) ∑_{kx, ky}  F[kx,ky]  e^{i (2π/Lx kx x + 2π/Ly ky y)}
    
    F[kx, ky] = (nx⋅ny)/(Lx⋅Ly) ∫ P(x,y)  e^{-i (2π/Lx kx x + 2π/Ly ky y)}
              = (nx⋅ny)/(Lx⋅Ly) ∑ P[jx,jy]  e^{-i (2π kx jx/nx + 2π ky jy/ny)} ΔxΔy
              = ∑ P[jx,jy]  e^{-i (2π kx jx/nx + 2π ky jy/ny)}
    
    @test
    u = [1 2 3 4; 1.1 2.2 1.3 2.4; 2.1 3.2 4.1 1.2]
    nx, ny = size(u)
    u_hat = zeros(ComplexF64, nx, ny)
    for jx = 0:nx-1
        for jy = 0:ny-1
            for kx = 0:nx-1
                for ky = 0:ny-1
                    u_hat[jx+1, jy+1] += u[kx+1,ky+1] *  exp(-(2π*kx*jx/nx + 2.0*π*ky*jy/ny)*im)
                end
            end
        end
    end
    u_hat2 = fft(u)
    @info u_hat - u_hat2
    """
    
    
    u_hat .= fft(u)
    
    
end


function Apply_Laplacian_Init(nx::Int64, ny::Int64, Lx::Float64, Ly::Float64, kxx::Array{Int64, 1}, kyy::Array{Int64, 1}) 
    """
    See Compute_Laplacian!
    """
    laplacian_eig = zeros(Float64, nx, ny)
    
    for i = 1:nx
        for j = 1:ny
            kx, ky = kxx[i], kyy[j]
            laplacian_eig[i, j] = -((2*pi*kx/Lx)^2 + (2*pi*ky/Ly)^2)
        end
    end
    
    return laplacian_eig
    
end

function Apply_Laplacian!(mesh::Spectral_Mesh, u_hat::Array{ComplexF64,2}, Δu_hat::Array{ComplexF64,2}) 
    """
    Δ (ω_hat[kx,ky]  e^{i (2π/Lx kx x + 2π/Ly ky y)}) = -((2πkx/Lx)² + (2πky/Ly)² )ω_hat
    """
    eig = mesh.laplacian_eigs
    Δu_hat .= eig .* u_hat
end


function Apply_Gradient_Init(nx::Int64, ny::Int64, Lx::Float64, Ly::Float64, kxx::Array{Int64, 1}, kyy::Array{Int64, 1}) 
    """
    ∂f/∂x_hat = alpha_x f_hat
    ∂f/∂y_hat = alpha_y f_hat
    """
    d_x = zeros(ComplexF64, nx, ny)
    d_y = zeros(ComplexF64, nx, ny)
    
    for i = 1:nx
        for j = 1:ny
            kx, ky = kxx[i], kyy[j]
            d_x[i, j] = (2*pi/Lx * kx)*im
            d_y[i, j] = (2*pi/Ly * ky)*im
        end
    end
    
    return d_x, d_y
    
end


function Apply_Curl!(mesh::Spectral_Mesh, f::Array{Float64,2}, curl_f_hat::Array{ComplexF64,2}) 
    """
    ∇×f_hat = ∂f/∂x_hat - ∂f/∂y_hat
    """
    d_x, d_y = mesh.d_x, mesh.d_y
    
    Trans_Grid_To_Spectral!(mesh, f, curl_f_hat)
    
    curl_f_hat .*= (d_x - d_y)
end



function UV_Grid_Init(nx::Int64, ny::Int64, Lx::Float64, Ly::Float64, kxx::Array{Int64, 1}, kyy::Array{Int64, 1}) 
    """
    u_hat =  (i 2π/Ly ky) /|k| ω_hat = alpha_x[kx,ky] ω_hat
    v_hat = -(i 2π/Lx kx) /|k| ω_hat = alpha_y[kx,ky] ω_hat
    """
    alpha_x = zeros(ComplexF64, nx, ny)
    alpha_y = zeros(ComplexF64, nx, ny)
    
    for i = 1:nx
        for j = 1:ny
            kx, ky = kxx[i], kyy[j]
            mag = (2*pi/Lx * kx)^2 + (2*pi/Ly * ky)^2
            alpha_x[i, j] =  (2*pi/Ly * ky)/mag * im
            alpha_y[i, j] = -(2*pi/Lx * kx)/mag * im
        end
    end
    alpha_x[1,1] = 0.0
    alpha_y[1,1] = 0.0
    
    return alpha_x, alpha_y
    
end

"""
Vorticity             ω = ∇×U,  
Stream-function       U = curl ψ = [∂ψ/∂y, -∂ψ/∂x]
ω = -Δψ
We have       
ψ_hat =  1/|k| ω_hat
u_hat =  (i 2π/Ly ky) /|k| ω_hat
v_hat = -(i 2π/Lx kx) /|k| ω_hat
"""
function UV_Spectral_From_Vor!(mesh::Spectral_Mesh, ω_hat::Array{ComplexF64,2}, u_hat::Array{ComplexF64,2}, v_hat::Array{ComplexF64,2})
    
    alpha_x, alpha_y = mesh.alpha_x, mesh.alpha_y
    
    u_hat .= alpha_x .* ω_hat
    v_hat .= alpha_y .* ω_hat
    
end


function Vor_From_UV_Grid!(mesh::Spectral_Mesh, u_hat::Array{ComplexF64,2}, v_hat::Array{ComplexF64,2},ω_hat::Array{ComplexF64,2}, ω::Array{Float64,2}, )
    d_x, d_y = mesh.d_x, mesh.d_y
    ω_hat .= d_x.*v_hat - d_y.*u_hat
    
    Trans_Spectral_To_Grid!(mesh, ω_hat, ω)    
end

function Apply_Gradient!(mesh::Spectral_Mesh, ω_hat::Array{ComplexF64,2}, ω_x::Array{Float64,2}, ω_y::Array{Float64,2})
    """
    ωx = ifft(ωx_hat)
    ωy = ifft(ωy_hat)
    ωx_hat = d_x ω_hat
    ωy_hat = d_y ω_hat
    """
    d_x, d_y = mesh.d_x, mesh.d_y
    ω_x_hat, ω_y_hat = mesh.ω_x_hat, mesh.ω_y_hat
    
    ω_x_hat .= d_x .* ω_hat
    ω_y_hat .= d_y .* ω_hat
    
    Trans_Spectral_To_Grid!(mesh, ω_x_hat, ω_x)
    Trans_Spectral_To_Grid!(mesh, ω_y_hat, ω_y)
end


function Add_Horizontal_Advection!(mesh::Spectral_Mesh, ω_hat::Array{ComplexF64,2}, δω_hat::Array{ComplexF64, 2})
    
    """
    δω_hat -= hat { (U⋅∇)ω }
    
    """
    u, v = mesh.u, mesh.v
    u_hat, v_hat = mesh.u_hat, mesh.v_hat
    ω_x, ω_y = mesh.ω_x, mesh.ω_y
    UV_Spectral_From_Vor!(mesh, ω_hat, u_hat, v_hat)

    Trans_Spectral_To_Grid!(mesh, u_hat, u)
    Trans_Spectral_To_Grid!(mesh, v_hat, v)

    Apply_Gradient!(mesh, ω_hat, ω_x, ω_y)
    
    δω_hat .-= mesh.alias_filter .* fft(u.*ω_x + v.*ω_y)
    
end


function Visual(mesh::Spectral_Mesh, u::Array{Float64,2}, var_name::String, save_file_name::String="None")

        nx, ny = mesh.nx, mesh.ny
        xx, yy = mesh.xx, mesh.yy
        X,Y = repeat(xx, 1, ny), repeat(yy, 1, nx)'
        
        
        pcolormesh(X, Y, u, shading= "gouraud", cmap="viridis")
        xlabel("X")
        ylabel("Y")
        colorbar()
    
        
        if save_file_name != "None"
            savefig(save_file_name)
            close("all")
        end

end

