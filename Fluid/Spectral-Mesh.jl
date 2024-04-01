using FFTW
using PyPlot

mutable struct Spectral_Mesh
    N_x::Int64
    N_y::Int64
    
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




function Spectral_Mesh(N_x::Int64, N_y::Int64, Lx::Float64, Ly::Float64)
    @assert(N_x%2 == 0 && N_y%2 == 0)
    
    Δx = Lx/N_x
    xx = LinRange(0, Lx-Δx, N_x)
    
    Δy = Ly/N_y
    yy = LinRange(0, Ly-Δy, N_y)

    kxx, kyy, alias_filter = Spectral_Init(N_x, N_y)
    
    
    d_x, d_y = Apply_Gradient_Init(N_x, N_y, Lx, Ly, kxx, kyy) 
    
    alpha_x, alpha_y = UV_Grid_Init(N_x, N_y, Lx, Ly, kxx, kyy) 
    
    laplacian_eigs = Apply_Laplacian_Init(N_x, N_y, Lx, Ly, kxx, kyy) 

    
    # container
    u= zeros(Float64, N_x, N_y)
    v= zeros(Float64, N_x, N_y)
    ω_x= zeros(Float64, N_x, N_y)
    ω_y= zeros(Float64, N_x, N_y)
    u_hat= zeros(ComplexF64, N_x, N_y)
    v_hat= zeros(ComplexF64, N_x, N_y)
    ω_x_hat= zeros(ComplexF64, N_x, N_y)
    ω_y_hat= zeros(ComplexF64, N_x, N_y)
    
    
    Spectral_Mesh(N_x, N_y, Lx, Ly, Δx, Δy, xx, yy, kxx, kyy, alias_filter, 
                  d_x, d_y, alpha_x, alpha_y, laplacian_eigs, 
                  u, v, ω_x, ω_y, u_hat, v_hat, ω_x_hat, ω_y_hat)
end

"""
Compute mode numbers kxx and kyy
kxx = [0,1,...,[N_x/2]-1, -[N_x/2], -[N_x/2]+1, ... -1]   
kyy = [0,1,...,[N_y/2]-1, -[N_y/2], -[N_y/2]+1, ... -1]
and alias filter based O
"""
function Spectral_Init(N_x::Int64, N_y::Int64)
    kxx = mod.((1:N_x) .- ceil(Int64, N_x/2+1),N_x) .- floor(Int64, N_x/2)
    kyy = mod.((1:N_y) .- ceil(Int64, N_y/2+1),N_y) .- floor(Int64, N_y/2)

    alias_filter = zeros(Float64, N_x, N_y)
    for i = 1:N_x
        for j = 1:N_y
            # if (abs(kxx[i]) < N_x/3  && abs(kyy[j]) < N_y/3)
            if (abs(kxx[i]) < N_x/4  && abs(kyy[j]) < N_y/4)
                alias_filter[i, j] = 1.0
            end
        end
    end
    return kxx, kyy, alias_filter
end




function Trans_Spectral_To_Grid!(mesh::Spectral_Mesh, u_hat::Array{ComplexF64,2},  u::Array{Float64,2})
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



function Trans_Grid_To_Spectral!(mesh::Spectral_Mesh, u::Array{Float64,2}, u_hat::Array{ComplexF64,2})
    
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


function Apply_Laplacian_Init(N_x::Int64, N_y::Int64, Lx::Float64, Ly::Float64, kxx::Array{Int64, 1}, kyy::Array{Int64, 1}) 
    """
    See Apply_Laplacian!
    """
    laplacian_eig = zeros(Float64, N_x, N_y)
    
    for i = 1:N_x
        for j = 1:N_y
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

function Solve_Laplacian!(mesh::Spectral_Mesh, Δu_hat::Array{ComplexF64,2}, u_hat::Array{ComplexF64,2}) 
    """
    Δ (ω_hat[kx,ky]  e^{i (2π/Lx kx x + 2π/Ly ky y)}) = -((2πkx/Lx)² + (2πky/Ly)² )ω_hat
    """
    eig = mesh.laplacian_eigs
    u_hat .= Δu_hat ./ eig
    u_hat[1, 1] = 0
end


function Apply_Gradient_Init(N_x::Int64, N_y::Int64, Lx::Float64, Ly::Float64, kxx::Array{Int64, 1}, kyy::Array{Int64, 1}) 
    """
    ∂f/∂x_hat = alpha_x f_hat
    ∂f/∂y_hat = alpha_y f_hat
    """
    d_x = zeros(ComplexF64, N_x, N_y)
    d_y = zeros(ComplexF64, N_x, N_y)
    
    for i = 1:N_x
        for j = 1:N_y
            kx, ky = kxx[i], kyy[j]
            d_x[i, j] = (2*pi/Lx * kx)*im
            d_y[i, j] = (2*pi/Ly * ky)*im
        end
    end
    
    return d_x, d_y
    
end

"""
(∇×f)_hat = ∂fy/∂x_hat - ∂fx/∂y_hat
"""
function Apply_Curl!(mesh::Spectral_Mesh, fx::Array{Float64,2}, fy::Array{Float64,2}, curl_f_hat::Array{ComplexF64,2}) 
    d_x, d_y = mesh.d_x, mesh.d_y
    fx_hat, fy_hat = mesh.u_hat, mesh.v_hat

    Trans_Grid_To_Spectral!(mesh, fx, fx_hat)  
    Trans_Grid_To_Spectral!(mesh, fy, fy_hat)

    curl_f_hat .= d_x.*fy_hat - d_y.*fx_hat

end

"""
(∇f)_hat = ∂fy/∂x_hat - ∂fx/∂y_hat
"""
function Apply_Gradient!(mesh::Spectral_Mesh, fx::Array{Float64,2}, fy::Array{Float64,2}, ∇f_hat::Array{ComplexF64,2}) 
    d_x, d_y = mesh.d_x, mesh.d_y
    fx_hat, fy_hat = mesh.u_hat, mesh.v_hat

    Trans_Grid_To_Spectral!(mesh, fx, fx_hat)  
    Trans_Grid_To_Spectral!(mesh, fy, fy_hat)

    ∇f_hat .= d_x.*fx_hat + d_y.*fy_hat

end



function UV_Grid_Init(N_x::Int64, N_y::Int64, Lx::Float64, Ly::Float64, kxx::Array{Int64, 1}, kyy::Array{Int64, 1}) 
    """
    u_hat =  (i 2π/Ly ky) /|k| ω_hat = alpha_x[kx,ky] ω_hat
    v_hat = -(i 2π/Lx kx) /|k| ω_hat = alpha_y[kx,ky] ω_hat
    """
    alpha_x = zeros(ComplexF64, N_x, N_y)
    alpha_y = zeros(ComplexF64, N_x, N_y)
    
    for i = 1:N_x
        for j = 1:N_y
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
function UV_Spectral_From_Vor!(mesh::Spectral_Mesh, ω_hat::Array{ComplexF64,2}, u_hat::Array{ComplexF64,2}, v_hat::Array{ComplexF64,2}, ub::Float64, vb::Float64)
    N_x, N_y = mesh.N_x, mesh.N_y
    alpha_x, alpha_y = mesh.alpha_x, mesh.alpha_y
    
    u_hat .= alpha_x .* ω_hat
    v_hat .= alpha_y .* ω_hat

    u_hat[1,1] = ub * (N_x*N_y)
    v_hat[1,1] = vb * (N_x*N_y)
    
end


function Vor_From_UV_Spectral!(mesh::Spectral_Mesh, u_hat::Array{ComplexF64,2}, v_hat::Array{ComplexF64,2},ω_hat::Array{ComplexF64,2}, ω::Array{Float64,2}, )
    d_x, d_y = mesh.d_x, mesh.d_y
    ω_hat .= d_x.*v_hat - d_y.*u_hat
    
    Trans_Spectral_To_Grid!(mesh, ω_hat, ω)    
end

"""
Compute gradients ω_x, ω_y, from spectral input ω_hat
ωx_hat = d_x ω_hat
ωy_hat = d_y ω_hat
ωx = ifft(ωx_hat)
ωy = ifft(ωy_hat)
"""
function Apply_Gradient!(mesh::Spectral_Mesh, ω_hat::Array{ComplexF64,2}, ω_x::Array{Float64,2}, ω_y::Array{Float64,2})

    d_x, d_y = mesh.d_x, mesh.d_y
    ω_x_hat, ω_y_hat = mesh.ω_x_hat, mesh.ω_y_hat
    
    ω_x_hat .= d_x .* ω_hat
    ω_y_hat .= d_y .* ω_hat
    
    Trans_Spectral_To_Grid!(mesh, ω_x_hat, ω_x)
    Trans_Spectral_To_Grid!(mesh, ω_y_hat, ω_y)
end

"""
δω_hat = hat { (U⋅∇)ω }

"""
function Compute_Horizontal_Advection!(mesh::Spectral_Mesh, ω_hat::Array{ComplexF64,2}, δω_hat::Array{ComplexF64, 2}, ub::Float64, vb::Float64)
    u, v = mesh.u, mesh.v
    u_hat, v_hat = mesh.u_hat, mesh.v_hat
    ω_x, ω_y = mesh.ω_x, mesh.ω_y
    UV_Spectral_From_Vor!(mesh, ω_hat, u_hat, v_hat, ub, vb)
    
    Trans_Spectral_To_Grid!(mesh, u_hat, u)
    Trans_Spectral_To_Grid!(mesh, v_hat, v)
    
    Apply_Gradient!(mesh, ω_hat, ω_x, ω_y)
    
    Trans_Grid_To_Spectral!(mesh, u.*ω_x + v.*ω_y,  δω_hat)
    
end



function Visual(mesh::Spectral_Mesh, u::Array{Float64,2}, var_name::String,
    save_file_name::String="None", vmin=nothing, vmax=nothing)
    
    N_x, N_y = mesh.N_x, mesh.N_y
    xx, yy = mesh.xx, mesh.yy
    X,Y = repeat(xx, 1, N_y), repeat(yy, 1, N_x)'
    
    figure()
    pcolormesh(X, Y, u, shading= "gouraud", cmap="viridis", vmin=vmin, vmax =vmax)
    xlabel("X")
    ylabel("Y")
    colorbar()
    
    
    if save_file_name != "None"
        tight_layout()
        savefig(save_file_name)
    end
    
end


function Visual_Obs(mesh::Spectral_Mesh, u::Array{Float64,2}, x_locs::Array{Int64,1},  y_locs::Array{Int64,1}, var_name::String; symmetric::Bool=false, save_file_name::String="None", vmin=nothing, vmax=nothing)
    
    N_x, N_y = mesh.N_x, mesh.N_y
    xx, yy = mesh.xx, mesh.yy
    X,Y = repeat(xx, 1, N_y), repeat(yy, 1, N_x)'
    
    x_obs, y_obs = X[x_locs,y_locs][:], Y[x_locs,y_locs][:] 
    
    figure()
    pcolormesh(X, Y, u, shading= "gouraud", cmap="viridis", vmin=vmin, vmax =vmax)
    colorbar()
    
    scatter(x_obs, y_obs, color="black")
    if symmetric
        mirror_loc_ind = [1;N_x:-1:2]
        x_obs_sym, y_obs_sym = X[mirror_loc_ind[x_locs],y_locs][:], Y[mirror_loc_ind[x_locs],y_locs][:]   
        scatter(x_obs_sym, y_obs_sym, facecolors="none", edgecolors="black")
    end

    
    xlabel("X")
    ylabel("Y")
    
    
    
    if save_file_name != "None"
        tight_layout()
        savefig(save_file_name)
    end
    
end

