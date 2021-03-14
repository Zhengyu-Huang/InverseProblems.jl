using LinearAlgebra
using Random
using Distributions

include("Spectral-Mesh.jl")


"""
Navier stokes equation with double periodicity
U = (u,v) and p denote velocity and pressure 
∂U/∂t + U⋅∇U - νΔU + ∇p = f
∇⋅U = 0

Let ω = ∇×U, we have the vorticity equation
∂ω/∂t + (U⋅∇)ω - νΔω  = ∇×f

ω = 1/(N_xN_y)∑_{kx, ky}  ω_hat[kx,ky]  e^{i (2π/Lx kx x + 2π/Ly ky y)}

∂ω_hat/∂t + FFT[(U⋅∇)ω] + ν( (2πkx/Lx)² + (2πky/Ly)² )ω_hat  = 2π/Lx kx fy_hat - 2π/Ly ky fx_hat

The domain is [0,Lx][0,Ly]
"""
mutable struct Spectral_NS_Solver
    mesh::Spectral_Mesh

    ν::Float64

    curl_f_hat::Array{ComplexF64, 2}

    ub::Float64
    vb::Float64
    # Since the streamfunction is assumed to be periodic 
    # U = curl ψ  + Ub = [∂ψ/∂y + ub, -∂ψ/∂x + vb]

    ω_hat::Array{ComplexF64, 2}
    u_hat::Array{ComplexF64, 2}
    v_hat::Array{ComplexF64, 2}

    ω::Array{Float64, 2}
    u::Array{Float64, 2}
    v::Array{Float64, 2}

    Δω_hat::Array{ComplexF64, 2}
    δω_hat::Array{ComplexF64, 2}

    k1::Array{ComplexF64, 2}
    k2::Array{ComplexF64, 2}
    k3::Array{ComplexF64, 2}
    k4::Array{ComplexF64, 2}
end


# constructor of the Spectral_NS_Solver
#
# There are two forcing intialization approaches
# * initialize fx and fy components, user shoul make sure they are zero-mean and periodic
# * initialize ∇×f, curl_f
#
# There are velocity(vorticity) intialization approaches
# * initialize u0 and v0 components, user should make sure they are incompressible div ⋅ (u0 , v0) = 0
# * initialize ω0 and mean backgroud velocity ub and vb
#
function Spectral_NS_Solver(mesh::Spectral_Mesh, ν::Float64;
    fx::Union{Array{Float64, 2}, Nothing} = nothing, fy::Union{Array{Float64, 2}, Nothing} = nothing, 
    curl_f::Union{Array{Float64, 2}, Nothing} = nothing,
    u0::Union{Array{Float64, 2}, Nothing} = nothing, v0::Union{Array{Float64, 2}, Nothing} = nothing, 
    ω0::Union{Array{Float64, 2}, Nothing} = nothing, ub::Union{Float64, Nothing} = nothing, vb::Union{Float64, Nothing} = nothing)    
    N_x, N_y = mesh.N_x, mesh.N_y
    
    curl_f_hat = zeros(ComplexF64, N_x, N_y)
    if curl_f === nothing
        Apply_Curl!(mesh, fx, fy, curl_f_hat) 
    else
        Trans_Grid_To_Spectral!(mesh, curl_f, curl_f_hat)
    end
    
    
    u_hat = zeros(ComplexF64, N_x, N_y)
    v_hat = zeros(ComplexF64, N_x, N_y)
    ω_hat = zeros(ComplexF64, N_x, N_y)
    u = zeros(Float64, N_x, N_y)
    v = zeros(Float64, N_x, N_y)
    ω = zeros(Float64, N_x, N_y)
    if ω0 === nothing
    
        u .= u0
        v .= v0
        
        Trans_Grid_To_Spectral!(mesh, u, u_hat)
        Trans_Grid_To_Spectral!(mesh, v, v_hat)
        
        ub, vb = u_hat[1,1]/(N_x*N_y), v_hat[1,1]/(N_x*N_y)
        
        Vor_From_UV_Spectral!(mesh, u_hat, v_hat, ω_hat, ω)
        
    else
        ω = zeros(Float64, N_x, N_y)
        ω .= ω0
        ω_hat = zeros(ComplexF64, N_x, N_y)
        Trans_Grid_To_Spectral!(mesh, ω, ω_hat)
        
        
        UV_Spectral_From_Vor!(mesh, ω_hat, u_hat, v_hat, ub, vb)
        
        
        Trans_Spectral_To_Grid!(mesh, u_hat, u)
        Trans_Spectral_To_Grid!(mesh, v_hat, v)
    end
    
    
    δω_hat = zeros(ComplexF64, N_x, N_y)
    Δω_hat = zeros(ComplexF64, N_x, N_y)
    
    k1 = zeros(ComplexF64, N_x, N_y)
    k2 = zeros(ComplexF64, N_x, N_y)
    k3 = zeros(ComplexF64, N_x, N_y)
    k4 = zeros(ComplexF64, N_x, N_y)
    
    Spectral_NS_Solver(mesh, ν, curl_f_hat, ub, vb, ω_hat, u_hat, v_hat, ω, u, v, Δω_hat, δω_hat, k1, k2, k3, k4)
end


# compute convective and diffusive stable time step 
function Stable_Δt(mesh::Spectral_Mesh, ν::Float64, u::Array{Float64,2}, v::Array{Float64,2})
    Δx, Δy = mesh.Δx, mesh.Δy
    Δ = min(Δx, Δy)
    u_max, v_max = maximum(abs.(u)), maximum(abs.(v))


    Δt = min(Δx/u_max, Δy/v_max, Δ^2/(2*ν))

    #@info "maximum stable Δt = ", Δt

    return Δt
end

# ∂ω/∂t + (U⋅∇)ω - νΔω  = ∇×f
# ∂ω_hat/∂t = -FFT[(U⋅∇)ω] + ν(- ((2πkx/Lx)² + (2πky/Ly)²) )ω_hat  + curl_f_hat
# Compute the right hand-side
function Explicit_Residual!(self::Spectral_NS_Solver, ω_hat::Array{ComplexF64, 2}, δω_hat::Array{ComplexF64, 2})
    mesh = self.mesh
    ub, vb = self.ub, self.vb
    Compute_Horizontal_Advection!(mesh, ω_hat, δω_hat, ub, vb)

    δω_hat .= self.curl_f_hat - δω_hat

    Δω_hat = self.Δω_hat

    Apply_Laplacian!(mesh, ω_hat, Δω_hat)

    δω_hat .+= self.ν * Δω_hat
end

# Also the Crank-Nicolson
# ∂ω/∂t + (U⋅∇)ω - νΔω  = ∇×f
# (ω_hat(n+1) - ω_hat(n))/Δt  = -FFT[(U⋅∇)ω] + ν(- ((2πkx/Lx)² + (2πky/Ly)²) )(ω_hat(n)+ω_hat(n+1))/2  + curl_f_hat
# [1 - νΔt(-(2πkx/Lx)² + -(2πky/Ly)²)/2] (ω_hat(n+1)-ω_hat(n))/Δt = -FFT[(U⋅∇)ω] + ν(-((2πkx/Lx)² + (2πky/Ly)²))ω_hat(n)/2  + curl_f_hat
# compute (ω_hat(n+1)-ω_hat(n))/Δt
function Semi_Implicit_Residual!(self::Spectral_NS_Solver, ω_hat::Array{ComplexF64, 2}, Δt::Float64, δω_hat::Array{ComplexF64, 2})

    mesh = self.mesh
    ub, vb = self.ub, self.vb
    Compute_Horizontal_Advection!(mesh, ω_hat, δω_hat, ub, vb)

    δω_hat .= self.curl_f_hat - δω_hat

    Δω_hat = self.Δω_hat
    
    Apply_Laplacian!(mesh, ω_hat, Δω_hat)

    δω_hat .+= self.ν * Δω_hat

    δω_hat ./= (1.0 .- 0.5*self.ν*Δt * mesh.laplacian_eigs)

end


# Update ω, u_hat, v_hat, u, v from ω_hat
function Update_Grid_Vars!(self::Spectral_NS_Solver)

    ω_hat, u_hat, v_hat = self.ω_hat, self.u_hat, self.v_hat
    ω, u, v = self.ω, self.u, self.v
    ub, vb = self.ub, self.vb

    mesh = self.mesh

    Trans_Spectral_To_Grid!(mesh, ω_hat, ω)
    UV_Spectral_From_Vor!(mesh, ω_hat, u_hat, v_hat, ub, vb)
    Trans_Spectral_To_Grid!(mesh, u_hat, u)
    Trans_Spectral_To_Grid!(mesh, v_hat, v)

    return ω, u, v
    
end


function Solve!(self::Spectral_NS_Solver, Δt::Float64, method::String)
    ω_hat, δω_hat = self.ω_hat, self.δω_hat

    if method == "Crank-Nicolson"
        Semi_Implicit_Residual!(self, ω_hat, Δt, δω_hat)
        ω_hat .+= Δt*δω_hat
    elseif method == "RK4"
        k1, k2, k3, k4 = self.k1, self.k2, self.k3, self.k4 
        Explicit_Residual!(self, ω_hat,  k1)
        Explicit_Residual!(self, ω_hat + Δt/2.0*k1, k2)
        Explicit_Residual!(self, ω_hat + Δt/2.0*k2, k3)
        Explicit_Residual!(self, ω_hat + Δt*k3, k4)
        
        ω_hat .+= Δt/6.0*(k1 + 2*k2 + 2*k3 + k4)
    end
end






#########################
struct Setup_Param
    # physics
    ν::Float64        # dynamic viscosity
    ub::Float64       # background velocity 
    vb::Float64
    fx::Array{Float64, 2}
    fy::Array{Float64, 2}

    # discretization
    N::Int64            # number of grid points for both x and y directions (including both ends)
    L::Float64            # computational domain [0, L]×[0, L]
    Δx::Float64
    xx::Array{Float64, 1}  # uniform grid [0, Δx, 2Δx ... L]

    method::String    # time discretization method 
    N_t::Int64        # number of time steps
    T::Float64        # total time

    # observation locations is tensor product x_locs × y_locs
    x_locs::Array{Int64, 1}
    y_locs::Array{Int64, 1}
    t_locs::Array{Int64, 1}

    # for parameterization
    seq_pairs::Array{Int64, 2} # indexing eigen pairs of KL expansion, length(seq_pairs) >= N_θ 
    N_KL::Int64                # this is for generating the with certain modes (N_KL > 0), when N_KL <= 0, generating ω0_ref with all modes
    θ_ref::Array{Float64, 1}   # coefficient of these truth modes (N_KL > 0)
    ω0_ref::Array{Float64, 2}

    
    # inverse parameters
    θ_names::Array{String, 1}
    N_θ::Int64   
    N_y::Int64   
        
end

function Setup_Param(ν::Float64, ub::Float64, vb::Float64,  
    N::Int64, L::Float64,  
    method::String, N_t::Int64,
    obs_ΔNx::Int64, obs_ΔNy::Int64, obs_ΔNt::Int64, 
    N_KL::Int64,
    N_θ::Int64; 
    f::Union{Function, Nothing} = nothing,
    σ::Float64 = sqrt(2)*pi, seed::Int64=123)
    
    #observation
    x_locs = Array(1:obs_ΔNx:N)
    y_locs = Array(1:obs_ΔNy:N)
    t_locs = Array(obs_ΔNt:obs_ΔNt:N_t)

    
    N_y = length(x_locs)*length(y_locs)*length(t_locs)
    seq_pairs = Compute_Seq_Pairs(max(N_KL, N_θ))

    mesh = Spectral_Mesh(N, N, L, L)

    if N_KL > 0
        # The truth is generated with first N_KL modes
        Random.seed!(seed);
        θ_ref = rand(Normal(0, σ), N_KL)
        ω0_ref = Initial_ω0_KL(mesh, θ_ref, seq_pairs) 
    else
        
        # The truth is generated with all modes
        ω0_ref = Initial_ω0_KL(mesh, σ; seed=seed)
        # project to obtain the first N_θ modes
        θ_ref = Construct_θ0(ω0_ref, N_θ, seq_pairs, mesh)
    end

    θ_names = ["ω0"]

    xx = Array(LinRange(0, L, N+1))

    fx = zeros(Float64, N, N)
    fy = zeros(Float64, N, N)
    if f !== nothing
        for i_x = 1:N
            for i_y = 1:N
                fx[i_x, i_y], fy[i_x, i_y] = f(xx[i_x], xx[i_y])
            end
        end
    end
    Setup_Param(ν, ub, vb, fx, fy, N, L, L/N, xx,  method, N_t, T, x_locs, y_locs, t_locs, seq_pairs, N_KL, θ_ref, ω0_ref, θ_names, N_θ, N_y)
end


# For fourier modes, consider the zero-mean real field
# u = ∑ f_k e^{ik⋅x}, we have  f_{k} = conj(f_{-k})
# we can also use cos(k⋅x)/2π^2 and sin(k⋅x)/2π^2 as basis (L2 norm is 1)
# with (kx + ky > 0 || (kx + ky == 0 && kx > 0)) 
function Mode_Helper(kx, ky)
    return (kx + ky > 0 || (kx + ky == 0 && kx > 0)) 
end


# Generate random numbers and initialize ω0 with all fourier modes
function Initial_ω0_KL(mesh::Spectral_Mesh, σ::Float64; seed::Int64 = 123)
    # consider C = -Δ^{-1}
    # C u = λ u  => -u = λ Δ u
    # u = e^{ik⋅x}, λ Δ u = -λ |k|^2 u = - u => λ = 1/|k|^2
    # basis are cos(k⋅x)/√2π and sin(k⋅x)/√2π, k!=(0,0), zero mean.
    # ω = ∑ ak cos(k⋅x)/√2π|k|^2 + ∑ bk sin(k⋅x)/√2π|k|^2,    &  kx + ky > 0 or (kx + ky = 0 and kx > 0) 
    #   = ∑ (ak/(2√2π|k|^2) - i bk/(2√2π|k|^2) )e^{ik⋅x} + (ak/(2√2π|k|^2) + i bk/(2√2π^2|k|^2) )e^{-ik⋅x}
    
    # the basis is ordered as 0,1,...,N_x/2-1, -N_x/2, -N_x/2+1, ... -1
    
    N_x , N_y = mesh.N_x, mesh.N_y
    kxx, kyy = mesh.kxx, mesh.kyy
    
    Random.seed!(seed);
    abk = rand(Normal(0, σ), N_x, N_y, 2)
    
    ω0_hat = zeros(ComplexF64, N_x, N_y)
    ω0 = zeros(Float64, N_x, N_y)
    for ix = 1:N_x
        for iy = 1:N_y
            kx, ky = kxx[ix], kyy[iy]
            if Mode_Helper(kx, ky)
                ak, bk = abk[ix, iy, 1], abk[ix, iy, 2]
                ω0_hat[ix, iy] = (ak - bk*im)/(2*sqrt(2)*pi*(kx^2+ky^2))
                # (ix=1 iy=1 => kx=0 ky=0) 1 => 1, i => n-i+2
                ω0_hat[(ix==1 ? 1 : N_x-ix+2), (iy==1 ? 1 : N_y-iy+2)] = (ak + bk*im)/(2*sqrt(2)*pi*(kx^2+ky^2))
                
            end
        end
    end
    ω0_hat .*= mesh.alias_filter
    Trans_Spectral_To_Grid!(mesh, N_x*N_y * ω0_hat, ω0)
    
    
    # ω0_test = zeros(Float64, N_x, N_y)
    # X, Y = zeros(Float64, N_x, N_y), zeros(Float64, N_x, N_y)
    # Δx, Δy = mesh.Δx, mesh.Δy
    # for ix = 1:N_x
    #     for iy = 1:N_y
    #         X[ix, iy] = (ix-1)*Δx
    #         Y[ix, iy] = (iy-1)*Δy
    #     end
    # end
    
    # for ix = 1:N_x
    #     for iy = 1:N_y
    #         kx, ky = kxx[ix], kyy[iy]
    #         if (abs(kx) < N_x/3 && abs(ky) < N_y/3) && (kx + ky > 0 || (kx + ky == 0 && kx > 0)) 
    #             ak, bk = abk[ix, iy, 1], abk[ix, iy, 2]
    #             ω0_test .+= (ak * cos.(kx*X + ky*Y) + bk * sin.(kx*X + ky*Y))/(sqrt(2)*pi*(kx^2 + ky^2))
    #         end
    #     end
    # end
    # @info  "Error", norm(ω0 - ω0_test)
    # error("Stop")
    
    
    
    ω0
end



# For fourier modes, consider the zero-mean real field
# u = ∑ f_k e^{ik⋅x}, we have  f_{k} = conj(f_{-k})
# we can also use cos(k⋅x)/2π^2 and sin(k⋅x)/2π^2 as basis (L2 norm is 1)
# with (kx + ky > 0 || (kx + ky == 0 && kx > 0)) 
# And these fourier modes are sorted by 1/|k|^2 and the first N_KL modes 
# are computed in terms of (kx, ky) pair
function Compute_Seq_Pairs(N_KL::Int64)
    seq_pairs = zeros(Int64, N_KL, 2)
    trunc_N_x = trunc(Int64, sqrt(2*N_KL)) + 1
    
    seq_pairs = zeros(Int64, div(((2trunc_N_x+1)^2 - 1), 2), 2)
    seq_pairs_mag = zeros(Int64, div(((2trunc_N_x+1)^2 - 1),2))
    
    seq_pairs_i = 0
    for kx = -trunc_N_x:trunc_N_x
        for ky = -trunc_N_x:trunc_N_x
            if Mode_Helper(kx, ky)
                
                seq_pairs_i += 1
                seq_pairs[seq_pairs_i, :] .= kx, ky
                seq_pairs_mag[seq_pairs_i] = kx^2 + ky^2
            end
        end
    end
    
    seq_pairs = seq_pairs[sortperm(seq_pairs_mag), :]
    return seq_pairs[1:N_KL, :]
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
    
    
    
    N_x, N_y = mesh.N_x, mesh.N_y
    kxx, kyy = mesh.kxx, mesh.kyy
    
    ω0_hat = zeros(ComplexF64, N_x, N_y)
    ω0 = zeros(Float64, N_x, N_y)
    abk = reshape(θ, Int64(length(θ)/2), 2)
    N_KL = size(abk,1)
    for i = 1:N_KL
        kx, ky = seq_pairs[i,:]
        if Mode_Helper(kx, ky)
            ak, bk = abk[i,:]
            ix = (kx >= 0 ? kx + 1 : N_x + kx + 1) 
            iy = (ky >= 0 ? ky + 1 : N_y + ky + 1) 
            
            ω0_hat[ix, iy] = (ak - bk*im)/(2*sqrt(2)*pi*(kx^2+ky^2))
            
            # 1 => 1, i => n-i+2
            ω0_hat[(ix==1 ? 1 : N_x-ix+2), (iy==1 ? 1 : N_y-iy+2)] = (ak + bk*im)/(2*sqrt(2)*pi*(kx^2+ky^2))
        end
    end
    ω0_hat .*= mesh.alias_filter
    Trans_Spectral_To_Grid!(mesh, N_x*N_y * ω0_hat, ω0)
    
    
    # ω0_test = zeros(Float64, N_x, N_y)
    # X, Y = zeros(Float64, N_x, N_y), zeros(Float64, N_x, N_y)
    # Δx, Δy = mesh.Δx, mesh.Δy
    # for ix = 1:N_x
    #     for iy = 1:N_y
    #         X[ix, iy] = (ix-1)*Δx
    #         Y[ix, iy] = (iy-1)*Δy
    #     end
    # end
    
    # for i = 1:N_KL
    #     kx, ky = seq_pairs[i,:]
    #     ak, bk = abk[i,:]
    #     @info kx, ky, N_x, N_y
    
    #     @assert((abs(kx) < N_x/3 && abs(ky) < N_y/3) && (kx + ky > 0 || (kx + ky == 0 && kx > 0))) 
    #     ω0_test .+= (ak * cos.(kx*X + ky*Y) + bk * sin.(kx*X + ky*Y))/(2*pi^2*(kx^2 + ky^2))
    # end
    # @info  "Error", norm(ω0 - ω0_test), norm(ω0) , norm(ω0_test)
    # error("Stop")
    
    
    ω0
end

# project  ω0 on the first N_θ dominant modes
# namely, ∑ ak cos(k⋅x) + bk sin(k⋅x) with least square fitting
function Construct_θ0(ω0::Array{Float64, 2}, N_θ::Int64, seq_pairs::Array{Int64,2}, mesh::Spectral_Mesh)
    @assert(N_θ%2 == 0)
    abk0 = zeros(Float64, Int64(N_θ/2), 2)
    
    
    N_x , N_y = mesh.N_x, mesh.N_y
    Δx , Δy   = mesh.Lx/N_x, mesh.Ly/N_y
    # fit the dominant modes from data0 by solving A x = data0
    # ω = ∑ ak cos(k⋅x)/2π^2|k|^2 + ∑ bk sin(k⋅x)/2π^2|k|^2,    &  kx + ky > 0 or (kx + ky = 0 and kx > 0) 
    A = zeros(Float64, length(ω0), N_θ)
    X, Y = zeros(Float64, N_x, N_y), zeros(Float64, N_x, N_y)
    for ix = 1:N_x
        for iy = 1:N_y
            X[ix, iy] = (ix-1)*Δx
            Y[ix, iy] = (iy-1)*Δy
        end
    end
    
    for i = 1:Int64(N_θ/2)
        kx, ky = seq_pairs[i,:]
        
        
        A[:,i]              = (cos.(kx*X + ky*Y)/(2*pi^2*(kx^2 + ky^2)))[:]
        A[:,i+Int64(N_θ/2)] = (sin.(kx*X + ky*Y)/(2*pi^2*(kx^2 + ky^2)))[:]
        
    end
    x = A\ω0[:]
    
    abk0[1:Int64(N_θ/2), :] = reshape(x, Int64(N_θ/2), 2)
    θ0 = abk0[:]

    return θ0
    
end



# generate truth (all Fourier modes) and observations
# observation are at frame 0, Δd_t, 2Δd_t, ... N_t
# with sparse points at Array(1:Δd_x:N_x) × Array(1:Δd_y:N_y)
function forward_helper(s_param::Setup_Param, ω0::Array{Float64,2}, save_file_name::String="None", vmin=nothing, vmax=nothing)

    N_x = N_y = s_param.N
    Lx = Ly = s_param.L
    seq_pairs = s_param.seq_pairs
    method = s_param.method
    x_locs, y_locs, t_locs = s_param.x_locs, s_param.y_locs, s_param.t_locs
 
    mesh = Spectral_Mesh(N_x, N_y, Lx, Ly)
    
    solver = Spectral_NS_Solver(mesh, ν; fx = s_param.fx, fy = s_param.fy, ω0 = ω0, ub = s_param.ub, vb = s_param.vb)  
    Δt_max = Stable_Δt(mesh, ν, solver.u, solver.v)
    
    Δt = T/N_t
    @assert(Δt <= Δt_max)
    # initialize observation 
    
    data = zeros(Float64, size(x_locs,1), size(y_locs,1), size(t_locs,1))
    
    obs_t_id = 1
    for i = 1:N_t
        Solve!(solver, Δt, method)
        if obs_t_id <= size(t_locs,1) && i == t_locs[obs_t_id]
            Update_Grid_Vars!(solver)
            if save_file_name != "None"
                Visual_Obs(mesh, solver.ω, x_locs, y_locs, "ω", save_file_name*string(i)*".pdf", vmin, vmax)
            end
            data[:, :, obs_t_id] = solver.ω[x_locs, y_locs]
            obs_t_id += 1
        end
    end
    
    return data[:]
    
    
end




function forward(s_param::Setup_Param, θ::Array{Float64,1})
    # θ = [ak; bk] and abk = [ak, bk], we have θ = abk[:] and abk = reshape(θ, N_θ/2, 2)
    

    N_x = N_y = s_param.N
    Lx = Ly = s_param.L
    seq_pairs = s_param.seq_pairs
    mesh = Spectral_Mesh(N_x, N_y, Lx, Ly)

    ω0 = Initial_ω0_KL(mesh, θ, seq_pairs)   
    data = forward_helper(s_param, ω0)
    
    return data[:]
end







