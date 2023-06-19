using NNGCM
using JLD2
using Statistics
using LinearAlgebra



include("../Plot.jl")
include("Barotropic.jl")
include("../RExKI.jl")
include("../REnKI.jl")
include("../TRUKI.jl")

mutable struct Barotropic_Data
  mesh::Spectral_Spherical_Mesh
  obs_coord::Array{Int64,2}
  obs_data::Array{Float64, 1}
  grid_vor0_ref::Array{Float64, 3}
  
  
  grid_vor0::Array{Float64, 3}
  spe_vor0::Array{ComplexF64, 3}
  
  nframes::Int64
  init_type::String
end


function convert_obs(obs_coord, obs_raw; antisymmetric=false)
  # update obs
  nobs = size(obs_coord, 1)
  nframes = length(obs_raw)
  obs = zeros(Float64, nobs, nframes)
  
  for i = 1:nframes
    for j = 1:nobs
      if antisymmetric
        obs[j,i] = abs(obs_raw[i][obs_coord[j,1], obs_coord[j,2]] - obs_raw[i][obs_coord[j,1], end-obs_coord[j,2] + 1])
      else
        obs[j,i] = obs_raw[i][obs_coord[j,1], obs_coord[j,2]]
      end
    end
  end
  
  return obs[:]
end


function Barotropic_run(nframes::Int64, init_type::String; init_data=nothing, obs_coord=nothing, spe_vor_b=nothing, symmetric=false)
  
  _, _, _, _, _, _, obs_raw = Barotropic_Main(nframes, init_type, init_data, spe_vor_b, obs_coord)
  
  # update obs
  obs = convert_obs(obs_coord, obs_raw, symmetric)
  
  return obs
end


function Barotropic_ensemble(params_i::Array{Float64, 2},  nframes::Int64, init_type::String,  obs_coord::Array{Int64, 2}, spe_vor_b=nothing)
  
  N_ens,  N_θ = size(params_i)
  N_data = size(obs_coord, 1) * nframes
  
  g_ens = zeros(Float64, N_ens,  N_data)
  
  Threads.@threads for i = 1:N_ens 
    # g: N_ens x N_data
    g_ens[i, :] .= Barotropic_run(nframes, init_type; init_data=params_i[i, :], obs_coord=obs_coord, spe_vor_b=spe_vor_b, symmetric=false)
        
  end
  
  
  return g_ens
end





function Barotropic_GMKI()
  
    mesh, obs_coord, obs_data, init_type = barotropic.mesh, barotropic.obs_coord, barotropic.obs_data, barotropic.init_type
    grid_vor0_ref, grid_vor0, spe_vor0 = barotropic.grid_vor0_ref, barotropic.grid_vor0, barotropic.spe_vor0
    nframes = barotropic.nframes
  
    parameter_names = ["spe_ω"] 
  
    ens_func(θ_ens) = Barotropic_ensemble(θ_ens, nframes, init_type, obs_coord, spe_vor_b)
    
   
    
    
    update_freq = 1
    N_modes = 3
    θ0_w  = fill(1.0, N_modes)/N_modes
    θ0_mean, θθ0_cov  = zeros(N_modes, N_θ), zeros(N_modes, N_θ, N_θ)



    Random.seed!(63);
    σ_0 = 10
    θ0_mean[1, :]    .= rand(Normal(0, σ_0), N_θ) 
    θθ0_cov[1, :, :] .= Array(Diagonal(fill(1.0^2, N_θ)))
    θ0_mean[2, :]    .= rand(Normal(0, σ_0), N_θ) 
    θθ0_cov[2, :, :] .= Array(Diagonal(fill(1.0^2, N_θ)))
    θ0_mean[3, :]    .= rand(Normal(0, σ_0), N_θ) 
    θθ0_cov[3, :, :] .= Array(Diagonal(fill(1.0^2, N_θ)))


    γ = 1.0
# Δt = γ/(1+γ)
ukiobj = GMUKI_Run(darcy, aug_forward, θ0_w, θ0_mean, θθ0_cov, aug_y, aug_Σ_η, γ, update_freq, N_iter; unscented_transform="modified-2n+1")

  
  rukiobj = ExKIObj(parameter_names,
  θ0_bar, 
  θθ0_cov,
  t_mean, # observation
  t_cov,
  α_reg)
  
  for i in 1:N_iter
    
    if i==1
      Barotropic_ω0!(mesh, init_type, rukiobj.θ_bar[end], spe_vor0, grid_vor0; spe_vor_b=spe_vor_b)
      @info "i0: error of ω0 :", norm(grid_vor0_ref - grid_vor0), " / ",  norm(grid_vor0_ref)
    end
    
    update_ensemble!(rukiobj, ens_func) 
    
    
    params_i = deepcopy(rukiobj.θ_bar[end])
    
    
    Barotropic_ω0!(mesh, init_type, params_i, spe_vor0, grid_vor0, spe_vor_b=spe_vor_b)
    @info "error of ω0 :", norm(grid_vor0_ref - grid_vor0), " / ",  norm(grid_vor0_ref)
    
    @info "optimization error :", norm(obs_data - rukiobj.g_bar[end]), " / ",  norm(obs_data)
    if i%10 == 0
      Lat_Lon_Pcolormesh(mesh, grid_vor0, 1; save_file_name = "Figs/RUKI_Barotropic_vor_"*string(i)*".png", vmax = vor0_max, vmin = vor0_vmin, cmap = "jet")
      
      
      begin # compute standard deviation on the diagonal
        
        α_vor0 = rukiobj.θθ_cov[end]
        α_vor0_θ_basis = θ_basis*α_vor0
        
        d_Cov_0 = similar(grid_vor0)
        for j = 1:length(grid_vor0)
          d_Cov_0[j] = sqrt( sum(θ_basis[j, :] .* α_vor0_θ_basis[j,:]) )
        end
        Lat_Lon_Pcolormesh(mesh, d_Cov_0, 1;  save_file_name = "Figs/RUKI_Barotropic_vor_std"*string(i)*".png", cmap = "jet")
      end
      
      
    end
  end
  
  return rukiobj
  
end









# Compare(0.5, 0.05, false)
