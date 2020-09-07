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
  grid_vor0_ref::Array{Float64, 3}


  grid_vor0::Array{Float64, 3}
  spe_vor0::Array{ComplexF64, 3}
  
  ndays::Int64
  init_type::String
end


function convert_obs(obs_coord, obs_raw)
  # update obs
  nobs = size(obs_coord, 1)
  nframes = length(obs_raw)
  obs = zeros(Float64, nobs, nframes)
  
  for i = 1:nframes
    for j = 1:nobs

      obs[j,i] = obs_raw[i][obs_coord[j,1], obs_coord[j,2]]
    end
  end
  
  return obs[:]
end


function Barotropic_run(ndays::Int64, init_type::String, init_data=nothing, obs_coord=nothing)
  
  _, _, _, _, _, obs_raw = Barotropic_Main(ndays, init_type, init_data, obs_coord)

  # update obs
  obs = convert_obs(obs_coord, obs_raw)

  return obs
end

function Barotropic_ensemble(params_i::Array{Float64, 2},  ndays::Int64, init_type::String,  obs_coord::Array{Int64, 2})
  
  N_ens,  N_θ = size(params_i)
  N_data = size(obs_coord, 1) * ndays
  
  g_ens = zeros(Float64, N_ens,  N_data)

  Threads.@threads for i = 1:N_ens 
    # g: N_ens x N_data
    g_ens[i, :] .= Barotropic_run(ndays, init_type, params_i[i, :], obs_coord)
  end
  
  return g_ens
end




function Barotropic_RUKI(barotropic::Barotropic_Data, t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, θ0_bar::Array{Float64,1}, θθ0_cov::Array{Float64,2}, 
  α_reg::Float64 = 1.0, N_iter::Int64 = 100)
  
  mesh, obs_coord, init_type = barotropic.mesh, barotropic.obs_coord, barotropic.init_type
  grid_vor0_ref, grid_vor0, spe_vor0 = barotropic.grid_vor0_ref, barotropic.grid_vor0, barotropic.spe_vor0
  ndays = barotropic.ndays
  
  parameter_names = ["spe_ω"] 
  
  ens_func(θ_ens) = Barotropic_ensemble(θ_ens, ndays, init_type, obs_coord)
  
  rukiobj = ExKIObj(parameter_names,
  θ0_bar, 
  θθ0_cov,
  t_mean, # observation
  t_cov,
  α_reg)
  
  for i in 1:N_iter
    
    update_ensemble!(rukiobj, ens_func) 
    
    
    params_i = deepcopy(rukiobj.θ_bar[end])
    
    
    Barotropic_ω0!(mesh, init_type, params_i, spe_vor0, grid_vor0)
    
    
    @info "F error of ω0 :", norm(grid_vor0_ref - grid_vor0), " / ",  norm(grid_vor0_ref)
    
  end
  
  return rukiobj
  
end

function Barotropic_TRUKI(barotropic::Barotropic_Data, t_mean::Array{Float64,1}, t_cov, θ0_bar::Array{Float64,1}, θθ0_cov_sqr::Array{Float64,2},  
                          N_r::Int64, α_reg::Float64,  N_iter::Int64 = 100)


  mesh, obs_coord, init_type = barotropic.mesh, barotropic.obs_coord, barotropic.init_type
  grid_vor0_ref, grid_vor0, spe_vor0 = barotropic.grid_vor0_ref, barotropic.grid_vor0, barotropic.spe_vor0
  ndays = barotropic.ndays
                          
  parameter_names = ["spe_ω"] 
  
  ens_func(θ_ens) = Barotropic_ensemble(θ_ens, ndays, init_type, obs_coord)
  
  trukiobj = TRUKIObj(parameter_names,
  N_r,
  θ0_bar, 
  θθ0_cov_sqr,
  t_mean, # observation
  t_cov,
  α_reg)
  
  
  for i in 1:N_iter
    
   

    update_ensemble!(trukiobj, ens_func) 

    params_i = deepcopy(trukiobj.θ_bar[end])
    
    
    Barotropic_ω0!(mesh, init_type, params_i, spe_vor0, grid_vor0)
    
    
    @info "F error of ω0 :", norm(grid_vor0_ref - grid_vor0), " / ",  norm(grid_vor0_ref)
    
    
  end
  
  return trukiobj
  
end


function Barotropic_EnKI(filter_type::String, barotropic::Barotropic_Data, t_mean::Array{Float64,1}, t_cov, θ0_bar::Array{Float64,1}, θθ0_cov::Array{Float64,2},  
  N_ens::Int64, α_reg::Float64, N_iter::Int64 = 100)
  parameter_names = ["θ"]
  
  
  
  mesh, obs_coord, init_type = barotropic.mesh, barotropic.obs_coord, barotropic.init_type
  grid_vor0_ref, grid_vor0, spe_vor0 = barotropic.grid_vor0_ref, barotropic.grid_vor0, barotropic.spe_vor0
  ndays = barotropic.ndays
  
  parameter_names = ["spe_ω"] 
  
  ens_func(θ_ens) = Barotropic_ensemble(θ_ens, ndays, init_type, obs_coord)

  ekiobj = EnKIObj(filter_type, 
  parameter_names,
  N_ens, 
  θ0_bar,
  θθ0_cov_sqr,
  t_mean, # observation
  t_cov,
  α_reg)
  
  
  for i in 1:N_iter
    
    update_ensemble!(ekiobj, ens_func) 

    params_i = deepcopy(ekiobj.θ_bar[end])
    
    
    Barotropic_ω0!(mesh, init_type, params_i, spe_vor0, grid_vor0)
    
    
    @info "F error of ω0 :", norm(grid_vor0_ref - grid_vor0), " / ",  norm(grid_vor0_ref)
    
    
  end
  
  return ekiobj
  
end



###############################################################################################
# Recover the initial vorticity field by vel_u or vorticity observations
# vor ∼ O(1/R)   vel ∼ O(1)
# 
###############################################################################################

function Compare(α_reg::Float64, noise_level::Int64)
  ndays  = 2
  mesh, dyn_data, grid_vor0, spe_vor0, obs_coord, obs_raw = Barotropic_Main(ndays, "truth")
  t_mean = convert_obs(obs_coord, obs_raw)
  
  
  barotropic = Barotropic_Data(mesh, obs_coord, grid_vor0, similar(dyn_data.grid_vor), similar(dyn_data.spe_vor_c), ndays, "truth")
  
  
  N_data = length(t_mean)
  t_cov = Array(Diagonal(fill(1.0, N_data))) 
  
  # RUKI
  begin
    barotropic.init_type = "spec_vor"
    # N = 7, n, m = 0,1, ... 7  
    N = 7 
    nθ = (N+3)*N
    θ0_bar = zeros(Float64, nθ)                              # mean 
    θθ0_cov = Array(Diagonal(fill(10.0^2, nθ)))           # standard deviation
    
    N_ite = 50 
    
    ukiobj = Barotropic_RUKI(barotropic,  t_mean, t_cov, θ0_bar, θθ0_cov, α_reg,  N_ite)
  end
  
  
end


Compare(1.0, 0)
