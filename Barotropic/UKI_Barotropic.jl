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


function Barotropic_run(nframes::Int64, init_type::String, init_data=nothing, obs_coord=nothing, spe_vor_b=nothing)
  
  _, _, _, _, _, _, obs_raw = Barotropic_Main(nframes, init_type, init_data, spe_vor_b, obs_coord)
  
  # update obs
  obs = convert_obs(obs_coord, obs_raw)
  
  return obs
end

function Barotropic_ensemble(params_i::Array{Float64, 2},  nframes::Int64, init_type::String,  obs_coord::Array{Int64, 2}, spe_vor_b=nothing)
  
  N_ens,  N_θ = size(params_i)
  N_data = size(obs_coord, 1) * nframes
  
  g_ens = zeros(Float64, N_ens,  N_data)
  
  Threads.@threads for i = 1:N_ens 
    # g: N_ens x N_data
    g_ens[i, :] .= Barotropic_run(nframes, init_type, params_i[i, :], obs_coord, spe_vor_b)
  end
  
  
  return g_ens
end




function Barotropic_RUKI(barotropic::Barotropic_Data, t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, θ0_bar::Array{Float64,1}, spe_vor_b::Array{ComplexF64,3}, θθ0_cov::Array{Float64,2}, 
  α_reg::Float64 = 1.0, N_iter::Int64 = 100, θ_basis = nothing)
  
  mesh, obs_coord, obs_data, init_type = barotropic.mesh, barotropic.obs_coord, barotropic.obs_data, barotropic.init_type
  grid_vor0_ref, grid_vor0, spe_vor0 = barotropic.grid_vor0_ref, barotropic.grid_vor0, barotropic.spe_vor0
  nframes = barotropic.nframes
  
  parameter_names = ["spe_ω"] 
  
  ens_func(θ_ens) = Barotropic_ensemble(θ_ens, nframes, init_type, obs_coord, spe_vor_b)
  
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

function Barotropic_TRUKI(barotropic::Barotropic_Data, t_mean::Array{Float64,1}, t_cov, θ0_bar::Array{Float64,1}, θθ0_cov_sqr::Array{Float64,2},  
  N_r::Int64, α_reg::Float64,  N_iter::Int64 = 100)
  
  
  mesh, obs_coord, obs_data, init_type = barotropic.mesh, barotropic.obs_coord, barotropic.obs_data, barotropic.init_type
  grid_vor0_ref, grid_vor0, spe_vor0 = barotropic.grid_vor0_ref, barotropic.grid_vor0, barotropic.spe_vor0
  nframes = barotropic.nframes
  
  parameter_names = ["spe_ω"] 
  
  ens_func(θ_ens) = Barotropic_ensemble(θ_ens, nframes, init_type, obs_coord)
  
  trukiobj = TRUKIObj(parameter_names,
  N_r,
  θ0_bar, 
  θθ0_cov_sqr,
  t_mean, # observation
  t_cov,
  α_reg)
  
  @info "start"
  for i in 1:N_iter
    
    if i==1
      Barotropic_ω0!(mesh, init_type, trukiobj.θ_bar[end], spe_vor0, grid_vor0)
      @info "i0: error of ω0 :", norm(grid_vor0_ref - grid_vor0), " / ",  norm(grid_vor0_ref)
    end
    
    update_ensemble!(trukiobj, ens_func) 
    
    params_i = deepcopy(trukiobj.θ_bar[end])
    
    
    Barotropic_ω0!(mesh, init_type, params_i, spe_vor0, grid_vor0)
    
    
    @info "F error of ω0 :", norm(grid_vor0_ref - grid_vor0), " / ",  norm(grid_vor0_ref)
    
    @info "optimization error :", norm(obs_data - trukiobj.g_bar[end]), " / ",  norm(obs_data)
    
    if i%10 == 0
      Lat_Lon_Pcolormesh(mesh, grid_vor0, 1; save_file_name = "Figs/TRUKI_Barotropic_vor_"*string(i)*".png", vmax = vor0_max, vmin = vor0_vmin, cmap = "jet")
      
      
      begin # compute standard deviation on the diagonal
        Z_vor0 = trukiobj.θθ_cov_sqr[end]
        d_Cov_0 = similar(grid_vor0)
        for j = 1:length(grid_vor0)
          d_Cov_0[j] = sqrt( sum(Z_vor0[j, :].^2))
        end
        Lat_Lon_Pcolormesh(mesh, d_Cov_0, 1;  save_file_name = "Figs/TRUKI_Barotropic_vor_std"*string(i)*".png", cmap = "jet")
      end
      
    end
    
  end
  
  return trukiobj
  
end


function Barotropic_EnKI(filter_type::String, barotropic::Barotropic_Data, t_mean::Array{Float64,1}, t_cov, θ0_bar::Array{Float64,1}, θθ0_cov_sqr::Array{Float64,2},  
  N_ens::Int64, α_reg::Float64, N_iter::Int64 = 100)
  parameter_names = ["θ"]
  
  
  
  mesh, obs_coord, obs_data, init_type = barotropic.mesh, barotropic.obs_coord, barotropic.obs_data, barotropic.init_type
  grid_vor0_ref, grid_vor0, spe_vor0 = barotropic.grid_vor0_ref, barotropic.grid_vor0, barotropic.spe_vor0
  nframes = barotropic.nframes
  
  parameter_names = ["spe_ω"] 
  
  ens_func(θ_ens) = Barotropic_ensemble(θ_ens, nframes, init_type, obs_coord)
  
  ekiobj = EnKIObj(filter_type, 
  parameter_names,
  N_ens, 
  θ0_bar,
  θθ0_cov_sqr,
  t_mean, # observation
  t_cov,
  α_reg)
  
  
  for i in 1:N_iter
    
    if i==1
      Barotropic_ω0!(mesh, init_type, dropdims(mean(ekiobj.θ[end], dims=1), dims=1), spe_vor0, grid_vor0)
      @info "i0: error of ω0 :", norm(grid_vor0_ref - grid_vor0), " / ",  norm(grid_vor0_ref)
    end
    
    
    update_ensemble!(ekiobj, ens_func) 
    
    params_i = deepcopy(dropdims(mean(ekiobj.θ[end], dims=1), dims=1))
    
    
    Barotropic_ω0!(mesh, init_type, params_i, spe_vor0, grid_vor0)
    
    
    @info "F error of ω0 :", norm(grid_vor0_ref - grid_vor0), " / ",  norm(grid_vor0_ref)
    
    @info "optimization error :", norm(obs_data - ekiobj.g_bar[end]), " / ",  norm(obs_data)
    if i%10 == 0
      Lat_Lon_Pcolormesh(mesh, grid_vor0, 1;  save_file_name = "Figs/"*filter_type*"_Barotropic_vor_"*string(i)*".png", vmax = vor0_max, vmin = vor0_vmin, cmap = "jet")
      
    end
    
  end
  
  return ekiobj
  
end



###############################################################################################
# Recover the initial vorticity field by vel_u or vorticity observations
# vor ∼ O(1/R)   vel ∼ O(1)
# 
###############################################################################################

function Compare(α_reg::Float64, noise_level::Int64, EnKI_run::Bool = false)
  
  
  
  
  
  @info "α_reg::Float64, noise_level::Int64 : ", α_reg, noise_level
  nframes  = 2
  mesh,  grid_vor_b, spe_vor_b, grid_vor0, spe_vor0, obs_coord, obs_data = Barotropic_Main(nframes, "truth")
  obs_data = convert_obs(obs_coord, obs_data)
  
  
  barotropic = Barotropic_Data(mesh, obs_coord, obs_data, grid_vor0, similar(grid_vor0), similar(spe_vor0), nframes, "truth")
  
  t_mean = obs_data
  N_data = length(t_mean)
  t_cov = Array(Diagonal(fill(1.0, N_data))) 
  
  radius = 6371.2e3  
  N_ite = 20
  N = 7
  
  
  
  
  
  begin
    barotropic.init_type = "grid_vor"
    # N = 7, n, m = 0,1, ... 7  
    N_r = (N+3)*N
    N_ens = 2N_r + 1
    θ0_bar = copy(grid_vor_b[:])                      # mean 
    nθ = length( θ0_bar )
    
    Z0_cov = ones(Float64, nθ, N_r)
    
    spe_θ0 = similar(spe_vor0)
    grid_θ0 = similar(grid_vor0)
    
    for n = 1:N
      for m = 0:n
        i = Int64((n+2)*(n-1)/2) + m + 1
        
        spe_θ0 .= 0.0
        spe_θ0[m+1,n+1] = 1.0/radius
        Trans_Spherical_To_Grid!(mesh, spe_θ0, grid_θ0)
        Z0_cov[:, 2i-1] .= grid_θ0[:]
        
        spe_θ0 .= 0.0
        spe_θ0[m+1,n+1] = 1.0/radius *im
        Trans_Spherical_To_Grid!(mesh, spe_θ0, grid_θ0)
        Z0_cov[:, 2i] .= grid_θ0[:]
      end
    end
    
    
    
    
    trukiobj = Barotropic_TRUKI(barotropic,  t_mean, t_cov, θ0_bar, Z0_cov, N_r, α_reg,  N_ite)
    
    if EnKI_run
      Z0_cov ./= N_r
      
      enkiobj = Barotropic_EnKI("EnKI", barotropic,  t_mean, t_cov, θ0_bar, Z0_cov, 2N_r+1, α_reg,  N_ite)
      eakiobj = Barotropic_EnKI("EAKI", barotropic,  t_mean, t_cov, θ0_bar, Z0_cov, 2N_r+1, α_reg,  N_ite)
      etkiobj = Barotropic_EnKI("ETKI", barotropic,  t_mean, t_cov, θ0_bar, Z0_cov, 2N_r+1, α_reg,  N_ite)
    end
    
  end

  # RUKI
  begin
    barotropic.init_type = "spec_vor"
    # N = 7, n, m = 0,1, ... 7  
    nθ = (N+3)*N
    θ0_bar = zeros(Float64, nθ)                              # mean 
    
    # for n = 1:N
    #   for m = 0:n
    #     i_init_data = Int64((n+2)*(n-1)/2) + m + 1
    #     θ0_bar[2*i_init_data-1],  θ0_bar[2*i_init_data] = real(spe_vor_b[m+1,n+1])*radius,  imag(spe_vor_b[m+1,n+1])*radius
    #   end
    # end
    
    θθ0_cov = Array(Diagonal(fill(1.0^2, nθ)))           # standard deviation
    
    rukiobj = Barotropic_RUKI(barotropic,  t_mean, t_cov, θ0_bar, spe_vor_b, θθ0_cov, α_reg,  N_ite, Z0_cov)
    
  end

  
  
  
  errors = zeros(Float64, 5, N_ite, 2)
  
  spe_vor_i = similar(spe_vor0)
  grid_vor_i = similar(grid_vor0)
  
  for i = 1:N_ite
    # trukiobj
    params_i = deepcopy(trukiobj.θ_bar[i])
    Barotropic_ω0!(mesh, "grid_vor", params_i, spe_vor_i, grid_vor_i)
    errors[1, i, 1] = norm(grid_vor_i - grid_vor0)  /   norm(grid_vor0)
    errors[1, i, 2] = norm(obs_data - trukiobj.g_bar[i])   /  norm(obs_data)
    
    # rukiobj
    params_i = deepcopy(rukiobj.θ_bar[i])
    Barotropic_ω0!(mesh, "spec_vor", params_i, spe_vor_i, grid_vor_i, spe_vor_b = spe_vor_b)
    errors[2, i, 1] = norm(grid_vor_i - grid_vor0)  /   norm(grid_vor0)
    errors[2, i, 2] = norm(obs_data - rukiobj.g_bar[i])   /  norm(obs_data)
    
    
    if EnKI_run
      # enki
      params_i = deepcopy(dropdims(mean(enkiobj.θ[i], dims=1), dims=1))
      Barotropic_ω0!(mesh, "grid_vor", params_i, spe_vor_i, grid_vor_i)
      errors[3, i, 1] = norm(grid_vor_i - grid_vor0)  /   norm(grid_vor0)
      errors[3, i, 2] = norm(obs_data - enkiobj.g_bar[i])   /  norm(obs_data)
      
      # eaki
      params_i = deepcopy(dropdims(mean(eakiobj.θ[i], dims=1), dims=1))
      Barotropic_ω0!(mesh, "grid_vor", params_i, spe_vor_i, grid_vor_i)
      errors[4, i, 1] = norm(grid_vor_i - grid_vor0)  /   norm(grid_vor0)
      errors[4, i, 2] = norm(obs_data - eakiobj.g_bar[i])   /  norm(obs_data)
      
      
      # etki
      params_i = deepcopy(dropdims(mean(etkiobj.θ[i], dims=1), dims=1))
      Barotropic_ω0!(mesh, "grid_vor", params_i, spe_vor_i, grid_vor_i)
      errors[5, i, 1] = norm(grid_vor_i - grid_vor0)  /   norm(grid_vor0)
      errors[5, i, 2] = norm(obs_data - etkiobj.g_bar[i])   /  norm(obs_data)
      
    end
    
  end
  
  
  rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
  mysize = 24
  font0 = Dict(
  "font.size" => mysize,
  "axes.labelsize" => mysize,
  "xtick.labelsize" => mysize,
  "ytick.labelsize" => mysize,
  "legend.fontsize" => mysize,
  )
  merge!(rcParams, font0)

  fig, (ax1, ax2) = PyPlot.subplots(ncols=2, figsize=(20,8))
  
  labels = ["TUKI", "UKI", "EnKI", "EAKI", "ETKI"]
  linestyles = ["-", "-.", ":", "--", ":"]
  markers = ["o", "^", "h", "s", "d"]
  colors = ["C1", "C2", "C3", "C4", "C5"]
  ites = Array(0:N_ite-1)
  for i in (EnKI_run ? [1,2,3,4,5] : [1,2])
    ax1.plot(ites, errors[i, :, 1], linestyle=linestyles[i], marker=markers[i], color = colors[i], fillstyle="none", markevery=5, label= labels[i])
    ax2.plot(ites, errors[i, :, 2], linestyle=linestyles[i], marker=markers[i], color = colors[i], fillstyle="none", markevery=5, label= labels[i])
  end
  
  
  ax1.set_xlabel("Iterations")
  ax1.set_ylabel("Relative L₂ norm error")
  ax1.grid(true)
  
  ax2.set_xlabel("Iterations")
  ax2.set_ylabel("Optimization error")
  ax2.grid(true)
  ax2.legend()
  
  fig.tight_layout()
  
  fig.savefig("Figs/Barotropic.pdf")
  close(fig)
  
  
end


Compare(0.5, 0, false)
