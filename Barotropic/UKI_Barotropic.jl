using NNGCM
using JLD2
using Statistics
using LinearAlgebra



include("../Plot.jl")
include("Barotropic.jl")
include("RUKI.jl")

function Barotropic_run(init_type::String, init_data=nothing)
  
  _, obs_raw = Barotropic_Main(init_type, init_data)
  
  # update obs
  
  return obs
end

function Barotropic_ensemble(params_i::Array{Float64, 2},  N_data::Int64)
  
  N_ens,  N_θ = size(params_i)
  
  g_ens = zeros(Float64, N_ens,  N_data)
  
  Threads.@threads for i = 1:N_ens 
    # g: N_ens x N_data
    g_ens[i, :] .= forward_run(params_i[i, :])
  end
  
  return g_ens
end



function visualize(uki::UKIObj{Float64}, θ_ref::Array{Float64, 1}, file_name::String)
  
  θ_bar_raw_arr = hcat(uki.θ_bar...)
  θ_bar_arr = constraint_trans(θ_bar_raw_arr)
  
  n_θ, N_ite = size(θ_bar_arr,1), size(θ_bar_arr,2)-1
  ites = Array(LinRange(1, N_ite+1, N_ite+1))
  
  parameter_names = uki.unames
  for i_θ = 1:n_θ
    plot(ites, θ_bar_arr[i_θ,:], "--o", fillstyle="none", label=parameter_names[i_θ])
    plot(ites, fill(θ_ref[i_θ], N_ite+1), "--", color="gray")
  end
  
  xlabel("Iterations")
  legend()
  grid("on")
  tight_layout()
  savefig(file_name)
  close("all")
  
end



function Barotropic_RUKI(t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, θ_bar::Array{Float64,1}, θθ_cov::Array{Float64,2}, 
  N_data::Int64, α_reg::Float64 = 1.0, N_iter::Int64 = 100)
  
  parameter_names = ["spe_ω"] 
  
  ens_func(θ_ens) = Barotropic_ensemble(θ_ens, N_data)
  
  ukiobj = UKIObj(parameter_names,
  θ_bar, 
  θθ_cov,
  t_mean, # observation
  t_cov,
  α_reg)
  
  for i in 1:N_iter
    
    update_ensemble!(rukiobj, ens_func) 
    
  end
  
  return rukiobj
  
end


###############################################################################################

function Compare(α_reg::Float64, noise_level::Int64)

  mesh, t_mean = Barotropic_run("truth")
  N_data = length(t_mean)
  t_cov = Array(Diagonal(fill(1.0, N_data))) 
  
  begin
  # N = 7, n, m = 0,1, ... 7  
  N = 7 
  nθ = (N+3)*N
  θ0_bar = zeros(Float64, nθ)                              # mean 
  θθ0_cov = Array(Diagonal(fill(10.0^2, nθ)))           # standard deviation
  
  N_ite = 50 
  
  ukiobj = Barotropic_RUKI(t_mean, t_cov, θ0_bar, θθ0_cov, α_reg, N_data, N_ite)
  end


  
end



