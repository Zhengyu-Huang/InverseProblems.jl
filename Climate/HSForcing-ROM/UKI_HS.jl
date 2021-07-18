using NNGCM
using JLD2
using Statistics
using LinearAlgebra


ROM = true
include("../../Inversion/Plot.jl")
include("HS.jl")
include("../../Inversion/KalmanInversion.jl")

function HS_run(id::Int64, params::Array{Float64, 1})
  
  mesh_size = (id == 1 && ROM ? "T42" : "T21")
  


  kt_1, kt_2, ΔT_y, Δθ_z = constraint(params)
  
  physics_params = Dict{String,Float64}("σ_b"=>0.7, "k_f" => 1.0, 
  "k_a" => (kt_1), "k_s" => (kt_2), 
  "ΔT_y" => (ΔT_y), "Δθ_z" => (Δθ_z))
  
  op_man = Atmos_Spectral_Dynamics_Main(physics_params, end_day, spinup_day, mesh_size)
  Finalize_Output!(op_man)
  
  t_zonal_mean = op_man.t_zonal_mean
  # u_zonal_mean = op_man.u_zonal_mean
  # v_zonal_mean = op_man.v_zonal_mean
  
  if mesh_size == "T21"
    t_zonal_mean_ref = zeros(Float64, 2 .* size(t_zonal_mean))
    t_zonal_mean_ref[1:2:end, 1:2:end] .= t_zonal_mean
    t_zonal_mean_ref[1:2:end, 2:2:end] .= t_zonal_mean
    t_zonal_mean_ref[2:2:end, 1:2:end] .= t_zonal_mean
    t_zonal_mean_ref[2:2:end, 2:2:end] .= t_zonal_mean
    
    return t_zonal_mean_ref[:]
    
  elseif mesh_size == "T42"

    return t_zonal_mean[:]
    
  else
    
    error("mesh size : " ,  mesh_size, " is not recognized")
    
  end
end



function HS_run_aug(id::Int64, params::Array{Float64, 1})
  
  y = HS_run(id, params)
  return [y ; params]  
end



function HS_ensemble(params_i::Array{Float64, 2},  end_day::Int64, spinup_day::Int64, N_y::Int64)
  
  N_ens,  N_θ = size(params_i)
  
  g_ens = zeros(Float64, N_ens,  N_y)
  
  Threads.@threads for i = 1:N_ens 
    # g: N_ens x N_y
    g_ens[i, :] .= HS_run_aug(i, params_i[i, :])
  end
  
  return g_ens
end


function constraint(θ_bar_raw_arr::Array{Float64, 1})
  θ_bar_arr = similar(θ_bar_raw_arr)
  θ_bar_arr[1] = exp(θ_bar_raw_arr[1])
  θ_bar_arr[2] = exp(θ_bar_raw_arr[2])
  θ_bar_arr[3] = exp(θ_bar_raw_arr[3])
  θ_bar_arr[4] = exp(θ_bar_raw_arr[4])
  
  return θ_bar_arr
end

function constraint_all(θ_bar_raw_arr::Array{Float64, 2})
  θ_bar_arr = similar(θ_bar_raw_arr)
  N = size(θ_bar_arr, 2)
  for i = 1:N
    θ_bar_arr[:, i] = constraint(θ_bar_raw_arr[:, i])
  end
  return θ_bar_arr
end

function visualize(ukiobj::UKIObj{Float64}, θ_ref::Array{Float64, 1}, file_name::String)
  
  θ_bar_raw_arr = hcat(ukiobj.θ_bar...)
  θ_bar_arr = constraint_all(θ_bar_raw_arr)
  
  n_θ, N_ite = size(θ_bar_arr,1), size(θ_bar_arr,2)-1
  ites = Array(LinRange(1, N_ite+1, N_ite+1))
  
  parameter_names = ukiobj.unames
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



function HS_UKI(t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, θ_bar::Array{Float64,1}, θθ_cov::Array{Float64,2}, 
  end_day::Int64, spinup_day::Int64, N_y::Int64, α_reg::Float64, 
  N_iter::Int64 = 100)
  
  parameter_names = ["ka", "ks", "ΔTy", "Δθz"]
  
  ens_func(θ_ens) = HS_ensemble(θ_ens, end_day, spinup_day, N_y)
  
  ukiobj = UKIObj(parameter_names,
  θ_bar, 
  θθ_cov,
  t_mean, # observation
  t_cov,
  α_reg)
  
  θ_ref = [1.0/40.0; 1.0/4.0; 60.0; 10.0]
  for i in 1:N_iter
    
    params_i = deepcopy(ukiobj.θ_bar[end])
    
    @info "At iter ", i, " params_i : ", params_i
    @info "At iter ", i, " params_i : ", ukiobj.θθ_cov[end]
    
    update_ensemble!(ukiobj, ens_func) 
    
    
    if i%10 == 0
      @save "UKI_Obj_Ite_"*string(i)*".dat" ukiobj
      visualize(ukiobj, θ_ref, "UKI_Obj_Ite_"*string(i)*".pdf") 
    end
  end
  
  @info "θ is ", ukiobj.θ_bar[end], " θ_ref is ",  
  
  return ukiobj
  
end

function Comp_Mean_Cov(ση)
  @load "HS_OpM.dat" output_manager
  
  
  n_day = output_manager.n_day
  spinup_day = output_manager.spinup_day
  nθ, nd = output_manager.nθ, output_manager.nd
  
  
  t_daily_zonal_mean, u_daily_zonal_mean, v_daily_zonal_mean = output_manager.t_daily_zonal_mean, output_manager.u_daily_zonal_mean, output_manager.v_daily_zonal_mean
  
  
  # Currently, I have 1200 day data, spinup first 200 days
  n_data = nθ*nd
  obs = reshape(t_daily_zonal_mean, (n_data, n_day))
  n_obs = 200
  n_obs_box = Int64((n_day - spinup_day)/n_obs)
  obs_box = zeros(Float64, n_data, n_obs_box)
  
  for i = 1:n_obs_box
    n_obs_start = spinup_day + (i - 1)*n_obs + 1
    obs_box[:, i] = mean(obs[:, n_obs_start : n_obs_start + n_obs - 1], dims = 2)
  end
  
  
  obs_mean = dropdims(mean(obs_box, dims=2), dims = 2)
  
  Random.seed!(123);
  noise = rand(Normal(0, ση), length(obs_mean))
  return obs_mean + noise
end

###############################################################################################
# end_day = 1200
# spinup_day = 200

end_day = 400
spinup_day = 200

ση = 1.0
t_mean = Comp_Mean_Cov(ση)
N_y = length(t_mean)

# error("t_cov")
t_cov = Array(Diagonal(fill(ση^2, N_y))) 
# initial distribution is 
# θ0_bar = [1.0; 1.0/40.0; 1.0/4.0-1.0/40.0; 60.0; 10.0]

θ0_bar = [0.0; 0.0; 0.0 ; 0.0]                                  # mean 
θθ0_cov = Array(Diagonal([1.0^2; 1.0^2; 1.0^2; 1.0^2]))           # standard deviation

N_ite = 20
α_reg = 1.0

t_mean_aug = [t_mean ; θ0_bar]
t_cov_aug = [t_cov zeros(Float64, N_y, N_θ); zeros(Float64, N_θ, N_y)  θθ0_cov]
    

ukiobj = HS_UKI(t_mean_aug, t_cov_aug, θ0_bar, θθ0_cov,  end_day, spinup_day, length(t_mean_aug), α_reg, N_ite)






