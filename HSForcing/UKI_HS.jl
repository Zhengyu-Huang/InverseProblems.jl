using NNGCM
using JLD2
using Statistics
using LinearAlgebra



include("../Plot.jl")
include("HS.jl")
include("UKI.jl")

function HS_run(params::Array{Float64, 1})
  
  kt_1 = params[1]
  kt_2 = params[2]
  ΔT_y = params[3]
  Δθ_z = params[4]

  physics_params = Dict{String,Float64}("σ_b"=>0.7, "k_f" => 1.0, 
  "k_a" => 1.0/(1+abs(kt_1)), "k_s" => 1.0/(1+abs(kt_1)) + 1.0/(1+abs(kt_2)), 
  "ΔT_y" => abs(ΔT_y), "Δθ_z" => abs(Δθ_z))
  
  op_man = Atmos_Spectral_Dynamics_Main(physics_params, end_day, spinup_day)
  Finalize_Output!(op_man)
  
  t_zonal_mean = op_man.t_zonal_mean
  # u_zonal_mean = op_man.u_zonal_mean
  # v_zonal_mean = op_man.v_zonal_mean
  
  return t_zonal_mean[:]
end

function HS_ensemble(params_i::Array{Float64, 2},  end_day::Int64, spinup_day::Int64, N_data::Int64)
  
  N_ens,  N_θ = size(params_i)
  
  g_ens = zeros(Float64, N_ens,  N_data)
  
  Threads.@threads for i = 1:N_ens 
    # g: N_ens x N_data
    g_ens[i, :] .= HS_run(params_i[i, :])
  end
  
  return g_ens
end

function constraint_trans(θ_bar_raw_arr::Array{Float64, 2})
  θ_bar_arr = similar(θ_bar_raw_arr)
  θ_bar_arr[1, :] = 1.0 ./ (1 .+ abs.(θ_bar_raw_arr[1, :]))
  θ_bar_arr[2, :] = 1.0 ./ (1 .+ abs.(θ_bar_raw_arr[1, :])) + 1.0 ./ (1 .+ abs.(θ_bar_raw_arr[2, :]))
  θ_bar_arr[3, :] = θ_bar_raw_arr[3, :]
  θ_bar_arr[4, :] = θ_bar_raw_arr[4, :]

  return θ_bar_arr
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



function HS_UKI(t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, θ_bar::Array{Float64,1}, θθ_cov::Array{Float64,2}, 
  end_day::Int64, spinup_day::Int64, N_data::Int64,
  N_iter::Int64 = 100, update_cov::Int64 = 0)
  
  parameter_names = ["ka", "ks", "ΔTy", "Δθz"]

  ens_func(θ_ens) = HS_ensemble(θ_ens, end_day, spinup_day, N_data)
  
  ukiobj = UKIObj(parameter_names,
  θ_bar, 
  θθ_cov,
  t_mean, # observation
  t_cov)
  
  θ_ref = [1.0/40.0; 1.0/4.0; 60.0; 10.0]
  for i in 1:N_iter
    
    params_i = deepcopy(ukiobj.θ_bar[end])
    
    @info "At iter ", i, " params_i : ", params_i
    @info "At iter ", i, " params_i : ", ukiobj.θθ_cov[end]
    
    update_ensemble!(ukiobj, ens_func) 
    
    if (update_cov) > 0 && (i%update_cov == 1) 
      reset_θθ0_cov!(ukiobj)
    end
    
    if i%10 == 0
      @save "UKI_Obj_Ite_"*string(i)*".dat" ukiobj
      visualize(ukiobj, θ_ref, "UKI_Obj_Ite_"*string(i)*".pdf") 
    end
  end
  
  @info "θ is ", ukiobj.θ_bar[end], " θ_ref is ",  
  
  return ukiobj
  
end

function Comp_Mean_Cov()
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
  obs_cov = zeros(Float64, n_data, n_data)
  for i = 1:n_obs_box
    obs_cov += (obs_box[:,i] - obs_mean) *(obs_box[:,i] - obs_mean)'
  end
  obs_cov ./= (n_obs_box - 1)
  return obs_mean, obs_cov
end

###############################################################################################
# end_day = 1200
# spinup_day = 200

end_day = 400
spinup_day = 200



t_mean, t_cov = Comp_Mean_Cov()
N_data = length(t_mean)

# error("t_cov")
t_cov = Array(Diagonal(fill(10.0, N_data))) 
# initial distribution is 
# θ0_bar = [1.0; 1.0/40.0; 1.0/4.0-1.0/40.0; 60.0; 10.0]

θ0_bar = [2.0; 2.0; 20.0 ; 20.0]                                  # mean 
θθ0_cov = Array(Diagonal([1.0^2; 1.0^2; 1.0^2; 1.0^2]))           # standard deviation

N_ite = 50 
update_cov = 0

ukiobj = HS_UKI(t_mean, t_cov, θ0_bar, θθ0_cov,  end_day, spinup_day, N_data, N_ite, update_cov)






