using NCDatasets
using Statistics
using Interpolations
using LinearAlgebra
using JLD
include("../../Inversion/KI.jl")




α_reg = 1.0
update_freq = 1
sigma_points_type = "modified-2n+1"
Δt = 0.5
augment = true

include("helper_funcs_fms.jl")




N_ens = 2N_params  + 1
function constraint(u::Float64, u_low, u_up)
    if(isnothing(u_low) && isnothing(u_up))
    	return u
    elseif (isnothing(u_low)  &&  !isnothing(u_up))
        return u_up - exp(u)
    elseif (!isnothing(u_low)  &&  isnothing(u_up))	
        return u_low + exp(u)
    else
        return u_low + (u_up - u_low)/(1 + exp(u))
    end
end

function dconstraint(u::Float64, u_low, u_up)
    if(isnothing(u_low) && isnothing(u_up))
        return 1
    elseif (isnothing(u_low)  &&  !isnothing(u_up))
    	return -exp(u)
    elseif (!isnothing(u_low)  &&  isnothing(u_up))
        return exp(u)
    else
        return  -(u_up - u_low)*exp(u)/(1 + exp(u))^2
    end
end

function constraint(u::Array{Float64, 1})
  constraint_u = similar(u)
  for (i, u_i) in enumerate(u)
      constraint_u[i] = constraint(u_i, param_bounds[i][1], param_bounds[i][2])
  end
  return constraint_u
end

function constraint(u_mean::Array{Float64, 1}, uu_cov::Array{Float64, 2})
  N_params = length(u_mean)
  dc = zeros(N_params, N_params)
  for i = 1:N_params
      dc[i,i] = dconstraint(u_mean[i], param_bounds[i][1], param_bounds[i][2])
  end
  
  constraint_uu_cov = dc * uu_cov * dc'
  return constraint_uu_cov
end

function constraint(u_ens::Array{Float64, 2})
  constraint_u_ens = similar(u_ens)
  N_ens, N_params = size(u_ens)

  for i = 1:N_ens
      constraint_u_ens[i,:] = constraint(u_ens[i,:])
  end

  return constraint_u_ens
end


# save mean and covariance and u_ens
function save_params(ukiobj, iteration_::Int64)
    mean, cov  = ukiobj.θ_mean[end], ukiobj.θθ_cov[end]
    θ_ens      = construct_sigma_ensemble(ukiobj, mean, cov)
    θ_p_ens, _ = prediction_ensemble(ukiobj)  
    
    save(params_prefix*string(iteration_)*".jld", "mean", mean, "cov", cov, "θ_ens", θ_ens, "θ_p_ens", θ_p_ens)	 
end


function read_params( iteration_::Int64)
  
   data = load(params_prefix*string(iteration_)*".jld")
   
   return data["mean"], data["cov"], data["θ_ens"], data["θ_p_ens"]

end


function read_all_params(N_iterations::Int64)
    constraint_u_mean_all = zeros(N_iterations + 1, N_params)
    constraint_uu_cov_all = zeros(N_iterations + 1, N_params, N_params)

    for iteration_ = 0:N_iterations
        data = load(params_prefix*string(iteration_)*".jld")
	u_mean, uu_cov = data["mean"], data["cov"]
 	
	constraint_u_mean_all[iteration_+1, :] = constraint(u_mean)
        constraint_uu_cov_all[iteration_+1, :, :] = constraint(u_mean, uu_cov)
	
    end

    return constraint_u_mean_all, constraint_uu_cov_all
end



