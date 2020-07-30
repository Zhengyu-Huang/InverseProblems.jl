using NNFEM
using JLD2
using Statistics
using LinearAlgebra



include("../Plot.jl")
include("Damage.jl")
include("../REKI.jl")

function Foward(phys_params::Params, θ_c::Array{Float64,1})

  _, data = Run_Damage(phys_params, "Piecewise", θ_c)
  return data
end

function Ensemble(phys_params::Params,  params_i::Array{Float64, 2})
  n_data = phys_params.n_data
  
  N_ens,  N_θ = size(params_i)
  
  g_ens = zeros(Float64, N_ens,  n_data)
  
  Threads.@threads for i = 1:N_ens 
    # g: N_ens x N_data
    g_ens[i, :] .= Foward(phys_params, params_i[i, :])
  end

  
  return g_ens
end


function EKI(phys_params::Params,
  N_ens::Int64,
  t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
  θ0_bar::Array{Float64,1}, θθ_cov::Array{Float64,2}, 
  α_reg::Float64, 
  θ_dam_ref::Array{Float64,1}, N_iter::Int64 = 100)
  
  
  
  parameter_names = ["E"]
  
  ens_func(θ_ens) = Ensemble(phys_params, θ_ens)
  
  nθ = length(θ0_bar)
  priors = [Distributions.Normal(θ0_bar[i], sqrt(θθ0_cov[i,i])) for i=1:nθ]
  
  θ0 = construct_initial_ensemble(N_ens, priors; rng_seed=42)
  
  ekiobj = EKIObj(parameter_names,
  θ0,
  θ0_bar, 
  θθ_cov,
  t_mean, # observation
  t_cov,
  α_reg)
  
  
  for i in 1:N_iter
    
    params_i = dropdims(mean(ekiobj.θ[end], dims=1), dims=1) 
    
    θ_dam = Get_θ_Dam_From_Raw(phys_params.domain_c, phys_params.interp_e, phys_params.interp_sdata, params_i)
    
    @info "θ error :", norm(θ_dam_ref - θ_dam), " / ",  norm(θ_dam_ref)
        
    update_ensemble!(ekiobj, ens_func) 
    
    @info "data_mismatch :", (ekiobj.g_bar[end] - ekiobj.g_t)'*(ekiobj.obs_cov\(ekiobj.g_bar[end] - ekiobj.g_t))
    
    # visulize
    if i%10 == 0
      Run_Damage(phys_params, "Piecewise", params_i,  "Figs/eki."*string(i)*".disp", "Figs/eki."*string(i)*".E")

      ekiobj_θ, ekiobj_g_bar = ekiobj.θ, ekiobj.g_bar

      @save "ekiobj.dat" ekiobj_θ ekiobj_g_bar
    end
    
  end
  
  return ekiobj
end


###############################################################################################
ns, ns_obs, porder, problem, ns_c, porder_c = 8, 5, 2, "Static", 2, 2
phys_params = Params(ns, ns_obs, porder, problem, ns_c, porder_c)

# data
noise_level = -1.0
θ_dam_ref, t_mean =  Run_Damage(phys_params, "Analytic", nothing,  "Figs/disp-high", "Figs/E-high", noise_level)


t_cov = Array(Diagonal(fill(0.01, length(t_mean)))) 

nθ = size(phys_params.domain_c.nodes, 1)
θ0_bar = zeros(Float64, nθ)
θθ0_cov = Array(Diagonal(fill(1.0, nθ)))           # standard deviation




N_iter = 100 

α_reg = 1.0
N_ens = 251
ekiobj = EKI(phys_params,
N_ens,
t_mean, t_cov, 
θ0_bar, θθ0_cov, 
α_reg,
θ_dam_ref,
N_iter)








