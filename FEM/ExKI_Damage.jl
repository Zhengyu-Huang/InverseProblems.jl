using NNFEM
using JLD2
using Statistics
using LinearAlgebra



include("../Plot.jl")
include("Damage.jl")
include("../RExKI.jl")


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



function ExKI(phys_params::Params, 
  t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
  θ_bar::Array{Float64,1}, θθ_cov::Array{Float64,2}, 
  α_reg::Float64, 
  θ_dam_ref::Array{Float64,1}, N_iter::Int64 = 100)


  parameter_names = ["E"]
  
  ens_func(θ_ens) = Ensemble(phys_params, θ_ens)
  
  exkiobj = ExKIObj(parameter_names,
  θ_bar, 
  θθ_cov,
  t_mean, # observation
  t_cov,
  α_reg)
  
  
  for i in 1:N_iter
    
    params_i = deepcopy(exkiobj.θ_bar[end])

    θ_dam = Interp_θ(phys_params.domain_c, phys_params.interp_e, phys_params.interp_sdata, params_i)

    @info "θ error :", norm(θ_dam_ref - θ_dam), " / ",  norm(θ_dam_ref)
    
    update_ensemble!(exkiobj, ens_func) 
    
    @info "data_mismatch :", (exkiobj.g_bar[end] - exkiobj.g_t)'*(exkiobj.obs_cov\(exkiobj.g_bar[end] - exkiobj.g_t))
    

    # visulize
    if i%10 == 0
      Run_Damage(phys_params, params_i,  "Figs/exki."*string(i)*".disp", "Figs/exki."*string(i)*".E")
      
      exkiobj_θ_bar, exkiobj_θθ_cov, exkiobj_g_bar = exkiobj.θ_bar, exkiobj.θθ_cov, exkiobj.g_bar
      exkiobj_θθ_cov = [diag(exkiobj.θθ_cov[i]) for i=1:length(exkiobj.θθ_cov)]

      @save "exkiobj.dat" exkiobj_θ_bar exkiobj_g_bar exkiobj_θθ_cov
    end
    
  end
  
  return exkiobj
end


###############################################################################################
ns, ns_obs, porder, problem, ns_c, porder_c = 4, 3, 2, "Static", 2, 2
phys_params = Params(ns, ns_obs, porder, problem, ns_c, porder_c)

# data
noise_level = -1.0
θ_dam, t_mean =  Run_Damage(phys_params, "Analytical", nothing,  "Figs/disp", "Figs/E", noise_level)

t_cov = Array(Diagonal(fill(0.01, length(t_mean)))) 


nθ = size(phys_params.domain_c.nodes, 1)
θ0_bar = zeros(Float64, nθ)
θθ0_cov = Array(Diagonal(fill(1.0, nθ)))           # standard deviation

####



N_iter = 100 

α_reg = 1.0
exkiobj = ExKI(phys_params,
t_mean, t_cov, 
θ0_bar, θθ0_cov, 
α_reg,
θ_dam_ref,
N_iter)






