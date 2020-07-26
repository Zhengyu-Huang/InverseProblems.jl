using NNFEM
using JLD2
using Statistics
using LinearAlgebra



include("../Plot.jl")
include("Damage.jl")
include("../RUKI.jl")


function Foward(phys_params::Params, θ::Array{Float64,1})

  _, data = Run_Damage(phys_params, θ)
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



function UKI(phys_params::Params, 
  t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
  θ_bar::Array{Float64,1}, θθ_cov::Array{Float64,2}, 
  α_reg::Float64, 
  θ_dam_ref::Array{Float64,1}, N_iter::Int64 = 100)


  parameter_names = ["E"]
  
  ens_func(θ_ens) = Ensemble(phys_params, θ_ens)
  
  ukiobj = UKIObj(parameter_names,
  θ_bar, 
  θθ_cov,
  t_mean, # observation
  t_cov,
  α_reg)
  
  
  for i in 1:N_iter
    
    params_i = deepcopy(ukiobj.θ_bar[end])

    θ_dam = Get_θ_Dam(params_i)

    @info "θ error :", norm(θ_dam_ref - θ_dam), " / ",  norm(θ_dam_ref)
    
    update_ensemble!(ukiobj, ens_func) 
    
    @info "data_mismatch :", (ukiobj.g_bar[end] - ukiobj.g_t)'*(ukiobj.obs_cov\(ukiobj.g_bar[end] - ukiobj.g_t))
    

    # visulize
    if i%1 == 0
      Run_Damage(phys_params, params_i,  "uki.disp", "uki.E")
      
      ukiobj_θ_bar, ukiobj_θθ_cov, ukiobj_g_bar = ukiobj.θ_bar, ukiobj.θθ_cov, ukiobj.g_bar
      @save "ukiobj.dat" ukiobj_θ_bar ukiobj_g_bar
    end
    
  end
  
  return ukiobj
end


###############################################################################################
phys_params = Params()

# data
noise_level = -1.0
θ_dam_ref, t_mean =  Run_Damage(phys_params, nothing,  "disp", "E", noise_level)


t_cov = Array(Diagonal(fill(1.0, length(t_mean)))) 

nθ = length(θ_dam_ref)
θ0_bar = zeros(Float64, nθ)
θθ0_cov = Array(Diagonal(fill(1.0, nθ)))           # standard deviation

####



N_iter = 50 

α_reg = 1.0
ukiobj = UKI(phys_params,
t_mean, t_cov, 
θ0_bar, θθ0_cov, 
α_reg,
θ_dam_ref,
N_iter)






