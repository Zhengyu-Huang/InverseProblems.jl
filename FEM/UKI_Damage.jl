using NNFEM
using JLD2
using Statistics
using LinearAlgebra



include("../Plot.jl")
include("Random_Init.jl")
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
  θ_ref::Array{Float64,2}, N_iter::Int64 = 100)


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

    

    @info "θ error :", norm(θ_ref - params_i), " / ",  norm(θ_ref)
    
    update_ensemble!(ukiobj, ens_func) 
    
    @info "data_mismatch :", (ukiobj.g_bar[end] - ukiobj.g_t)'*(ukiobj.obs_cov\(ukiobj.g_bar[end] - ukiobj.g_t))
    

    # visulize
    if i%10 == 0
      Run_Damage(phys_params, params_i,  "uki.disp", "uki.E")
      @save "ukiobj.dat" ukiobj
    end
    
  end
  
  return ukiobj
end


###############################################################################################
phys_params = Params()

# data
noise_level = 0.05
θ_ref, data =  Run_Damage(phys_params, nothing,  "None", "None", noise_level)
t_cov = Array(Diagonal(fill(1.0, phys_params.n_data))) 

nθ = length(θ_ref)
θ0_bar = zeros(Float64, nθ)
θθ0_cov = Array(Diagonal(fill(1.0, nθ)))           # standard deviation

####



N_iter = 100 

α_reg = 0.5
ukiobj = UKI(phys_params,
t_mean, t_cov, 
θ0_bar, θθ0_cov, 
α_reg,
E_ref,
N_iter)

@save "ukiobj.dat" ukiobj






