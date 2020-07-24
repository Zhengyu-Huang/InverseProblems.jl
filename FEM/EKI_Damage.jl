using NNGCM
using JLD2
using Statistics
using LinearAlgebra



include("../Plot.jl")
include("Damage.jl")
include("../REKI.jl")

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


function EKI(phys_params::Params,
  N_ens::Int64,
  t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
  θ0_bar::Array{Float64,1}, θθ_cov::Array{Float64,2}, 
  α_reg::Float64, 
  θ_ref::Array{Float64,2}, N_iter::Int64 = 100)



  parameter_names = ["E"]
  
  ens_func(θ_ens) = Ensemble_Random_Init(phys_params, θ_ens)
  
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

    
    @info "θ error :", norm(θ_ref - params_i), " / ",  norm(θ_ref)
    
    
    update_ensemble!(ekiobj, ens_func) 
    
    @info "F error of data_mismatch :", (ekiobj.g_bar[end] - ekiobj.g_t)'*(ekiobj.obs_cov\(ekiobj.g_bar[end] - ekiobj.g_t))
    
        # visulize
        if i%10 == 0
          Run_Damage(phys_params, params_i,  "eki.disp", "eki.E")
          @save "ekiobj.dat" ukiobj
        end
    
  end
  
  return ekiobj
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




N_iter = 100 

α_reg = 0.5
N_ens = 20
ekiobj = EKI(phys_params, seq_pairs,
N_ens,
t_mean, t_cov, 
θ0_bar, θθ0_cov, 
α_reg,
ω0,
N_iter)






@save "ekiobj.dat" ekiobj






