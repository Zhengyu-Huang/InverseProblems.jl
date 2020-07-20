using NNGCM
using JLD2
using Statistics
using LinearAlgebra



include("../Plot.jl")
include("Random_Init.jl")
include("../UKI.jl")

function Run_Random_Init(phys_params::Params, seq_pairs::Array{Int64,2}, θ::Array{Float64,1})
  nθ = size(θ) 
  abk = reshape(θ, Int64(nθ/2), 2)
  data = RandomInit_Main(abk, seq_paris, phys_params)
  
  return data[:]
end


function Ensemble_Random_Init(phys_params::Params, seq_pairs::Array{Int64,2}, params_i::Array{Float64, 2})
  n_data = phys_params.n_data
  
  N_ens,  N_θ = size(params_i)
  
  g_ens = zeros(Float64, N_ens,  N_data)
  
  Threads.@threads for i = 1:N_ens 
    # g: N_ens x N_data
    g_ens[i, :] .= Run_Random_Init(phys_params, seq_pairs, params_i[i, :])
  end
  
  return g_ens
end


function UKI(phys_params::Params, seq_pairs::Array{Int64,2},
  t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
  θ_bar::Array{Float64,1}, θθ_cov::Array{Float64,2}, 
  ω0_ref::Array{Float64,2},
  N_iter::Int64 = 100)


  mesh = Spectral_Mesh(phys_params.nx, phys_params.ny, phys_params.Lx, phys_params.Ly)
  
  parameter_names = ["ω0"]
  
  ens_func(θ_ens) = Ensemble_Random_Init(phys_params, seq_pairs, θ_ens)
  
  ukiobj = UKIObj(parameter_names,
  θ_bar, 
  θθ_cov,
  t_mean, # observation
  t_cov)
  
  
  for i in 1:N_iter
    
    params_i = deepcopy(ukiobj.θ_bar[end])

    abk = reshape(params_i, Int64(size(params_i,1)/2), 2)
    ω0 = Initial_ω0_KL(mesh, abk, seq_pairs)
    
    @info "F error of logκ :", norm(ω0_ref - ω0), " / ",  norm(ω0_ref)
    
    
    update_ensemble!(ukiobj, ens_func) 
    
    @info "F error of data_mismatch :", (ukiobj.g_bar[end] - ukiobj.g_t)'*(ukiobj.obs_cov\(ukiobj.g_bar[end] - ukiobj.g_t))
    
    
  end
  
  return ukiobj
end


###############################################################################################
phys_params = Params()

# data
ω0, t_mean =  Generate_Data(phys_params)
t_cov = Array(Diagonal(fill(1.0, phys_params.n_data))) 

# initial prior distribution is 
na = 20
θ0_bar = zeros(Float64, 2*na)                                 # mean 
θθ0_cov = Array(Diagonal(fill(1.0, 2*na)))           # standard deviation

seq_pairs = Compute_Seq_Pairs(na)
N_iter = 50 

ukiobj = UKI(phys_params, seq_pairs,
t_mean, t_cov, 
θ0_bar, θθ0_cov, 
ω0,
N_iter)






