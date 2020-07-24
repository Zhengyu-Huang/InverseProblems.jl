using JLD2
using Statistics
using LinearAlgebra



include("../Plot.jl")
include("Random_Init.jl")
include("../RUKI.jl")

function Foward(phys_params::Params, seq_pairs::Array{Int64,2}, θ::Array{Float64,1})

  
  
  return RandomInit_Main(θ, seq_pairs, phys_params)


end


function Ensemble_Random_Init(phys_params::Params, seq_pairs::Array{Int64,2}, params_i::Array{Float64, 2})
  n_data = phys_params.n_data
  
  N_ens,  N_θ = size(params_i)
  
  g_ens = zeros(Float64, N_ens,  n_data)
  
  Threads.@threads for i = 1:N_ens 
    # g: N_ens x N_data
    g_ens[i, :] .= Foward(phys_params, seq_pairs, params_i[i, :])
  end
  
  return g_ens
end


function UKI(phys_params::Params, seq_pairs::Array{Int64,2},
  t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
  θ_bar::Array{Float64,1}, θθ_cov::Array{Float64,2}, 
  α_reg::Float64, 
  ω0_ref::Array{Float64,2}, N_iter::Int64 = 100)


  mesh = Spectral_Mesh(phys_params.nx, phys_params.ny, phys_params.Lx, phys_params.Ly)
  
  parameter_names = ["ω0"]
  
  ens_func(θ_ens) = Ensemble_Random_Init(phys_params, seq_pairs, θ_ens)
  
  ukiobj = UKIObj(parameter_names,
  θ_bar, 
  θθ_cov,
  t_mean, # observation
  t_cov,
  α_reg)
  
  
  for i in 1:N_iter
    
    params_i = deepcopy(ukiobj.θ_bar[end])

    ω0 = Initial_ω0_KL(mesh, params_i, seq_pairs)
    
    
    @info "F error of ω0 :", norm(ω0_ref - ω0), " / ",  norm(ω0_ref)
    
    
    update_ensemble!(ukiobj, ens_func) 
    
    @info "F error of data_mismatch :", (ukiobj.g_bar[end] - ukiobj.g_t)'*(ukiobj.obs_cov\(ukiobj.g_bar[end] - ukiobj.g_t))
    
    # visulize
    if i%10 == 0
      Foward_Helper(phys_params, ω0, "uki.vor-"*string(i)*".")
      @save "ukiobj.dat" ukiobj
    end
    
  end
  
  return ukiobj
end


###############################################################################################
phys_params = Params()

# data
noise_level = 0.05
ω0, t_mean =  Generate_Data(phys_params, noise_level )
t_cov = Array(Diagonal(fill(1.0, phys_params.n_data))) 

# initial prior distribution is 
na = 100
seq_pairs = Compute_Seq_Pairs(na)

#nx, ny, Δd_x, Δd_y = phys_params.nx, phys_params.ny, phys_params.Δd_x, phys_params.Δd_y
    
#ndata0 = min((div(nx-1,Δd_x)+1)*(div(ny-1,Δd_y)+1), 2*na)


#θ0_bar = Construct_θ0(phys_params, ω0, div(ndata0,2), seq_pairs)
#@show size(θ0_bar)
θ0_bar = zeros(Float64, 2na)
θθ0_cov = Array(Diagonal(fill(100.0, 2*na)))           # standard deviation

####


####



N_iter = 100 

α_reg = 0.5
ukiobj = UKI(phys_params, seq_pairs,
t_mean, t_cov, 
θ0_bar, θθ0_cov, 
α_reg,
ω0,
N_iter)

@save "ukiobj.dat" ukiobj













