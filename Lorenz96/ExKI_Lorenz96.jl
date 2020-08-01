using JLD2
using Statistics
using LinearAlgebra



include("../Plot.jl")
include("Lorenz96.jl")
include("../RExKI.jl")


function Foward(phys_params::Params, Q0::Array{Float64, 1}, θ::Array{Float64, 1}, Φ::Function)
  
  data = Run_Lorenz96(phys_params, Q0, θ, Φ)
  
  obs = Compute_Obs(phys_params, data)

  return obs
  
end


function Ensemble(phys_params::Params, Q0::Array{Float64, 1}, params_i::Array{Float64, 2}, Φ::Function)
  n_data = phys_params.n_data
  
  N_ens,  N_θ = size(params_i)
  
  g_ens = zeros(Float64, N_ens,  n_data)
  
  Threads.@threads for i = 1:N_ens 
    # g: N_ens x N_data
    g_ens[i, :] .= Foward(phys_params, Q0, params_i[i, :], Φ)
  end
  
  return g_ens
end


function Show_Result(phys_params::Params, data_ref::Array{Float64, 2}, Q0::Array{Float64, 1}, θ::Array{Float64, 1}, Φ::Function, ite::Int64)
  
  T,  ΔT, NT = phys_params.T,  phys_params.ΔT, phys_params.NT
  tt = Array(LinRange(ΔT, T, NT))
  
  
  #K, J = 36, 10
  K, J = phys_params.K, phys_params.J
  
  data = Run_Lorenz96(phys_params, Q0, θ, Φ)
  
  figure(1)
  plot(tt, data_ref[:,1], label = "Ref")
  plot(tt, data[:,1], label = "DA")
  legend()

  savefig("Figs/Sample_Traj."*string(ite)*".png"); close("all")
  
  figure(2) # hist
  hist(data_ref[:,1:K][:], bins = 1000, density = true, histtype = "step", label="Ref")
  hist(data[:,1:K][:], bins = 1000, density = true, histtype = "step", label="DA")
  legend()
  savefig("Figs/X_density."*string(ite)*".png"); close("all")
  
  figure(3) # modeling term
  # Xg[k] vs - h*c/d*(sum(Yg[ng+1:J+ng, k]))
  h, c, d = phys_params.h, phys_params.c, phys_params.d
  X = (data_ref[:,1:K]')[:]
  Φ_ref = -h*c/d * sum(reshape(data_ref[:, K+1:end]', J, K*size(data_ref, 1)), dims=1)
  scatter(X, Φ_ref, s = 0.1, c="grey")
  xx = Array(LinRange(-10, 15, 1000))
  
  fwilks = -(0.262 .+ 1.45*xx - 0.0121*xx.^2 - 0.00713*xx.^3 + 0.000296*xx.^4)
  plot(xx, fwilks, label="Wilks")
  
  fAMP = -(0.341 .+ 1.3*xx - 0.0136*xx.^2 - 0.00235*xx.^3)
  plot(xx, fAMP, label="Arnold")

  fDA = similar(xx)
  for i = 1:length(xx)
    fDA[i] = Φ(xx[i], θ) 
  end

  plot(xx, fDA, label="DA")
  legend()
  savefig("Figs/Closure."*string(ite)*".png"); close("all")
  
end

function ExKI(phys_params::Params, 
  t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
  θ_bar::Array{Float64,1}, θθ_cov::Array{Float64,2}, 
  α_reg::Float64, 
  N_iter::Int64,
  data_ref::Array{Float64,2}, Q0::Array{Float64,1}, Φ::Function
  )
  
  
  parameter_names = ["NN"]
  
  ens_func(θ_ens) = Ensemble(phys_params, Q0, θ_ens, Φ)
  
  exkiobj = ExKIObj(parameter_names,
  θ_bar, 
  θθ_cov,
  t_mean, # observation
  t_cov,
  α_reg)
  
  
  for i in 1:N_iter
    
    
    
    update_ensemble!(exkiobj, ens_func) 
    
    @info "data_mismatch :", (exkiobj.g_bar[end] - exkiobj.g_t)'*(exkiobj.obs_cov\(exkiobj.g_bar[end] - exkiobj.g_t))
    
    params_i = deepcopy(exkiobj.θ_bar[end])

    @info "θ: ", params_i
    @info "Varθ: ", diag(exkiobj.θθ_cov[end])
    # visulize
    if i%10 == 0
      
      Show_Result(phys_params, data_ref, Q0, params_i, Φ, i)
      
      
      exkiobj_θ_bar, exkiobj_θθ_cov, exkiobj_g_bar = exkiobj.θ_bar, exkiobj.θθ_cov, exkiobj.g_bar
      exkiobj_θθ_cov = [diag(exkiobj.θθ_cov[i]) for i=1:length(exkiobj.θθ_cov)]
      
      @save "exkiobj.dat" exkiobj_θ_bar exkiobj_g_bar exkiobj_θθ_cov
    end
    
  end
  
  return exkiobj
end


###############################################################################################
RK_order = 4
T,  ΔT = 20.0, 0.005
NT = Int64(T/ΔT)
tt = Array(LinRange(ΔT, T, NT))
Δobs = 200

K, J = 8, 32
obs_type,  kmode = "Statistics", 8
phys_params = Params(K, J, RK_order, T,  ΔT, obs_type, kmode)
Random.seed!(42);
Q0_ref = rand(Normal(0, 1), K*(J+1))
Q0 = Q0_ref[1:K]
#Q0 = Array(LinRange(1.0, K*(J+1), K*(J+1)))
data_ref = Run_Lorenz96(phys_params, Q0_ref )
t_mean =  Compute_Obs(phys_params, data_ref)

t_cov = Array(Diagonal(fill(1.0, length(t_mean)))) 


nθ = 12
θ0_bar = rand(Normal(0, 1), nθ)  #zeros(Float64, nθ)
θθ0_cov = Array(Diagonal(fill(1.0, nθ)))           # standard deviation

Φ = ΦQP



N_iter = 100

α_reg = 1.0
exkiobj = ExKI(phys_params,
t_mean, t_cov, 
θ0_bar, θθ0_cov, 
α_reg,
N_iter,
data_ref, Q0, Φ)






