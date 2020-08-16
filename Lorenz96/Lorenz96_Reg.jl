using JLD2
using Statistics
using LinearAlgebra



include("../Plot.jl")
include("Lorenz96.jl")
include("../RExKI.jl")
include("../RUKI.jl")


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
  
  savefig("Figs/Sample_Traj."*string(ite)*".pdf"); close("all")
  
  figure(2) # hist
  hist(data_ref[:,1:K][:], bins = 1000, density = true, histtype = "step", label="Ref")
  hist(data[:,1:K][:], bins = 1000, density = true, histtype = "step", label="DA")
  legend()
  savefig("Figs/X_density."*string(ite)*".pdf"); close("all")
  
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
  savefig("Figs/Closure."*string(ite)*".pdf"); close("all")
  
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
    
    
  end
  
  return exkiobj
end

function UKI(phys_params::Params, 
  t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
  θ_bar::Array{Float64,1}, θθ_cov::Array{Float64,2}, 
  α_reg::Float64, 
  N_iter::Int64,
  data_ref::Array{Float64,2}, Q0::Array{Float64,1}, Φ::Function
  )
  
  
  parameter_names = ["NN"]
  
  ens_func(θ_ens) = Ensemble(phys_params, Q0, θ_ens, Φ)
  
  ukiobj = UKIObj(parameter_names,
  θ_bar, 
  θθ_cov,
  t_mean, # observation
  t_cov,
  α_reg)
  
  
  for i in 1:N_iter
    
    
    
    update_ensemble!(ukiobj, ens_func) 
    
    @info "data_mismatch :", (ukiobj.g_bar[end] - ukiobj.g_t)'*(ukiobj.obs_cov\(ukiobj.g_bar[end] - ukiobj.g_t))
    
    params_i = deepcopy(ukiobj.θ_bar[end])
    

    # visulize
    # if i%10 == 0
      
    #   Show_Result(phys_params, data_ref, Q0, params_i, Φ, i)
      
      
    #   ukiobj_θ_bar, ukiobj_θθ_cov, ukiobj_g_bar = ukiobj.θ_bar, ukiobj.θθ_cov, ukiobj.g_bar
    #   ukiobj_θθ_cov = [diag(ukiobj.θθ_cov[i]) for i=1:length(ukiobj.θθ_cov)]
      
    #   @save "ukiobj.dat" ukiobj_θ_bar ukiobj_g_bar ukiobj_θθ_cov
    # end
    
  end
  
  return ukiobj
end

###############################################################################################

case = "Wilks"  # Wilks or ESM2.0
obs_type, obs_p = "Statistics", 4


T, ΔT = 1000, 0.005

phys_params = Params(case, obs_type, obs_p,  T, ΔT) 
K, J = phys_params.K, phys_params.J 
NT = phys_params.NT

Random.seed!(42);
Q0_ref = [rand(Normal(0, 1.0), K) ; rand(Normal(0, 0.01), K*J)]
Q0 = rand(Normal(0, 1.0), K)


data_ref = Run_Lorenz96(phys_params, Q0_ref)
data_0 = Run_Lorenz96(phys_params, Q0, [0.0], Φpoly)

nθ = 12
θ0_bar = zeros(Float64, nθ)  #rand(Normal(0, 1), nθ)                    # 
θθ0_cov = Array(Diagonal(fill(1.0, nθ)))           # standard deviation
Φ = ΦQP
N_iter = 100

α_reg = 1.0


noise_levels = [0.0; 1.0/100; 5.0/100]
exkiobjs, rexkiobjs, ukiobjs = [], [], []
for i = 1:3
  noise_level = noise_levels[i]
  
  # set mean and covariance
  t_mean =  Compute_Obs(phys_params, data_ref)
  t_cov = Array(Diagonal(0.05^2*t_mean.^2))
  
  if noise_level > 0.0
    Random.seed!(123);
    for i = 1:length(t_mean)
      noise = rand(Normal(0, noise_level*t_mean[i]))
      t_mean[i] += noise
    end
  end
  
  
  
  exkiobj = ExKI(phys_params,
  t_mean, t_cov, 
  θ0_bar, θθ0_cov, 
  α_reg,
  N_iter,
  data_ref, Q0, Φ)

  push!(exkiobjs, exkiobj)
  
  
  ukiobj = UKI(phys_params,
  t_mean, t_cov, 
  θ0_bar, θθ0_cov, 
  α_reg,
  N_iter,
  data_ref, Q0, Φ)
  push!(ukiobjs, ukiobj)


  rexkiobj = ExKI(phys_params,
  t_mean, t_cov, 
  θ0_bar, θθ0_cov, 
  0.9,
  N_iter,
  data_ref, Q0, Φ)

  push!(rexkiobjs, rexkiobj)

end


  fig, ax = PyPlot.subplots(ncols=3, sharey=true,figsize=(18,6))
  nbin = 100
  ax[1].hist(data_ref[:,1:K][:], bins = nbin, color="black", density = true, histtype = "step", label="Truth")
  ax[1].hist(data_0[:,1:K][:], bins = nbin, color="C5", density = true, histtype = "step", label="Prior")
  data = Run_Lorenz96(phys_params, Q0, ukiobjs[1].θ_bar[end], Φ)
  ax[1].hist(data[:,1:K][:], bins = nbin, color="C1", density = true, histtype = "step", label="0%")
  data = Run_Lorenz96(phys_params, Q0, ukiobjs[2].θ_bar[end], Φ)
  ax[1].hist(data[:,1:K][:], bins = nbin, color="C2", density = true, histtype = "step", label="1%")
  data = Run_Lorenz96(phys_params, Q0, ukiobjs[3].θ_bar[end], Φ)
  ax[1].hist(data[:,1:K][:], bins = nbin, color="C3", density = true, histtype = "step", label="5%")
  
  ax[2].hist(data_ref[:,1:K][:], bins = nbin, color="black", density = true, histtype = "step", label="Truth")
  ax[2].hist(data_0[:,1:K][:], bins = nbin, color="C5", density = true, histtype = "step", label="Prior")
  data = Run_Lorenz96(phys_params, Q0, exkiobjs[1].θ_bar[end], Φ)
  ax[2].hist(data[:,1:K][:], bins = nbin, color="C1", density = true, histtype = "step", label="0%")
  data = Run_Lorenz96(phys_params, Q0, exkiobjs[2].θ_bar[end], Φ)
  ax[2].hist(data[:,1:K][:], bins = nbin, color="C2", density = true, histtype = "step", label="1%")
  data = Run_Lorenz96(phys_params, Q0, exkiobjs[3].θ_bar[end], Φ)
  ax[2].hist(data[:,1:K][:], bins = nbin, color="C3", density = true, histtype = "step", label="5%")

  ax[3].hist(data_ref[:,1:K][:], bins = nbin, color="black", density = true, histtype = "step", label="Truth")
  ax[3].hist(data_0[:,1:K][:], bins = nbin, color="C5", density = true, histtype = "step", label="Prior")
  data = Run_Lorenz96(phys_params, Q0, rexkiobjs[1].θ_bar[end], Φ)
  ax[3].hist(data[:,1:K][:], bins = nbin, color="C1", density = true, histtype = "step", label="0%")
  data = Run_Lorenz96(phys_params, Q0, rexkiobjs[2].θ_bar[end], Φ)
  ax[3].hist(data[:,1:K][:], bins = nbin, color="C2", density = true, histtype = "step", label="1%")
  data = Run_Lorenz96(phys_params, Q0, rexkiobjs[3].θ_bar[end], Φ)
  ax[3].hist(data[:,1:K][:], bins = nbin, color="C3", density = true, histtype = "step", label="5%")



  ax[1].set_xlabel("X")
  #ax[1].legend()

  ax[2].set_xlabel("X")
  #ax[2].legend()

  ax[3].set_xlabel("X")
  ax[3].legend()
  fig.tight_layout()
  
  fig.savefig("Figs/Lorenz96-density.pdf"); 
  close(fig)
  




  fig, ax = PyPlot.subplots(ncols=3, sharey=true, figsize=(18,6))
  # Xg[k] vs - h*c/d*(sum(Yg[ng+1:J+ng, k]))
  h, c, d = phys_params.h, phys_params.c, phys_params.d
  X = (data_ref[:,1:K]')[:]
  Φ_ref = -h*c/d * sum(reshape(data_ref[:, K+1:end]', J, K*size(data_ref, 1)), dims=1)
  ax[1].scatter(X, Φ_ref, s = 0.1, c="grey")
  ax[2].scatter(X, Φ_ref, s = 0.1, c="grey")
  ax[3].scatter(X, Φ_ref, s = 0.1, c="grey")
  
  xx = Array(LinRange(-10, 15, 1000))
  
  fwilks = -(0.262 .+ 1.45*xx - 0.0121*xx.^2 - 0.00713*xx.^3 + 0.000296*xx.^4)
  ax[1].plot(xx, fwilks, c="C4", label="Wilks")
  ax[2].plot(xx, fwilks, c="C4", label="Wilks")
  ax[3].plot(xx, fwilks, c="C4", label="Wilks")
  
  fAMP = -(0.341 .+ 1.3*xx - 0.0136*xx.^2 - 0.00235*xx.^3)
  ax[1].plot(xx, fAMP, c="C5", label="Arnold")
  ax[2].plot(xx, fAMP, c="C5", label="Arnold")
  ax[3].plot(xx, fAMP, c="C5", label="Arnold")
  
  fDA = similar(xx)
  for i = 1:length(xx);  fDA[i] = Φ(xx[i], ukiobjs[1].θ_bar[end]); end
  ax[1].plot(xx, fDA, c="C1", linestyle="--", marker="o", fillstyle="none", markevery=100, label="0%")
  for i = 1:length(xx);  fDA[i] = Φ(xx[i], ukiobjs[2].θ_bar[end]); end
  ax[1].plot(xx, fDA, c="C2", linestyle="--", marker="o", fillstyle="none", markevery=100, label="1%")
  for i = 1:length(xx);  fDA[i] = Φ(xx[i], ukiobjs[3].θ_bar[end]); end
  ax[1].plot(xx, fDA, c="C3", linestyle="--", marker="o", fillstyle="none", markevery=100, label="5%")

  for i = 1:length(xx);  fDA[i] = Φ(xx[i], exkiobjs[1].θ_bar[end]); end
  ax[2].plot(xx, fDA, c="C1", linestyle="--", marker="o", fillstyle="none", markevery=100,  label="0%")
  for i = 1:length(xx);  fDA[i] = Φ(xx[i], exkiobjs[2].θ_bar[end]); end
  ax[2].plot(xx, fDA, c="C2", linestyle="--", marker="o", fillstyle="none", markevery=100, label="1%")
  for i = 1:length(xx);  fDA[i] = Φ(xx[i], exkiobjs[3].θ_bar[end]); end
  ax[2].plot(xx, fDA, c="C3", linestyle="--", marker="o", fillstyle="none", markevery=100, label="5%")

  for i = 1:length(xx);  fDA[i] = Φ(xx[i], rexkiobjs[1].θ_bar[end]); end
  ax[3].plot(xx, fDA, c="C1", linestyle="--", marker="o", fillstyle="none", markevery=100, label="0%")
  for i = 1:length(xx);  fDA[i] = Φ(xx[i], rexkiobjs[2].θ_bar[end]); end
  ax[3].plot(xx, fDA, c="C2", linestyle="--", marker="o", fillstyle="none", markevery=100, label="1%")
  for i = 1:length(xx);  fDA[i] = Φ(xx[i], rexkiobjs[3].θ_bar[end]); end
  ax[3].plot(xx, fDA, c="C3", linestyle="--", marker="o", fillstyle="none", markevery=100, label="5%")


  ax[1].set_xlabel("X")
  ax[1].set_ylabel("ψ(X)")
  #ax[1].legend()
  ax[2].set_xlabel("X")
  #ax[2].legend()
  ax[3].set_xlabel("X")
  ax[3].legend()
  fig.tight_layout()
  # fig.subplots_adjust(right=0.8)
  # ax.legend(bbox_to_anchor=(0.8, 0.8))
  fig.savefig("Figs/Lorenz96-Closure.pdf"); 
  close(fig)
  
  
  # optimization error
  ites = Array(LinRange(1, N_iter, N_iter))
  errors = zeros(Float64, N_iter)
  
  fig, ax = PyPlot.subplots(ncols=3, sharey=true, figsize=(18,6))
  for i = 1:N_iter; errors[i] = 0.5*(ukiobjs[1].g_bar[i] - ukiobjs[1].g_t)'*(ukiobjs[1].obs_cov\(ukiobjs[1].g_bar[i] - ukiobjs[1].g_t)); end
  ax[1].plot(ites, errors, linestyle="--", marker="o", fillstyle="none", markevery=10,  label="0%")
  for i = 1:N_iter; errors[i] = 0.5*(ukiobjs[2].g_bar[i] - ukiobjs[2].g_t)'*(ukiobjs[2].obs_cov\(ukiobjs[2].g_bar[i] - ukiobjs[2].g_t)); end
  ax[1].plot(ites, errors, linestyle="--", marker="o", fillstyle="none", markevery=10,  label="1%")
  for i = 1:N_iter; errors[i] = 0.5*(ukiobjs[3].g_bar[i] - ukiobjs[3].g_t)'*(ukiobjs[3].obs_cov\(ukiobjs[3].g_bar[i] - ukiobjs[3].g_t)); end
  ax[1].plot(ites, errors, linestyle="--", marker="o", fillstyle="none", markevery=10,  label="5%")


  for i = 1:N_iter; errors[i] = 0.5*(exkiobjs[1].g_bar[i] - exkiobjs[1].g_t)'*(exkiobjs[1].obs_cov\(exkiobjs[1].g_bar[i] - exkiobjs[1].g_t)); end
  ax[2].plot(ites, errors, linestyle="--", marker="o", fillstyle="none", markevery=10,  label="0%")
  for i = 1:N_iter; errors[i] = 0.5*(exkiobjs[2].g_bar[i] - exkiobjs[2].g_t)'*(exkiobjs[2].obs_cov\(exkiobjs[2].g_bar[i] - exkiobjs[2].g_t)); end
  ax[2].plot(ites, errors, linestyle="--", marker="o", fillstyle="none", markevery=10,  label="1%")
  for i = 1:N_iter; errors[i] = 0.5*(exkiobjs[3].g_bar[i] - exkiobjs[3].g_t)'*(exkiobjs[3].obs_cov\(exkiobjs[3].g_bar[i] - exkiobjs[3].g_t)); end
  ax[2].plot(ites, errors, linestyle="--", marker="o", fillstyle="none", markevery=10,  label="5%")
  
  for i = 1:N_iter; errors[i] = 0.5*(rexkiobjs[1].g_bar[i] - rexkiobjs[1].g_t)'*(rexkiobjs[1].obs_cov\(rexkiobjs[1].g_bar[i] - rexkiobjs[1].g_t)); end
  ax[3].plot(ites, errors, linestyle="--", marker="o", fillstyle="none", markevery=10,  label="0%")
  for i = 1:N_iter; errors[i] = 0.5*(rexkiobjs[2].g_bar[i] - rexkiobjs[2].g_t)'*(rexkiobjs[2].obs_cov\(rexkiobjs[2].g_bar[i] - rexkiobjs[2].g_t)); end
  ax[3].plot(ites, errors, linestyle="--", marker="o", fillstyle="none", markevery=10,  label="1%")
  for i = 1:N_iter; errors[i] = 0.5*(rexkiobjs[3].g_bar[i] - rexkiobjs[3].g_t)'*(rexkiobjs[3].obs_cov\(rexkiobjs[3].g_bar[i] - rexkiobjs[3].g_t)); end
  ax[3].plot(ites, errors, linestyle="--", marker="o", fillstyle="none", markevery=10,  label="5%")
  

  ax[1].set_xlabel("Iterations")
  ax[1].set_ylabel("Optimization error")
  ax[1].grid(true)
  ax[2].set_xlabel("Iterations")
  ax[2].grid(true)
  ax[3].set_xlabel("Iterations")
  ax[3].grid(true)
  ax[3].legend()
  
  fig.savefig("Figs/Lorenz96-Opt-Error.pdf"); 
  close(fig)

  








