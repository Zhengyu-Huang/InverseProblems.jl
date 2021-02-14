using JLD2
using Statistics
using LinearAlgebra



include("../Plot.jl")
include("Random_Init.jl")
include("../RExKI.jl")
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



function Random_Init_Test(
  phys_params::Params, seq_pairs, 
  t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
  θ0_bar::Array{Float64,1}, θθ0_cov::Array{Float64,2}, 
  α_reg::Float64, 
  θ_ref::Array{Float64,1},
  ω0_ref::Array{Float64,2}, N_iter::Int64)
  
  mesh = Spectral_Mesh(phys_params.nx, phys_params.ny, phys_params.Lx, phys_params.Ly)
  
  
  kiobj = ExKI(phys_params,seq_pairs, t_mean, t_cov,  θ0_bar, θθ0_cov, α_reg, ω0_ref, N_iter)
  
  return kiobj, mesh
  
  
end


function ExKI(phys_params::Params, seq_pairs::Array{Int64,2},
  t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
  θ0_bar::Array{Float64,1}, θθ0_cov::Array{Float64,2}, 
  α_reg::Float64, 
  ω0_ref::Array{Float64,2}, N_iter::Int64 = 100)
  
  
  mesh = Spectral_Mesh(phys_params.nx, phys_params.ny, phys_params.Lx, phys_params.Ly)
  
  parameter_names = ["ω0"]
  
  ens_func(θ_ens) = Ensemble_Random_Init(phys_params, seq_pairs, θ_ens)
  
  exkiobj = ExKIObj(parameter_names,
  θ0_bar, 
  θθ0_cov,
  t_mean, # observation
  t_cov,
  α_reg)
  
  update_cov = 1
  for i in 1:N_iter
    
    params_i = deepcopy(exkiobj.θ_bar[end])
    
    ω0 = Initial_ω0_KL(mesh, params_i, seq_pairs)
    
    
    @info "F error of ω0 :", norm(ω0_ref - ω0), " / ",  norm(ω0_ref)
    
    
    update_ensemble!(exkiobj, ens_func) 
    
    @info "F error of data_mismatch :", (exkiobj.g_bar[end] - exkiobj.g_t)'*(exkiobj.obs_cov\(exkiobj.g_bar[end] - exkiobj.g_t))
    
    @info "norm(cov) is ", norm(exkiobj.θθ_cov[end])
    
    if (update_cov > 0) && (i%update_cov == 0) 
      exkiobj.θθ_cov[1] = copy(exkiobj.θθ_cov[end])
    end
    
  end
  
  return exkiobj
end





function Plot_Field(mesh, ω0,  ax)
  vmin, vmax = -5.0, 5.0
  
  nx, ny = mesh.nx, mesh.ny
  xx, yy = mesh.xx, mesh.yy
  X,Y = repeat(xx, 1, ny), repeat(yy, 1, nx)'
  
  
  ax.pcolormesh(X, Y, ω0, shading= "gouraud", cmap="jet", vmin=vmin, vmax =vmax)
  
  
end


###############################################################################################

function UQ_test()
  rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
  mysize = 20
  font0 = Dict(
  "font.size" => mysize,
  "axes.labelsize" => mysize,
  "xtick.labelsize" => mysize,
  "ytick.labelsize" => mysize,
  "legend.fontsize" => mysize,
  )
  merge!(rcParams, font0)
  
  
  phys_params = Params()
  
  
  # initial prior distribution is 
  na = 50
  N_θ = 2na
  
  seq_pairs = Compute_Seq_Pairs(na)
  # 30
  N_iter = 30
  
  mesh = Spectral_Mesh(phys_params.nx, phys_params.ny, phys_params.Lx, phys_params.Ly)
  
  
  na_ref = na
  seq_pairs_ref = Compute_Seq_Pairs(2*na_ref)
  ω0_ref, _, _ =  Generate_Data(phys_params, seq_pairs_ref, -1.0,  "Figs/NS-vor-perfect.", 2*na_ref)
  
  ω0_ref, θ_ref, t_mean_noiseless =  Generate_Data_Noiseless(phys_params, seq_pairs_ref, "None", 2*na_ref)
  θ_ref = reshape(θ_ref, na_ref, 2)[1:na, :][:]
  
  N_sample = 6
  
  fig_ite, ax_ite = PyPlot.subplots(ncols = 3, nrows=1, sharex=false, sharey=false, figsize=(18,6))
  fig_θ, ax_θ = PyPlot.subplots(ncols = 1, nrows=N_sample, sharex=true, sharey=true, figsize=(16,12))
  fig_omega0, ax_omega0 = PyPlot.subplots(ncols = 3, nrows=2, sharex=true, sharey=true, figsize=(18,12))
  for ax in ax_omega0 ;  ax.set_xticks([]) ; ax.set_yticks([]) ; end
  
  # data
  θ0_bar = zeros(Float64, 2*na)
  θθ0_cov = Array(Diagonal(fill(1.0, 2*na)))
  α_reg = 1.0
  noise_level = 0.05
  
  
  
  
  
  # observation
  t_cov = Array(Diagonal(noise_level^2 * t_mean_noiseless.^2))
  Random.seed!(123);
  
  ites = Array(LinRange(1, N_iter, N_iter))
  errors = zeros(Float64, (3, N_iter))
  
  for n = 1:N_sample
    t_mean = copy(t_mean_noiseless)
    for i = 1:length(t_mean)
      noise = noise_level*t_mean[i] * rand(Normal(0, 1))
      t_mean[i] += noise
    end
    
    kiobj, mesh = Random_Init_Test(phys_params, seq_pairs, t_mean, t_cov, θ0_bar, θθ0_cov, α_reg, θ_ref, ω0_ref, N_iter)
    
    
    for i = 1:N_iter
      
      ω0 = Initial_ω0_KL(mesh, kiobj.θ_bar[i], seq_pairs)
      
      errors[1, i] = norm(ω0_ref - ω0)/norm(ω0_ref)
      errors[2, i] = 0.5*(kiobj.g_bar[i] - kiobj.g_t)'*(kiobj.obs_cov\(kiobj.g_bar[i] - kiobj.g_t))
      errors[3, i] = norm(kiobj.θθ_cov[i])   
      
    end
    errors[3, 1] = norm(θθ0_cov) 
    
    ax_ite[1].semilogy(ites, errors[1, :], "-o", fillstyle="none", markevery=2 )
    ax_ite[2].semilogy(ites, errors[2, :], "-o", fillstyle="none", markevery=2 )
    ax_ite[3].semilogy(ites, errors[3, :], "-o", fillstyle="none", markevery=2 )
    
    θ_ind = Array(1:N_θ)

    ax_θ[n].plot(θ_ind , θ_ref[1:N_θ], "--o", color="grey", fillstyle="none", label="Reference")


    ki_θ_bar  = kiobj.θ_bar[end]
    ki_θθ_cov = kiobj.θθ_cov[end]
    ki_θθ_std = sqrt.(diag(kiobj.θθ_cov[end]))
    ax_θ[n].plot(θ_ind , ki_θ_bar,"-*", color="red", fillstyle="none")
    ax_θ[n].plot(θ_ind , ki_θ_bar + 3.0*ki_θθ_std, color="red")
    ax_θ[n].plot(θ_ind , ki_θ_bar - 3.0*ki_θθ_std, color="red")
    # ax_θ[n].grid(true)
    ax_θ[n].set_ylabel("θ")

    ω0 = Initial_ω0_KL(mesh, kiobj.θ_bar[end], seq_pairs)
    Plot_Field(mesh, ω0, ax_omega0[:][n])

  end
  
  ax_ite[1].set_xlabel("Iterations")
  ax_ite[1].set_ylabel("Relative L₂ norm error")
  ax_ite[1].grid(true)
  ax_ite[2].set_xlabel("Iterations")
  ax_ite[2].set_ylabel("Optimization error")
  ax_ite[2].grid(true)
  ax_ite[3].set_xlabel("Iterations")
  ax_ite[3].set_ylabel("Frobenius norm")
  ax_ite[3].grid(true)
  
  fig_ite.tight_layout()
  fig_ite.savefig("Figs/NS-error-perfect.png")
  
  ax_θ[N_sample].set_xlabel("θ indices")
  fig_θ.tight_layout()
  fig_θ.savefig("Figs/NS-theta-perfect.png")
  
  fig_omega0.savefig("Figs/NS-omega0-perfect.pdf")

  
  
end



UQ_test()



