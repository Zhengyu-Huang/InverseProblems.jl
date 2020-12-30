using JLD2
using Statistics
using LinearAlgebra



include("../Plot.jl")
include("Random_Init.jl")
include("../RUKI.jl")
include("../REKI.jl")
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



function Random_Init_Test(method::String, 
  phys_params::Params, seq_pairs, 
  t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
  θ0_bar::Array{Float64,1}, θθ0_cov::Array{Float64,2}, 
  N_ens::Int64, α_reg::Float64, 
  ω0_ref::Array{Float64,2}, N_iter::Int64, ax1, ax2)
  
  mesh = Spectral_Mesh(phys_params.nx, phys_params.ny, phys_params.Lx, phys_params.Ly)
  
  if method == "ExKI"
    label = "UKI"
    kiobj = ExKI(phys_params,seq_pairs, t_mean, t_cov,  θ0_bar, θθ0_cov, α_reg, ω0_ref, N_iter)
    θ_bar = kiobj.θ_bar
    linestyle, marker = "-", "o"
  elseif method =="EnKI"
    label = "EKI"
    kiobj = EnKI(phys_params, seq_pairs,t_mean, t_cov,  θ0_bar, θθ0_cov, N_ens, α_reg, ω0_ref, N_iter)
    θ_bar = [dropdims(mean(kiobj.θ[i], dims=1), dims=1) for i = 1:N_iter ]  
    linestyle, marker = "--", "s" 
  else
    error("method: ", method, "has not implemented")
  end
  
  
  label = label*" (α = "*string(α_reg)*")"
  
  ites = Array(LinRange(1, N_iter, N_iter))
  errors = zeros(Float64, (2, N_iter))
  for i = 1:N_iter
    
    ω0 = Initial_ω0_KL(mesh, θ_bar[i], seq_pairs)
    
    errors[1, i] = norm(ω0_ref - ω0)/norm(ω0_ref)
    errors[2, i] = 0.5*(kiobj.g_bar[i] - kiobj.g_t)'*(kiobj.obs_cov\(kiobj.g_bar[i] - kiobj.g_t))
    
  end

  @show method, α_reg
  
  
  if (!isnothing(ax1)  &&  !isnothing(ax2))
    ax1.plot(ites, errors[1, :], linestyle=linestyle, marker=marker, fillstyle="none", markevery=10, label= label)
    ax1.set_ylabel("Relative L₂ norm error")
    ax1.grid(true)
    
    
    ax2.plot(ites, errors[2, :], linestyle=linestyle, marker = marker,fillstyle="none", markevery=10, label= label)
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Optimization error")
    ax2.grid(true)
    ax2.legend()
  end
  
  
  
  return Initial_ω0_KL(mesh, θ_bar[end], seq_pairs)
  
  
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
  
  
  for i in 1:N_iter
    
    params_i = deepcopy(exkiobj.θ_bar[end])
    
    ω0 = Initial_ω0_KL(mesh, params_i, seq_pairs)
    
    
    @info "F error of ω0 :", norm(ω0_ref - ω0), " / ",  norm(ω0_ref)
    
    
    update_ensemble!(exkiobj, ens_func) 
    
    @info "F error of data_mismatch :", (exkiobj.g_bar[end] - exkiobj.g_t)'*(exkiobj.obs_cov\(exkiobj.g_bar[end] - exkiobj.g_t))
    
  end
  
  return exkiobj
end



function UKI(phys_params::Params, seq_pairs::Array{Int64,2},
  t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
  θ0_bar::Array{Float64,1}, θθ0_cov::Array{Float64,2}, 
  α_reg::Float64, 
  ω0_ref::Array{Float64,2}, N_iter::Int64 = 100)
  
  
  mesh = Spectral_Mesh(phys_params.nx, phys_params.ny, phys_params.Lx, phys_params.Ly)
  
  parameter_names = ["ω0"]
  
  ens_func(θ_ens) = Ensemble_Random_Init(phys_params, seq_pairs, θ_ens)
  
  ukiobj = UKIObj(parameter_names,
  θ0_bar, 
  θθ0_cov,
  t_mean, # observation
  t_cov,
  α_reg)
  
  
  for i in 1:N_iter
    
    params_i = deepcopy(ukiobj.θ_bar[end])
    
    ω0 = Initial_ω0_KL(mesh, params_i, seq_pairs)
    
    
    @info "F error of ω0 :", norm(ω0_ref - ω0), " / ",  norm(ω0_ref)
    
    
    update_ensemble!(ukiobj, ens_func) 
    
    @info "F error of data_mismatch :", (ukiobj.g_bar[end] - ukiobj.g_t)'*(ukiobj.obs_cov\(ukiobj.g_bar[end] - ukiobj.g_t))
    
  end
  
  return ukiobj
end

function EnKI(phys_params::Params, seq_pairs::Array{Int64,2},
  t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
  θ0_bar::Array{Float64,1}, θθ0_cov::Array{Float64,2}, 
  N_ens::Int64, α_reg::Float64, 
  ω0_ref::Array{Float64,2}, N_iter::Int64 = 100)
  
  
  mesh = Spectral_Mesh(phys_params.nx, phys_params.ny, phys_params.Lx, phys_params.Ly)
  
  parameter_names = ["ω0"]
  
  ens_func(θ_ens) = Ensemble_Random_Init(phys_params, seq_pairs, θ_ens)
  
  nθ = length(θ0_bar)
  priors = [Distributions.Normal(θ0_bar[i], sqrt(θθ0_cov[i,i])) for i=1:nθ]
  
  θ0 = construct_initial_ensemble(N_ens, priors; rng_seed=42)
  
  ekiobj = EKIObj(parameter_names,
  θ0,
  θ0_bar, 
  θθ0_cov,
  t_mean, # observation
  t_cov,
  α_reg)
  
  for i in 1:N_iter
    
    params_i = dropdims(mean(ekiobj.θ[end], dims=1), dims=1) 
    
    ω0 = Initial_ω0_KL(mesh, params_i, seq_pairs)
    
    @info "F error of ω0 :", norm(ω0_ref - ω0), " / ",  norm(ω0_ref)
    
    
    update_ensemble!(ekiobj, ens_func) 
    
    @info "F error of data_mismatch :", (ekiobj.g_bar[end] - ekiobj.g_t)'*(ekiobj.obs_cov\(ekiobj.g_bar[end] - ekiobj.g_t))
    
  end
  
  return ekiobj
end

function Plot_Field(mesh, ω0,  ax)
  vmin, vmax = -5.0, 5.0
  
  nx, ny = mesh.nx, mesh.ny
  xx, yy = mesh.xx, mesh.yy
  X,Y = repeat(xx, 1, ny), repeat(yy, 1, nx)'
  
  
  ax.pcolormesh(X, Y, ω0, shading= "gouraud", cmap="jet", vmin=vmin, vmax =vmax)
  
  
end


###############################################################################################

function Compare()
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
  seq_pairs = Compute_Seq_Pairs(na)
  t_cov = Array(Diagonal(fill(1.0, phys_params.n_data))) 
  θ0_bar = zeros(Float64, 2na)
  θθ0_cov = Array(Diagonal(fill(10.0, 2*na)))           # standard deviation
  
  N_iter = 50 
  N_ens = 201
  
  mesh = Spectral_Mesh(phys_params.nx, phys_params.ny, phys_params.Lx, phys_params.Ly)
  
  ω0_ref, _ =  Generate_Data(phys_params, -1.0,  "Figs/NS-vor.")


  fig_vor, ax_vor = PyPlot.subplots(ncols = 3, nrows=3, sharex=true, sharey=true, figsize=(12,12))
  for ax in ax_vor ;  ax.set_xticks([]) ; ax.set_yticks([]) ; end
  
  
  for noise_level_per in [0, 1, 5]
    
    # data
    noise_level = noise_level_per/100.0
    ω0_ref, t_mean =  Generate_Data(phys_params, noise_level)
    
    
    
    fig, (ax1, ax2) = PyPlot.subplots(nrows=2, sharex=true, figsize=(8,12))
    α_reg = 1.0

    ω0 = Random_Init_Test("ExKI", phys_params, seq_pairs, t_mean, t_cov, θ0_bar, θθ0_cov, N_ens, α_reg, ω0_ref, N_iter, ax1, ax2)
    if noise_level_per == 0
      Plot_Field(mesh, ω0, ax_vor[1])
    end


    ω0 = Random_Init_Test("EnKI", phys_params, seq_pairs, t_mean, t_cov, θ0_bar, θθ0_cov, N_ens, α_reg, ω0_ref, N_iter, ax1, ax2)
    if noise_level_per == 0
      Plot_Field(mesh, ω0, ax_vor[4])
    end
    
    
    
    
    α_reg = 0.9
    ω0 = Random_Init_Test("ExKI", phys_params, seq_pairs, t_mean, t_cov, θ0_bar, θθ0_cov, N_ens, α_reg, ω0_ref, N_iter, ax1, ax2)
    if noise_level_per != 0
      ax_id = (noise_level_per == 1 ? 2 : 3 ;)
      Plot_Field(mesh, ω0, ax_vor[ax_id])
    end

    ω0 = Random_Init_Test("EnKI", phys_params, seq_pairs, t_mean, t_cov, θ0_bar, θθ0_cov, N_ens, α_reg, ω0_ref, N_iter, ax1, ax2)
    if noise_level_per != 0
      ax_id = (noise_level_per == 1 ? 5 : 6 ;)
      Plot_Field(mesh, ω0, ax_vor[ax_id])
    end
    
    
    
    
    fig.tight_layout()
    fig.savefig("Figs/NS-noise-"*string(noise_level_per)*".pdf")
    close(fig)
  end
  

  
  Plot_Field(mesh, ω0_ref, ax_vor[7])
  im = Plot_Field(mesh, ω0_ref, ax_vor[8])
  Plot_Field(mesh, ω0_ref, ax_vor[9])
  
  
  fig_vor.tight_layout()
  cbar_ax = fig_vor.add_axes([0.90, 0.05, 0.02, 0.5])
  fig_vor.colorbar(im, cbar_ax)
  
  fig_vor.savefig("Figs/NS.pdf")
  close(fig_vor)
  
end

Compare()



