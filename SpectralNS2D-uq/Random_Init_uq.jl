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
  α_reg::Float64, 
  θ_ref::Array{Float64,1},
  ω0_ref::Array{Float64,2}, N_iter::Int64)
  
  mesh = Spectral_Mesh(phys_params.nx, phys_params.ny, phys_params.Lx, phys_params.Ly)
  
  
  kiobj = ExKI(phys_params,seq_pairs, t_mean, t_cov,  θ0_bar, θθ0_cov, α_reg, ω0_ref, N_iter)
  θ_bar = kiobj.θ_bar
  linestyle, marker = "-", "o"
  
  
  ites = Array(LinRange(1, N_iter, N_iter))
  errors = zeros(Float64, (3, N_iter))
  for i = 1:N_iter
    
    ω0 = Initial_ω0_KL(mesh, θ_bar[i], seq_pairs)
    
    errors[1, i] = norm(ω0_ref - ω0)/norm(ω0_ref)
    errors[2, i] = 0.5*(kiobj.g_bar[i] - kiobj.g_t)'*(kiobj.obs_cov\(kiobj.g_bar[i] - kiobj.g_t))
    errors[3, i] = norm(kiobj.θθ_cov[i])   
    
  end
  errors[3, 1] = norm(θθ0_cov) 
  
  fig, (ax1, ax2, ax3) = PyPlot.subplots(ncols=3, figsize=(18,6))
  ax1.plot(ites, errors[1, :], "-o",  fillstyle="none", markevery=5, label= "UKI")
  ax1.set_xlabel("Iterations")
  ax1.set_ylabel("Relative L₂ norm error")
  ax1.legend()
  ax1.grid(true)
  
  ax2.plot(ites, errors[2, :], "-o", fillstyle="none", markevery=5, label= "UKI")
  ax2.set_xlabel("Iterations")
  ax2.set_ylabel("Optimization error")
  ax2.grid(true)
  ax2.legend()

  ax3.plot(ites, errors[3, :], "-o", fillstyle="none", markevery=5, label= "UKI")
  ax3.set_xlabel("Iterations")
  ax3.set_ylabel("Frobenius norm")
  ax3.grid(true)
  ax3.legend()
  fig.tight_layout()
  fig.savefig("Figs/NS-UQ-Loss.pdf")
  fig.close()



  # plot 3std for initial variables
  fig, ax = PyPlot.subplots(figsize=(18,6))
  ki_θ_bar  = kiobj.θ_bar[end]
  ki_θθ_cov = kiobj.θθ_cov[end]
  ki_θθ_std = sqrt.(diag(kiobj.θθ_cov[end]))
  θ_ind = Array(1:length(ki_θ_bar))
  ax.plot(θ_ind , ki_θ_bar,"-*", color="red", fillstyle="none",  label="UKI")
  ax.plot(θ_ind , ki_θ_bar + 3.0*ki_θθ_std, color="red")
  ax.plot(θ_ind , ki_θ_bar - 3.0*ki_θθ_std, color="red")
  Random.seed!(123);
  nx, ny = mesh.nx, mesh.ny
  θ_ref = rand(Normal(0, phys_params.σ), nx, ny, 2)
  abk = reshape(θ, Int64(length(θ)/2), 2)
  ax.plot(θ_ind , θ_ref, "--o", color="grey", fillstyle="none", label="Reference")
  ax.legend()
  ax.set_xlabel("θ indices")
  fig.tight_layout()
  fig.savefig("SpectralNS-2d-uq.png")
  fig.close()

  
  return Initial_ω0_cov_KL(mesh, θ_bar[end], kiobj.θθ_cov[end], seq_pairs)
  
  
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





function Plot_Field(mesh,  ω0,  filename, vmin=-5.0, vmax=5.0)
  
  
  
  nx, ny = mesh.nx, mesh.ny
  xx, yy = mesh.xx, mesh.yy
  X,Y = repeat(xx, 1, ny), repeat(yy, 1, nx)'
  
  # fig, ax = PyPlot.subplots(figsize=(6,6))
  PyPlot.pcolormesh(X, Y, ω0, shading= "gouraud", cmap="jet", vmin=vmin, vmax =vmax)
  PyPlot.colorbar()
  PyPlot.axis("off")
  PyPlot.tight_layout()
  PyPlot.savefig("Figs/"*filename)
  PyPlot.close("all")
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
  seq_pairs = Compute_Seq_Pairs(na)
  t_cov = Array(Diagonal(fill(0.1^2, phys_params.n_data))) 
  θ0_bar = zeros(Float64, 2na)
  θθ0_cov = Array(Diagonal(fill(1.0, 2*na)))           # standard deviation
  
  N_iter = 50 
  
  mesh = Spectral_Mesh(phys_params.nx, phys_params.ny, phys_params.Lx, phys_params.Ly)
  
  ω0_ref, _, _ =  Generate_Data(phys_params, -1.0,  "Figs/NS-vor.")
  
  
  
  
  # data
  noise_level_per = 5
  noise_level = noise_level_per/100.0
  ω0_ref, θ_ref, t_mean =  Generate_Data(phys_params, noise_level, "None", 2*na)
  
  
  α_reg = 1.0
  ω0, std_ω0 = Random_Init_Test("ExKI", phys_params, seq_pairs, t_mean, t_cov, θ0_bar, θθ0_cov, α_reg, θ_ref, ω0_ref, N_iter)
  
  PyPlot.close("all")
  Plot_Field(mesh, ω0_ref, "NS-UQ-ref.pdf")
  Plot_Field(mesh, ω0, "NS-UQ-mean.pdf")
  Plot_Field(mesh, ω0 - ω0_ref, "NS-UQ-error.pdf", nothing, nothing)
  Plot_Field(mesh, 3*std_ω0, "NS-UQ-3std.pdf", nothing, nothing)
  
  
end

UQ_test()



