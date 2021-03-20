using JLD2
using Statistics
using LinearAlgebra



include("../Plot.jl")
include("Random_Init.jl")
include("../RExKI.jl")
include("../ModelError/Misfit2Diagcov.jl")
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




# generate truth (all Fourier modes) and observations
# observation are at frame 0, Δd_t, 2Δd_t, ... nt
# with sparse points at Array(1:Δd_x:nx) × Array(1:Δd_y:ny)
function Prediction_Helper(params::Params, θ::Array{Float64,1})
    
  ν = params.ν
  ub, vb = params.ub, params.vb
  nx, ny = params.nx, params.ny
  Lx, Ly = params.Lx, params.Ly
  nt, T = params.nt, params.T
  method = params.method 
  
  mesh = Spectral_Mesh(nx, ny, Lx, Ly)
  ω0 = Initial_ω0_KL(mesh, θ, seq_pairs)  

  
  mesh = Spectral_Mesh(nx, ny, Lx, Ly)
  
  fx,fy = Force(mesh)
  
  solver = SpectralNS_Solver(mesh, ν, fx, fy, ω0, ub, vb)  
  
  Δt = T/nt

  
  for i = 1:nt
      Solve!(solver, Δt, method)
  end

  Update_Grid_Vars!(solver , true)

  # return [solver.u[:, Int64(ny/2) + 1] ; solver.v[:, Int64(ny/2) + 1]; solver.ω[:, Int64(ny/2) + 1] ; solver.p[:, Int64(ny/2) + 1]]

  # return [solver.u[Int64(ny/2) + 1 ,:] ; solver.v[Int64(ny/2) + 1,:]; solver.ω[Int64(ny/2) + 1,:] ; solver.p[Int64(ny/2) + 1, :]]
  return [diag(solver.u) ; diag(solver.v); diag(solver.ω) ; diag(solver.p)]

end

# predict the velocity at future time at the central line of the domain
function Prediction(phys_params, kiobj, θ_mean, θθ_cov, θ_ref, T = 1, nt = 5000)

  phys_params.T = T
  phys_params.nt = nt
  nx, Lx = phys_params.nx, phys_params.Lx
  Δx = Lx/nx
  xx = LinRange(0, Lx-Δx, nx)

  obs_ref = Prediction_Helper(phys_params, θ_ref)

  N_ens = kiobj.N_ens
  θθ_cov = (θθ_cov+θθ_cov')/2 
  θ_p = construct_sigma_ensemble(kiobj, θ_mean, θθ_cov)
  obs = zeros(Float64, N_ens, 4nx)
  Threads.@threads for i = 1:N_ens
      θ = θ_p[i, :]
      obs[i, :] = Prediction_Helper(phys_params, θ)
  end

  obs_mean = obs[1, :]

  @info obs_mean
  obs_cov  = construct_cov(kiobj,  obs, obs_mean)  #+ Array(Diagonal( obs_noise_level^2 * obs_mean.^2)) 
  obs_std = sqrt.(diag(obs_cov))


  # optimization related plots
  fig_disp, ax_disp = PyPlot.subplots(ncols = 2, nrows=2, sharex=false, sharey=false, figsize=(16,12))
  markevery = 10
  # u velocity 
  ax_disp[1,1].plot(xx, obs_ref[1:nx], "--o", color="grey", fillstyle="none", label="Reference", markevery = markevery)
  ax_disp[1,1].plot(xx, obs_mean[1:nx], "-*r",  markevery = markevery, label="UKI")
  ax_disp[1,1].plot(xx, (obs_mean[1:nx] + 3obs_std[1:nx]),  "--r")
  ax_disp[1,1].plot(xx, (obs_mean[1:nx] - 3obs_std[1:nx]),  "--r")
  # v velocity
  ax_disp[1,2].plot(xx, obs_ref[nx+1:2nx], "--o", color="grey", fillstyle="none", label="Reference", markevery = markevery)
  ax_disp[1,2].plot(xx, obs_mean[nx+1:2nx], "-*r",  markevery = markevery, label="UKI")
  ax_disp[1,2].plot(xx, (obs_mean[nx+1:2nx] + 3obs_std[nx+1:2nx]),   "--r")
  ax_disp[1,2].plot(xx, (obs_mean[nx+1:2nx] - 3obs_std[nx+1:2nx]),   "--r")
  # ω 
  ax_disp[2,1].plot(xx, obs_ref[2nx+1:3nx], "--o", color="grey", fillstyle="none", label="Reference", markevery = markevery)
  ax_disp[2,1].plot(xx, obs_mean[2nx+1:3nx], "-*r",  markevery = markevery, label="UKI")
  ax_disp[2,1].plot(xx, (obs_mean[2nx+1:3nx] + 3obs_std[2nx+1:3nx]),   "--r")
  ax_disp[2,1].plot(xx, (obs_mean[2nx+1:3nx] - 3obs_std[2nx+1:3nx]),   "--r")
  # p 
  ax_disp[2,2].plot(xx, obs_ref[3nx+1:end], "--o", color="grey", fillstyle="none", label="Reference", markevery = markevery)
  ax_disp[2,2].plot(xx, obs_mean[3nx+1:end], "-*r",  markevery = markevery, label="UKI")
  ax_disp[2,2].plot(xx, (obs_mean[3nx+1:end] + 3obs_std[3nx+1:end]),   "--r")
  ax_disp[2,2].plot(xx, (obs_mean[3nx+1:end] - 3obs_std[3nx+1:end]),   "--r")
  
  ymin,ymax = -5.0, 10.0
  ax_disp[1,1].set_xlabel("X")
  ax_disp[1,1].set_ylabel("X-Velocity")
  ax_disp[1,1].set_ylim([ymin,ymax])
  ax_disp[1,1].legend()

  ax_disp[1,2].set_xlabel("X")
  ax_disp[1,2].set_ylabel("Y-Velocity")
  ax_disp[1,2].set_ylim([ymin,ymax])
  ax_disp[1,2].legend()
  
  ax_disp[2,1].set_xlabel("X")
  ax_disp[2,1].set_ylabel("Vorticity")
  ax_disp[2,1].set_ylim([ymin,ymax])
  ax_disp[2,1].legend()

  ax_disp[2,2].set_xlabel("X")
  ax_disp[2,2].set_ylabel("Pressure")
  ax_disp[2,2].set_ylim([ymin,ymax])
  ax_disp[2,2].legend()
  
  fig_disp.tight_layout()
  fig_disp.savefig("NS-Velocity.png")
  close(fig_disp)
  

end


###############################################################################################


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
  
  

  
  # 30
  N_iter = 30
  mesh = Spectral_Mesh(phys_params.nx, phys_params.ny, phys_params.Lx, phys_params.Ly)
  

  # compute only the first 10 modes
  na = 10
  N_θ = 2na

  na_ref = 50
  seq_pairs = Compute_Seq_Pairs(2na)

  seq_pairs_ref = Compute_Seq_Pairs(2*na_ref)
  ω0_ref, _, _ =  Generate_Data(phys_params, seq_pairs_ref, -1.0,  "Figs/NS-vor-perfect.", 2*na_ref)
  
 
  ω0_ref, θ_ref, t_mean_noiseless =  Generate_Data_Noiseless(phys_params, seq_pairs_ref, "None", 2*na_ref)
  θ_ref = reshape(θ_ref, na_ref, 2)[1:na, :][:]
  


  
  # data
  θ0_bar = zeros(Float64, 2*na)
  θθ0_cov = Array(Diagonal(fill(1.0, 2*na)))
  α_reg = 1.0
  noise_level = 0.05
  
  
  
  
  
  # observation
  t_cov = Array(Diagonal( fill(0.01^2, length(t_mean_noiseless)) ))
  # t_cov = Array(Diagonal(noise_level^2 * t_mean_noiseless.^2))
  Random.seed!(123);
  
  ites = Array(LinRange(1, N_iter, N_iter))
  errors = zeros(Float64, (3, N_iter))
  


  t_mean = copy(t_mean_noiseless)
  for i = 1:length(t_mean)
    noise = noise_level*t_mean[i] * rand(Normal(0, 1))
    t_mean[i] += noise
  end
    
  # # first round 
  # kiobj, mesh = Random_Init_Test(phys_params, seq_pairs, t_mean, t_cov, θ0_bar, θθ0_cov, α_reg, θ_ref, ω0_ref, N_iter)
  # # adjust model error , update t_cov
  # data_misfit = (kiobj.g_bar[end] - t_mean)
  # n_dm = length(kiobj.g_bar[end] - t_mean)
  # # @info "Mean error : ", sum(data_misfit)/n_dm, " Cov error : ", sum(data_misfit.^2)/n_dm
  # # @info "Mean max : ", maximum(abs.(data_misfit)), " Cov max : ", maximum(data_misfit.^2)
  # # new_cov =  sum(data_misfit.^2)/n_dm    # maximum(data_misfit.^2)
  # # # estimation of the constant model error, and re-train
  # # t_cov = Array(Diagonal(fill(new_cov, length(data_misfit))))

  # diag_cov = Misfit2Diagcov(2, data_misfit, t_mean)
  # t_cov = Array(Diagonal(diag_cov))

  # # second round 
  # kiobj, mesh = Random_Init_Test(phys_params, seq_pairs, t_mean, t_cov, θ0_bar, θθ0_cov, α_reg, θ_ref, ω0_ref, N_iter)
  # @save "exkiobj.dat" kiobj



  @load "exkiobj.dat" kiobj
  n_dm = length(kiobj.g_bar[end] - t_mean)
  


  Ny_Nθ = n_dm/length(kiobj.θ_bar[end])

  @info "Ny/Nθ is ", Ny_Nθ
  Prediction(phys_params, kiobj, kiobj.θ_bar[end], kiobj.θθ_cov[end]*Ny_Nθ, θ_ref)


  ω0 = Initial_ω0_KL(mesh, kiobj.θ_bar[end], seq_pairs)  
  Visual(mesh, ω0, "ω", "Predict.pdf", -5.0, 5.0)

# ########################################################

#   for i = 1:N_iter
    
#     ω0 = Initial_ω0_KL(mesh, kiobj.θ_bar[i], seq_pairs)
    
#     errors[1, i] = norm(ω0_ref - ω0)/norm(ω0_ref)
#     errors[2, i] = 0.5*(kiobj.g_bar[i] - kiobj.g_t)'*(kiobj.obs_cov\(kiobj.g_bar[i] - kiobj.g_t))
#     errors[3, i] = norm(kiobj.θθ_cov[i])   
    
#   end
#   errors[3, 1] = norm(θθ0_cov) 
  
#   ax_ite[1].semilogy(ites, errors[1, :], "-o", fillstyle="none", markevery=2 )
#   ax_ite[2].semilogy(ites, errors[2, :], "-o", fillstyle="none", markevery=2 )
#   ax_ite[3].semilogy(ites, errors[3, :], "-o", fillstyle="none", markevery=2 )
  
#   θ_ind = Array(1:N_θ)

#   ax_θ[n].plot(θ_ind , θ_ref[1:N_θ], "--o", color="grey", fillstyle="none", label="Reference")


#   ki_θ_bar  = kiobj.θ_bar[end]
#   ki_θθ_cov = kiobj.θθ_cov[end]
#   ki_θθ_std = sqrt.(diag(kiobj.θθ_cov[end]))
#   ax_θ[n].plot(θ_ind , ki_θ_bar,"-*", color="red", fillstyle="none")
#   ax_θ[n].plot(θ_ind , ki_θ_bar + 3.0*ki_θθ_std, "--", color="red")
#   ax_θ[n].plot(θ_ind , ki_θ_bar - 3.0*ki_θθ_std, "--", color="red")
#   # ax_θ[n].grid(true)
#   ax_θ[n].set_ylabel("θ")

#   ω0 = Initial_ω0_KL(mesh, kiobj.θ_bar[end], seq_pairs)
#   Plot_Field(mesh, ω0, ax_omega0[:][n])



