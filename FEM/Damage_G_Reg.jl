using NNFEM
using JLD2
using Statistics
using LinearAlgebra



include("../Inversion/Plot.jl")
include("../Inversion/KalmanInversion.jl")
include("../Inversion/RWMCMC.jl")
include("Damage.jl")


function Forward(phys_params::Params, θ_c::Array{Float64,1}, U, sqrt_Σ)
  _, data1 = Run_Damage(phys_params, "Piecewise", U * (sqrt_Σ .* θ_c), phys_params.P1, phys_params.P2)
  #_, data2 = Run_Damage(phys_params, "Piecewise", U * (sqrt_Σ .* θ_c), -phys_params.P1, phys_params.P2)
  #return [data1; data2]

  return data1
end

function Forward_Analytic(phys_params::Params, disp::String, damage::String)
  θ_dam, data1 = Run_Damage(phys_params, "Analytic", nothing,  phys_params.P1, phys_params.P2, disp*"-1", damage)
  # θ_dam, data2 = Run_Damage(phys_params, "Analytic", nothing, -phys_params.P1, phys_params.P2, disp*"-2", damage)
  # return θ_dam, [data1; data2]

  return θ_dam, data1
end

function Forward_Analytic(phys_params::Params, θ_c::Array{Float64,1}, U, sqrt_Σ, disp::String, damage::String)
  θ_dam, data1 = Run_Damage(phys_params, "Piecewise", U * (sqrt_Σ .* θ_c), phys_params.P1, phys_params.P2, disp*"-1", damage)# θ_dam, data2 = Run_Damage(phys_params, "Analytic", nothing, -phys_params.P1, phys_params.P2, disp*"-2", damage)
  # return θ_dam, [data1; data2]

  return θ_dam, data1
end


function aug_Forward(phys_params::Params, θ_c::Array{Float64,1}, U, sqrt_Σ)
  _, data1 = Run_Damage(phys_params, "Piecewise", U * (sqrt_Σ .* θ_c),  phys_params.P1, phys_params.P2)
  # _, data2 = Run_Damage(phys_params, "Piecewise", U * (sqrt_Σ .* θ_c), -phys_params.P1, phys_params.P2)
  # return [data1; data2; θ_c]
  return [data1; θ_c]
end


function Converge_Plot(kiobj)
  
  fig_loss, ax_loss = PyPlot.subplots(ncols = 2, nrows=1, sharex=true, sharey=false, figsize=(12,5))
  ax1, ax2 = ax_loss[1], ax_loss[2]

  N_iter = min(length(kiobj.y_pred), length(kiobj.θθ_cov))
  errors = zeros(Float64, N_iter)
  Fnorm = zeros(Float64, N_iter)

  for i = 1:N_iter
    errors[i] = 0.5*(kiobj.y_pred[i] - kiobj.y)'*(kiobj.Σ_η\(kiobj.y_pred[i] - kiobj.y))
    Fnorm[i] = norm(kiobj.θθ_cov[i])
  end
  
  ites = Array(0 : N_iter-1)

    
  ax1.semilogy(ites, errors, "-o", fillstyle="none", markevery=2)
  ax1.set_xlabel("Iterations")
  ax1.set_ylabel("Optimization error")
  ax1.grid(true)


  ax2.semilogy(ites, Fnorm, "-o", fillstyle="none", markevery=2)
  ax2.set_xlabel("Iterations")
  ax2.set_ylabel("Frobenius norm")
  ax2.grid(true)

  fig_loss.tight_layout()
  fig_loss.savefig("Figs/UKI-Loss.pdf")
end








function Visual_Block_E(block::Array{Int64, 2}, state::Array{Float64, 2}, Qoi::Array{Float64, 1}, vmin, vmax, ax = nothing)
  nbx, nby = size(block)
  X = zeros(Float64, nbx, nby)
  Y = zeros(Float64, nbx, nby)
  C = zeros(Float64, nbx, nby)
  
  for i = 1:nbx
    for j = 1:nby
      n_id = block[i,j]
      X[i,j] = state[n_id,1] 
      Y[i,j] = state[n_id,2] 
      C[i,j] = Qoi[n_id]
    end
  end
  
  ax.pcolormesh(X, Y, C, shading ="gouraud", cmap="jet", vmin=vmin, vmax=vmax)
end

function Plot_E_Field(phys_params, state, Qoi,  E_max, ax; vmin = 0.0)
  vmax = E_max
  ns, porder = phys_params.ns, phys_params.porder
  
  
  block = zeros(Int64, ns*porder+1, ns*porder+1)
  for i = 1:ns*porder+1
    start = 1+(i-1)*(2*ns*porder+2)
    block[i, :] .= start: start + ns*porder
  end
  Visual_Block_E(block, state, Qoi, vmin, vmax, ax)
  
  
  block = zeros(Int64, ns*porder+1, ns*porder+1)
  for i = 1:ns*porder
    start = ns*porder+2+(i-1)*(2*ns*porder+2)
    block[i, :] .= start:start + ns*porder
  end
  start = 1 + (2*ns*porder+2)*(ns*porder) + 3*ns*porder
  block[ns*porder+1, :] .= start: start + ns*porder
  Visual_Block_E(block, state, Qoi, vmin, vmax, ax)
  
  
  
  block = zeros(Int64, ns*porder+1, 4*ns*porder+1)
  for i = 1:ns*porder+1
    start = (2*ns*porder+2)*ns*porder +1 + (i-1)*(4*ns*porder+1)
    block[i, :] .= start : start + 4*ns*porder
  end
  
  Visual_Block_E(block, state, Qoi, vmin, vmax, ax)
  
  
end

###############################################################################################

function Compare()

  # n_test = 2
  n_test = 1
  # fine scale , avoid Bayesian Crime
  ns, ns_obs, porder, problem, ns_c, porder_c = 16, 5, 2, "Static", 16, 2
  
  phys_params_fine = Params(ns, ns_obs, porder, problem, ns_c, porder_c, n_test)
  
  nodes_fine, _, _, _, _, _, _, _ = Construct_Mesh(phys_params_fine.ns, phys_params_fine.porder, phys_params_fine.ls, phys_params_fine.ngp, phys_params_fine.prop, phys_params_fine.P1, phys_params_fine.P2, phys_params_fine.problem, phys_params_fine.T)
  E_max = phys_params_fine.prop["E"]
  θ_dam_fine_ref, t_mean_fine =  Forward_Analytic(phys_params_fine,  "Figs/Damage-disp-fine", "Figs/Damage-E-fine")
  
  nθ = 200
  vmin = minimum((1.0 .- θ_dam_fine_ref)*E_max)
  # coarse scale 

  ns, ns_obs, porder, problem, ns_c, porder_c = 8, 5, 2, "Static", 8, 2
  # ns, ns_obs, porder, problem, ns_c, porder_c = 4, 5, 2, "Static", 4, 2

  phys_params = Params(ns, ns_obs, porder, problem, ns_c, porder_c, n_test)
  nodes, _, _, _, _, _, _, _ = Construct_Mesh(phys_params.ns, phys_params.porder, phys_params.ls, phys_params.ngp, phys_params.prop, phys_params.P1, phys_params.P2, phys_params.problem, phys_params.T)
  E_max = phys_params.prop["E"]
  
  θ0_mean = zeros(Float64, nθ)
  θθ0_cov = Array(Diagonal(fill(1.0, nθ)))           # standard deviation
  N_iter = 20
  
  fig_y, ax_y = PyPlot.subplots(ncols = 1, nrows=1, sharex=true, sharey=true, figsize=(7,4))
  
  fig_logk, ax_logk = PyPlot.subplots(ncols = 3, nrows=1, sharex=true, sharey=true, figsize=(10,2))
  for ax in ax_logk ;  ax.set_xticks([]) ; ax.set_yticks([]);  end 
  
  noise_level_per = 5
  noise_level = noise_level_per/100.0
  y = copy(t_mean_fine)

  @info "N_y is ", length(y), " N_θ is ", nθ
  σ_η = noise_level * maximum(abs.(y))
  Σ_η = Array(Diagonal(fill(σ_η^2, length(y)))) 
  if noise_level > 0.0
    noise = similar(t_mean_fine)
    Random.seed!(123);
    for i = 1:length(t_mean_fine)
        noise[i] = rand(Normal(0, σ_η))
    end
    y .+= noise
  end

  ax_y.plot(t_mean_fine, color = "red", label = "y (truth)")
  ax_y.plot(y, "o", fillstyle = "none", color = "red", label = "y (observation)")

  # construct basis 
  domain_c = phys_params.domain_c
  # it is very sensity to τ
  σ, s0, τ =  10.0, 100.0, 1.0
  _, U, Σ = Kernel_function(domain_c, σ, s0, τ)
  sqrt_Σ = sqrt.(Σ)
  U, sqrt_Σ = U[:, 1:nθ] , sqrt_Σ[1:nθ] 
  
  ### Extended System
  phys_params.N_y += nθ
  aug_y = [y; θ0_mean]
  aug_Σ_η = [Σ_η zeros(Float64, length(y), nθ); zeros(Float64, nθ, length(y))  θθ0_cov]

  α_reg = 1.0
  update_freq = 1
  func = (phys_params, θ_c) -> aug_Forward(phys_params, θ_c, U, sqrt_Σ)
  # aug_ukiobj = UKI_Run(phys_params, func, 
  #   θ0_mean, θθ0_cov,
  #   aug_y, aug_Σ_η,
  #   α_reg,
  #   update_freq,
  #   N_iter)
  # @save "aug_ukiobj.dat" aug_ukiobj

  @load "aug_ukiobj.dat" aug_ukiobj

  θ_dam = Get_θ_Dam_From_Raw(phys_params.domain_c, phys_params.interp_e, phys_params.interp_sdata, U * (sqrt_Σ .* aug_ukiobj.θ_mean[end]))
  Plot_E_Field(phys_params, nodes, (1.0 .- θ_dam)*E_max,  E_max, ax_logk[1]; vmin = vmin)

  ax_y.plot(aug_ukiobj.y_pred[1][1:length(y)], "--", label = "UKI (initial)")
  ax_y.plot(aug_ukiobj.y_pred[end][1:length(y)], "--", fillstyle = "none", label = "UKI")
  
  
  
  Converge_Plot(aug_ukiobj)

  ### MCMC
  θ0 = aug_ukiobj.θ_mean[end]
  func(phys_params, θ_c) = Forward(phys_params, θ_c, U, sqrt_Σ)
  log_likelihood_func(θ) = log_likelihood(phys_params, θ, func, y,  Σ_η)
  n_iter =2*10^5
  θs = PCN_Run(log_likelihood_func, θ0, θθ0_cov, 0.1, n_iter)
  @save "mcmc.dat" θs

  n_burn_in = div(n_iter, 2)
  mcmc_θ_mean = dropdims( sum(θs[n_burn_in:end, :], dims = 1)/size(θs[n_burn_in:end, :], 1) , dims=1)
  θ_dam = Get_θ_Dam_From_Raw(phys_params.domain_c, phys_params.interp_e, phys_params.interp_sdata, U * (sqrt_Σ .* mcmc_θ_mean) )
  Plot_E_Field(phys_params, nodes, (1.0 .- θ_dam)*E_max,  E_max, ax_logk[2]; vmin = vmin)


  y_mcmc = Forward(phys_params, mcmc_θ_mean, U, sqrt_Σ)
  ax_y.plot(y_mcmc[1:length(y)], linestyle="dashdot", fillstyle = "none", label = "MCMC")

  ax_y.legend()
  ax_y.set_xlabel("Measurement")
  ax_y.set_ylabel("Displacement")

  fig_y.tight_layout()
  fig_y.savefig("Figs/Y.pdf")
  close(fig_y)

  ### Reference
  
  im = Plot_E_Field(phys_params_fine, nodes_fine, (1.0 .- θ_dam_fine_ref)*E_max,  E_max, ax_logk[3]; vmin = vmin)
  fig_logk.tight_layout()
  cbar_ax = fig_logk.add_axes([0.90, 0.05, 0.02, 0.5])
  fig_logk.colorbar(im, cbar_ax)
  
  fig_logk.savefig("Figs/Damage.pdf")
  close(fig_logk)

end

##### Convergence test
function Convergence_Test()
  # fine scale 
  ns, ns_obs, porder, problem, ns_c, porder_c = 8, 9, 2, "Static", 8, 2
  phys_params_fine = Params(ns, ns_obs, porder, problem, ns_c, porder_c)
  
  nodes, _, _, _, _, _, _, _ = Construct_Mesh(phys_params_fine.ns, phys_params_fine.porder, phys_params_fine.ls, phys_params_fine.ngp, phys_params_fine.prop, phys_params_fine.P1, phys_params_fine.P2, phys_params_fine.problem, phys_params_fine.T)
  E_max = phys_params_fine.prop["E"]
  θ_dam_fine_ref, t_mean_fine =  Run_Damage(phys_params_fine, "Analytic", nothing,  phys_params_fine.P1, phys_params_fine.P2, "Figs/Damage-disp-fine", "Figs/Damage-E-fine")
  
  # coarse scale 
  ns, ns_obs, porder, problem, ns_c, porder_c = 4, 9, 2, "Static", 4, 2
  phys_params_coarse = Params(ns, ns_obs, porder, problem, ns_c, porder_c)
  nodes, _, _, _, _, _, _, _ = Construct_Mesh(phys_params_coarse.ns, phys_params_coarse.porder, phys_params_coarse.ls, phys_params_coarse.ngp, phys_params_coarse.prop, phys_params_coarse.P1, phys_params_coarse.P2, phys_params_coarse.problem, phys_params_coarse.T)
  E_max = phys_params_coarse.prop["E"]
  θ_dam_coarse_ref, t_mean_coarse =  Run_Damage(phys_params_coarse, "Analytic", nothing, phys_params_coarse.P1, phys_params_coarse.P2, "Figs/Damage-disp-coarse", "Figs/Damage-E-coarse")
  
  plot(t_mean_coarse)
  plot(t_mean_fine, "--")
  savefig("t_mean.png")

  domain_c = phys_params_coarse.domain_c
  σ, s0, τ = 1.0, phys_params_coarse.ls/(ns_c * porder_c), 2.0
  U, Σ = Kernel_function(domain_c, σ, s0, τ)


  fig_logk, ax_logk = PyPlot.subplots(ncols = 3, nrows=1, sharex=true, sharey=true, figsize=(12,2))
  θ_dam =  Get_θ_Dam_From_Raw(phys_params_coarse.domain_c, phys_params_coarse.interp_e, phys_params_coarse.interp_sdata, U[:, 1]*Σ[1])
  Plot_E_Field(phys_params_coarse, nodes, (1.0 .- θ_dam)*E_max,  E_max, ax_logk[1])
  θ_dam =  Get_θ_Dam_From_Raw(phys_params_coarse.domain_c, phys_params_coarse.interp_e, phys_params_coarse.interp_sdata, U[:, 2]*Σ[1])
  Plot_E_Field(phys_params_coarse, nodes, (1.0 .- θ_dam)*E_max,  E_max, ax_logk[2])
  θ_dam =  Get_θ_Dam_From_Raw(phys_params_coarse.domain_c, phys_params_coarse.interp_e, phys_params_coarse.interp_sdata, U[:, 3]*Σ[1])
  Plot_E_Field(phys_params_coarse, nodes, (1.0 .- θ_dam)*E_max,  E_max, ax_logk[3])


  fig_logk, ax_logk = PyPlot.subplots(ncols = 3, nrows=1, sharex=true, sharey=true, figsize=(12,2))
  θ_dam =  Get_θ_Dam_From_Raw(phys_params_coarse.domain_c, phys_params_coarse.interp_e, phys_params_coarse.interp_sdata, U[:, 1]*sqrt(Σ[1]))
  vmax, vmin = maximum(θ_dam), minimum(θ_dam)
  Plot_E_Field(phys_params_coarse, nodes,  θ_dam,  vmax, ax_logk[1]; vmin = vmin)
  θ_dam =  Get_θ_Dam_From_Raw(phys_params_coarse.domain_c, phys_params_coarse.interp_e, phys_params_coarse.interp_sdata, U[:, 2]*sqrt(Σ[2]))
  
  vmax, vmin = maximum(θ_dam), minimum(θ_dam)
  Plot_E_Field(phys_params_coarse, nodes,  θ_dam,  vmax, ax_logk[2]; vmin = vmin)
  θ_dam =  Get_θ_Dam_From_Raw(phys_params_coarse.domain_c, phys_params_coarse.interp_e, phys_params_coarse.interp_sdata, U[:, 3]*sqrt(Σ[3]))
  
  vmax, vmin = maximum(θ_dam), minimum(θ_dam)
  im = Plot_E_Field(phys_params_coarse, nodes,  θ_dam,  vmax, ax_logk[3]; vmin = vmin)
  fig_logk.colorbar(im)

end

# Convergence_Test()
Compare()