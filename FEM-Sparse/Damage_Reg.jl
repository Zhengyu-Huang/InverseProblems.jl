using NNFEM
using JLD2
using Statistics
using LinearAlgebra




include("../Inversion/Plot.jl")
include("../Inversion/KalmanInversion.jl")
include("../Inversion/RWMCMC.jl")
include("Damage.jl")


function Forward(phys_params::Params, θ_c::Array{Float64,1})
  
  _, data = Run_Damage(phys_params, "Piecewise", θ_c)
  return data
end


function aug_Forward(phys_params::Params, θ_c::Array{Float64,1})
  _, data = Run_Damage(phys_params, "Piecewise", θ_c)
  return [data; sqrt.(abs.(θ))]
end


function Visual_Block_E(block::Array{Int64, 2}, state::Array{Float64, 2}, Qoi::Array{Float64, 1}, vmin::Float64, vmax::Float64, ax = nothing)
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



function Plot_E_Field(phys_params, state, Qoi,  E_max, ax)
  vmin, vmax = 0.0, E_max
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
  n_test = 1
  ns, ns_obs, porder, problem, ns_c, porder_c = 8, 5, 2, "Static", 2, 2
  
  phys_params_fine = Params(ns, ns_obs, porder, problem, ns_c, porder_c, n_test)
  
  nodes_fine, _, _, _, _, _, _, _ = Construct_Mesh(phys_params_fine.ns, phys_params_fine.porder, phys_params_fine.ls, phys_params_fine.ngp, phys_params_fine.prop, phys_params_fine.P1, phys_params_fine.P2, phys_params_fine.problem, phys_params_fine.T)
  E_max = phys_params_fine.prop["E"]
  θ_dam_fine_ref, t_mean_fine =  Run_Damage(phys_params_fine, "Analytic", nothing, phys_params_fine.P1, phys_params_fine.P2, "Figs/Damage-disp", "Figs/Damage-E")
  vmin = minimum((1.0 .- θ_dam_fine_ref)*E_max)
  

  ns, ns_obs, porder, problem, ns_c, porder_c = 8, 5, 2, "Static", 2, 2
  phys_params = Params(ns, ns_obs, porder, problem, ns_c, porder_c, n_test)
  nodes, _, _, _, _, _, _, _ = Construct_Mesh(phys_params.ns, phys_params.porder, phys_params.ls, phys_params.ngp, phys_params.prop, phys_params.P1, phys_params.P2, phys_params.problem, phys_params.T)
  E_max = phys_params.prop["E"]

  nθ = size(phys_params.domain_c.nodes, 1)
  θ0_mean = zeros(Float64, nθ)
  θθ0_cov = Array(Diagonal(fill(1.0, nθ)))           # standard deviation
  N_iter = 5

  fig_y, ax_y = PyPlot.subplots(ncols = 1, nrows=1, sharex=true, sharey=true, figsize=(7,4))
  
  fig_logk, ax_logk = PyPlot.subplots(ncols = 4, nrows=4, sharex=true, sharey=true, figsize=(32,16))
  for ax in ax_logk ;  ax.set_xticks([]) ; ax.set_yticks([]);  end
  
  noise_level = 5/100.0
  y = copy(t_mean_fine)
  
  @info "N_y is ", length(y), " N_θ is ", nθ
  σ_η = noise_level * maximum(abs.(y))
  Σ_η = Array(Diagonal(fill(σ_η^2, length(y)))) 
  # Σ_η = Array(Diagonal(fill(0.01, length(y)))) 

  # if noise_level > 0.0
  #   noise = similar(t_mean_fine)
  #   Random.seed!(123);
  #   for i = 1:length(t_mean_fine)
  #       noise[i] = rand(Normal(0, σ_η))
  #   end
  #   y .+= noise
  # end  
  
  ax_y.plot(t_mean_fine, color = "red", label = "y (truth)")
  ax_y.plot(y, "o", fillstyle = "none", color = "red", label = "y (observation)")


  ### Extended System
  # phys_params.N_y += nθ
  # aug_y = [y; θ0_mean]
  # λ_cov = Array(Diagonal(fill(1.0, nθ)))
  # aug_Σ_η = [Σ_η zeros(Float64, length(y), nθ); zeros(Float64, nθ, length(y)) λ_cov]

  

  α_reg = 1.0
  update_freq = 0

  aug_ukiobj = UKI_Run(phys_params, Forward, 
    θ0_mean, θθ0_cov,
    y, Σ_η,
    α_reg,
    update_freq,
    N_iter)
    

  θ_dam = Get_θ_Dam_From_Raw(phys_params.domain_c, phys_params.interp_e, phys_params.interp_sdata, aug_ukiobj.θ_mean[end])
  Plot_E_Field(phys_params, nodes, (1.0 .- θ_dam)*E_max,  E_max, ax_logk[1]; vmin = vmin)

  ax_y.plot(aug_ukiobj.y_pred[1][1:length(y)], "--", label = "UKI (initial)")
  ax_y.plot(aug_ukiobj.y_pred[end][1:length(y)], "--", fillstyle = "none", label = "UKI")
  
    
    
  im = Plot_E_Field(phys_params_fine, nodes_fine, (1.0 .- θ_dam_fine_ref)*E_max,  E_max, ax_logk[3]; vmin = vmin)
  fig_logk.tight_layout()
  cbar_ax = fig_logk.add_axes([0.90, 0.05, 0.02, 0.5])
  fig_logk.colorbar(im, cbar_ax)
  
  fig_logk.savefig("Figs/Damage.pdf")
  close(fig_logk)
   

  
  
  
end


Compare()