using NNFEM
using JLD2
using Statistics
using LinearAlgebra



include("../Plot.jl")
include("Damage.jl")
include("../RExKI.jl")
include("../REKI.jl")


function Foward(phys_params::Params, θ_c::Array{Float64,1})
  
  _, data = Run_Damage(phys_params, "Piecewise", θ_c)
  return data
end


function Ensemble(phys_params::Params,  params_i::Array{Float64, 2})
  n_data = phys_params.n_data
  
  N_ens,  N_θ = size(params_i)
  
  g_ens = zeros(Float64, N_ens,  n_data)
  
  Threads.@threads for i = 1:N_ens 
    # g: N_ens x N_data
    g_ens[i, :] .= Foward(phys_params, params_i[i, :])
  end
  
  return g_ens
end


function Damage_Test(method::String, phys_params::Params, 
  t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
  θ0_bar::Array{Float64,1}, θθ0_cov::Array{Float64,2}, 
  N_ens::Int64, α_reg::Float64, 
  θ_dam_ref::Array{Float64,1}, N_iter::Int64, ax1, ax2)
  
  
  if method == "ExKI"
    label = "UKI"
    kiobj = ExKI(phys_params, t_mean, t_cov,  θ0_bar, θθ0_cov, α_reg, θ_dam_ref, N_iter)
    θ_bar = kiobj.θ_bar
    linestyle, marker = "-", "o"
  elseif method =="EnKI"
    label = "EKI"
    kiobj = EnKI(phys_params, t_mean, t_cov,  θ0_bar, θθ0_cov, N_ens, α_reg, θ_dam_ref, N_iter)
    θ_bar = [dropdims(mean(kiobj.θ[i], dims=1), dims=1) for i = 1:N_iter ]  
    linestyle, marker = "--", "s" 
  else
    error("method: ", method, "has not implemented")
  end
  
  
  label = label*" (α = "*string(α_reg)*")"
  
  ites = Array(LinRange(1, N_iter, N_iter))
  errors = zeros(Float64, (2, N_iter))
  for i = 1:N_iter
    
    θ_dam = Get_θ_Dam_From_Raw(phys_params.domain_c, phys_params.interp_e, phys_params.interp_sdata, θ_bar[i])
    
    
    errors[1, i] = norm(θ_dam_ref - θ_dam)/norm(θ_dam_ref)
    errors[2, i] = 0.5*(kiobj.g_bar[i] - kiobj.g_t)'*(kiobj.obs_cov\(kiobj.g_bar[i] - kiobj.g_t))
    
  end
  
  
  if (!isnothing(ax1)  &&  !isnothing(ax2))
    ax1.plot(ites, errors[1, :], linestyle=linestyle, marker=marker, fillstyle="none", markevery=10, label= label)
    ax1.set_ylabel("Relative L₂ norm error")
    ax1.grid(true)
    
    
    ax2.semilogy(ites, errors[2, :], linestyle=linestyle, marker = marker,fillstyle="none", markevery=10, label= label)
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Optimization error")
    ax2.grid(true)
    ax2.legend()
  end
  
  return Get_θ_Dam_From_Raw(phys_params.domain_c, phys_params.interp_e, phys_params.interp_sdata, θ_bar[end])
  
  
end

function ExKI(phys_params::Params, 
  t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
  θ0_bar::Array{Float64,1}, θθ0_cov::Array{Float64,2}, 
  α_reg::Float64, 
  θ_dam_ref::Array{Float64,1}, N_iter::Int64 = 100)
  
  
  parameter_names = ["E"]
  
  ens_func(θ_ens) = Ensemble(phys_params, θ_ens)
  
  
  exkiobj = ExKIObj(parameter_names,
  θ0_bar, 
  θθ0_cov,
  t_mean, # observation
  t_cov,
  α_reg)
  
  
  for i in 1:N_iter
    
    params_i = deepcopy(exkiobj.θ_bar[end])
    
    θ_dam = Get_θ_Dam_From_Raw(phys_params.domain_c, phys_params.interp_e, phys_params.interp_sdata, params_i)
    
    @info "θ error :", norm(θ_dam_ref - θ_dam), " / ",  norm(θ_dam_ref)
    
    update_ensemble!(exkiobj, ens_func) 
    
    @info "data_mismatch :", (exkiobj.g_bar[end] - exkiobj.g_t)'*(exkiobj.obs_cov\(exkiobj.g_bar[end] - exkiobj.g_t))
    
  end
  
  return exkiobj
end

function UKI(phys_params::Params, 
  t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
  θ0_bar::Array{Float64,1}, θθ0_cov::Array{Float64,2}, 
  α_reg::Float64, 
  θ_dam_ref::Array{Float64,1}, N_iter::Int64 = 100)
  
  
  parameter_names = ["E"]
  
  ens_func(θ_ens) = Ensemble(phys_params, θ_ens)
  
  ukiobj = UKIObj(parameter_names,
  θ0_bar, 
  θθ0_cov,
  t_mean, # observation
  t_cov,
  α_reg)
  
  
  for i in 1:N_iter
    
    params_i = deepcopy(ukiobj.θ_bar[end])
    
    θ_dam = Get_θ_Dam_From_Raw(phys_params.domain_c, phys_params.interp_e, phys_params.interp_sdata, params_i)
    
    @info "θ error :", norm(θ_dam_ref - θ_dam), " / ",  norm(θ_dam_ref)
    
    update_ensemble!(ukiobj, ens_func) 
    
    @info "data_mismatch :", (ukiobj.g_bar[end] - ukiobj.g_t)'*(ukiobj.obs_cov\(ukiobj.g_bar[end] - ukiobj.g_t))
    
  end
  
  return ukiobj
end


function EnKI(phys_params::Params, 
  t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
  θ0_bar::Array{Float64,1}, θθ0_cov::Array{Float64,2}, 
  N_ens::Int64, α_reg::Float64, 
  θ_dam_ref::Array{Float64,1}, N_iter::Int64 = 100)
  
  
  parameter_names = ["E"]
  
  ens_func(θ_ens) = Ensemble(phys_params, θ_ens)
  Random.seed!(123)
  initial_params = Array(rand(MvNormal(θ0_bar, θθ0_cov), N_ens)')
  
  ekiobj = EKIObj(parameter_names,
  initial_params, 
  θ0_bar, 
  θθ0_cov,
  t_mean, # observation
  t_cov,
  α_reg)
  
  
  for i in 1:N_iter
    
    params_i = dropdims(mean(ekiobj.θ[end], dims=1), dims=1) 
    
    θ_dam = Get_θ_Dam_From_Raw(phys_params.domain_c, phys_params.interp_e, phys_params.interp_sdata, params_i)
    
    @info "θ error :", norm(θ_dam_ref - θ_dam), " / ",  norm(θ_dam_ref)
    
    update_ensemble!(ekiobj, ens_func) 
    
    @info "data_mismatch :", (ekiobj.g_bar[end] - ekiobj.g_t)'*(ekiobj.obs_cov\(ekiobj.g_bar[end] - ekiobj.g_t))
    
  end
  
  return ekiobj
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
  
  ns, ns_obs, porder, problem, ns_c, porder_c = 8, 5, 2, "Static", 2, 2
  phys_params = Params(ns, ns_obs, porder, problem, ns_c, porder_c)
  
  
  nodes, _, _, _, _, _, _, _ = Construct_Mesh(phys_params.ns, phys_params.porder, phys_params.ls, phys_params.ngp, phys_params.prop, phys_params.P1, phys_params.P2, phys_params.problem, phys_params.T)
  E_max = phys_params.prop["E"]
  
  nθ = size(phys_params.domain_c.nodes, 1)
  θ0_bar = zeros(Float64, nθ)
  θθ0_cov = Array(Diagonal(fill(1.0, nθ)))           # standard deviation
  N_iter = 50
  N_ens = 500

  
  θ_dam_ref, t_mean =  Run_Damage(phys_params, "Analytic", nothing,  "Figs/Damage-disp", "Figs/Damage-E")
  
  
  fig_logk, ax_logk = PyPlot.subplots(ncols = 3, nrows=4, sharex=true, sharey=true, figsize=(24,16))
  for ax in ax_logk ;  ax.set_xticks([]) ; ax.set_yticks([]);  end
  
  
  for noise_level_per in [0, 1, 5]
    
    noise_level = noise_level_per/100.0
    
    
    θ_dam_ref, t_mean =  Run_Damage(phys_params, "Analytic", nothing,  "None",  "None", noise_level)
    
    
    t_cov = Array(Diagonal(fill(0.01, length(t_mean)))) 
    
    
    fig, (ax1, ax2) = PyPlot.subplots(nrows=2, sharex=true, figsize=(8,12))
    α_reg = 1.0

    θ_bar = Damage_Test("ExKI", phys_params, t_mean, t_cov, θ0_bar, θθ0_cov, N_ens, α_reg, θ_dam_ref, N_iter, ax1, ax2)
    if noise_level_per == 0
      Plot_E_Field(phys_params,nodes, (1.0 .- θ_bar)*E_max,  E_max, ax_logk[1])
    end

    θ_bar = Damage_Test("EnKI", phys_params, t_mean, t_cov, θ0_bar, θθ0_cov, N_ens, α_reg, θ_dam_ref, N_iter, ax1, ax2)
    if noise_level_per == 0
      Plot_E_Field(phys_params,nodes, (1.0 .- θ_bar)*E_max,  E_max, ax_logk[5])
    end
    
    
    
    
    
    α_reg = 0.5
    θ_bar = Damage_Test("ExKI", phys_params, t_mean, t_cov, θ0_bar, θθ0_cov, N_ens, α_reg, θ_dam_ref, N_iter, ax1, ax2)
    if noise_level_per != 0
      ax_id = (noise_level_per == 1 ? 2 : 3 ;)
      Plot_E_Field(phys_params,nodes, (1.0 .- θ_bar)*E_max,  E_max, ax_logk[ax_id])
    end

    θ_bar = Damage_Test("EnKI", phys_params, t_mean, t_cov, θ0_bar, θθ0_cov, N_ens, α_reg, θ_dam_ref, N_iter, ax1, ax2)
    if noise_level_per != 0
      ax_id = (noise_level_per == 1 ? 6 : 7 ;)
      Plot_E_Field(phys_params,nodes, (1.0 .- θ_bar)*E_max,  E_max, ax_logk[ax_id])
    end
    
    
    
    if noise_level_per == 5
      α_reg = 0.0
      θ_bar = Damage_Test("ExKI", phys_params, t_mean, t_cov, θ0_bar, θθ0_cov, N_ens, α_reg, θ_dam_ref, N_iter, ax1, ax2)
      if noise_level_per != 0
        Plot_E_Field(phys_params,nodes, (1.0 .- θ_bar)*E_max,  E_max, ax_logk[4])
      end

      θ_bar = Damage_Test("EnKI", phys_params, t_mean, t_cov, θ0_bar, θθ0_cov, N_ens, α_reg, θ_dam_ref, N_iter, ax1, ax2)
      if noise_level_per != 0
        Plot_E_Field(phys_params,nodes, (1.0 .- θ_bar)*E_max,  E_max, ax_logk[8])
      end

    end
    
    
    
    
    
    fig.tight_layout()
    fig.savefig("Figs/Damage-noise-"*string(noise_level_per)*".pdf")
    close(fig)
  end
  
  Plot_E_Field(phys_params,nodes, (1.0 .- θ_dam_ref)*E_max,  E_max, ax_logk[9])
  im = Plot_E_Field(phys_params,nodes, (1.0 .- θ_dam_ref)*E_max,  E_max, ax_logk[10])
  Plot_E_Field(phys_params,nodes, (1.0 .- θ_dam_ref)*E_max,  E_max, ax_logk[11])
  Plot_E_Field(phys_params,nodes, (1.0 .- θ_dam_ref)*E_max,  E_max, ax_logk[12])
  
  
  fig_logk.tight_layout()
  cbar_ax = fig_logk.add_axes([0.90, 0.05, 0.02, 0.5])
  fig_logk.colorbar(im, cbar_ax)
  
  fig_logk.savefig("Figs/Damage.pdf")
  close(fig_logk)
  
  
  
end


Compare()