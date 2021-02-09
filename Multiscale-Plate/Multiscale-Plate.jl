using NNFEM
using JLD2
using Statistics
using Random
using LinearAlgebra
using Distributions



include("../Plot.jl")
include("CommonFuncs.jl")
include("../RExKI.jl")

mutable struct Params
    θ_name::Array{String, 1}
    θ_func::Function
    dθ_func::Function
    θ_func_inv::Function
    matlaw::String

    ρ::Float64
    tids::Array{Int64,1}
    force_scale::Float64
    
    NT::Int64
    T::Float64
    
    n_tids::Int64
    n_obs_point::Int64
    n_obs::Int64
    n_data::Int64
end

function Params(matlaw::String, tids::Array{Int64,1}, n_obs_point::Int64 = 2, n_obs_time::Int64 = 200, T::Float64 = 200.0, NT::Int64 = 200)
    if matlaw == "PlaneStressPlasticity"
        θ_name = ["E",   "nu",  "sigmaY", "K"] 
        # θ_func = (θ)-> [θ[1]*1.0e+5, 1/(2+3exp(θ[2])),  θ[3]*1.0e+3, θ[4]*1.0e+4]
        # θ_func_inv = (θ)-> [θ[1]/1.0e+5, log((1 - 2θ[2])/(3θ[2])),  θ[3]/1.0e+3, θ[4]/1.0e+4]
        # dθ_func = (θ)-> [1.0e+5        0                             0            0;
        #                  0      -3exp(θ[2])/(2+3exp(θ[2]))^2         0            0;
        #                  0             0                           1.0e+3         0;
        #                  0             0                              0         1.0e+4]
        θ_func = (θ)-> [θ[1]*1.0e+5, θ[2]*0.02,  θ[3]*1.0e+3, θ[4]*1.0e+4]
        θ_func_inv = (θ)-> [θ[1]/1.0e+5, θ[2]/0.02,  θ[3]/1.0e+3, θ[4]/1.0e+4]
        dθ_func = (θ)-> [1.0e+5        0               0            0;
                            0         0.02             0            0;
                            0             0         1.0e+3          0;
                            0             0            0         1.0e+4]

    else
        error("unrecognized matlaw: ", matlaw)
    end

    fiber_fraction = 0.25
    ρ = 4.5*(1 - fiber_fraction) + 3.2*fiber_fraction
    
    force_scale = 5.0
    n_tids = length(tids)
    n_data = 2n_obs_point * n_obs_time * n_tids
    
    return Params(θ_name, θ_func, dθ_func, θ_func_inv, matlaw, ρ, tids, force_scale, NT, T,  n_tids, n_obs_point, n_obs_point * n_obs_time, n_data)
end

function Foward(phys_params::Params, θ::Array{Float64,1})
    θ_func, ρ, tids, force_scale, n_data = phys_params.θ_func, phys_params.ρ, phys_params.tids, phys_params.force_scale, phys_params.n_data
    matlaw = phys_params.matlaw
    n_obs = div(n_data, length(tids))
    obs = zeros(Float64, n_data)
    
    for tid = 1:length(tids)
        _, data = Run_Homogenized(θ, θ_func, matlaw, ρ, tids[tid], force_scale)
        obs[(tid-1)*n_obs+1:tid*n_obs] = data[:]
    end
    
    return obs
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



function ExKI(phys_params::Params, 
    t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
    θ0_bar::Array{Float64,1}, θθ0_cov::Array{Float64,2}, 
    α_reg::Float64, N_iter::Int64 = 100)
    
    
    parameter_names = ["E"]
    
    ens_func(θ_ens) = Ensemble(phys_params, θ_ens)
    
    
    exkiobj = ExKIObj(parameter_names,
    θ0_bar, 
    θθ0_cov,
    t_mean, # observation
    t_cov,
    α_reg)
    
    update_cov = 1
    PLOT = true
    for i in 1:N_iter
        @info size(exkiobj.g_t),  phys_params.NT, 4
        update_ensemble!(exkiobj, ens_func) 
        @info "θ: ", exkiobj.θ_bar[end]
        @info "θθ_cov: ", exkiobj.θθ_cov[end]
        @info "norm(θ) :", norm(exkiobj.θ_bar[end]), "norm(θθ_cov) :", norm(exkiobj.θθ_cov[end]) 
        
        @info "F error of data_mismatch :", (exkiobj.g_bar[end] - exkiobj.g_t)'*(exkiobj.obs_cov\(exkiobj.g_bar[end] - exkiobj.g_t))
        
        
        if (update_cov > 0) && (i%update_cov == 0) 
            exkiobj.θθ_cov[1] = copy(exkiobj.θθ_cov[end])
        end


        if PLOT 

            T, NT, n_tids, matlaw = phys_params.T, phys_params.NT, phys_params.n_tids, phys_params.matlaw
         
            n_obs_time = NT
            n_obs_point = 2 # top left and right corners
            obs_ref = reshape(exkiobj.g_t,    n_obs_time ,  2n_obs_point, n_tids)
            obs = reshape(exkiobj.g_bar[end], n_obs_time ,  2n_obs_point, n_tids)
    
            fig_disp, ax_disp = PyPlot.subplots(nrows=2, ncols = 4,  sharex=true, sharey=true, figsize=(24,12))
            ts = LinRange(0, T, NT+1)
    

            for disp_id = 1:2
                # first tid first point
                ax_disp[disp_id,1].plot(ts[2:end], obs_ref[end-NT+1:end,disp_id, 1], "-o", label = "Ref")
                ax_disp[disp_id,1].plot(ts[2:end], obs[ end-NT+1:end,   disp_id, 1], label = matlaw)
                # first tid second point
                ax_disp[disp_id,2].plot(ts[2:end], obs_ref[end-NT+1:end,disp_id+2, 1], "-o", label = "Ref")
                ax_disp[disp_id,2].plot(ts[2:end], obs[end-NT+1:end,    disp_id+2, 1], label = matlaw)

                # second tid first point
                ax_disp[disp_id,3].plot(ts[2:end], obs_ref[end-NT+1:end,disp_id, 2], "-o", label = "Ref")
                ax_disp[disp_id,3].plot(ts[2:end], obs[ end-NT+1:end,   disp_id, 2], label = matlaw)
                 # second tid second point
                ax_disp[disp_id,4].plot(ts[2:end], obs_ref[end-NT+1:end,disp_id+2, 2], "-o", label = "Ref")
                ax_disp[disp_id,4].plot(ts[2:end], obs[end-NT+1:end,    disp_id+2, 2], label = matlaw)

            end
            
            fig_disp.tight_layout()
            fig_disp.savefig("Plate_disp_test"*string(i)*".png")
            close(fig_disp)
        end
        
    end
    
    return exkiobj
end

function Multiscale_Test(phys_params::Params, 
    t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
    θ0_bar::Array{Float64,1}, θθ0_cov::Array{Float64,2}, 
    α_reg::Float64, 
    N_iter::Int64;
    ki_file = nothing)
    
    if ki_file === nothing
        kiobj = ExKI(phys_params, t_mean, t_cov,  θ0_bar, θθ0_cov, α_reg, N_iter)
        @save "exkiobj.dat" kiobj
    else
        @load "exkiobj.dat" kiobj
    end
    
    # optimization related plots
    fig_ite, ax_ite = PyPlot.subplots(ncols = 2, nrows=1, sharex=false, sharey=false, figsize=(12,6))
    
    ites = Array(1:N_iter)
    errors = zeros(Float64, (3, N_iter))
    for i in ites
        errors[1, i] = NaN
        errors[2, i] = 0.5*(kiobj.g_bar[i] - kiobj.g_t)'*(kiobj.obs_cov\(kiobj.g_bar[i] - kiobj.g_t))
        errors[3, i] = norm(kiobj.θθ_cov[i])   
        
    end
    errors[3, 1] = norm(θθ0_cov) 
    ax_ite[1].semilogy(ites, errors[2, :], "-o", fillstyle="none", markevery=2 )
    ax_ite[2].semilogy(ites, errors[3, :], "-o", fillstyle="none", markevery=2 )
    
    ax_ite[1].set_xlabel("Iterations")
    ax_ite[1].set_ylabel("Optimization error")
    ax_ite[1].grid(true)
    ax_ite[2].set_xlabel("Iterations")
    ax_ite[2].set_ylabel("Frobenius norm")
    ax_ite[2].grid(true)
    
    fig_ite.tight_layout()
    fig_ite.savefig("Plate-error.png")
    close("all")
    
    # parameter plot
    N_θ = length(θ0_bar)
    θ_bar_arr = hcat(kiobj.θ_bar...)
    θθ_std = zeros(Float64, (N_θ, N_iter+1))
    for i = 1:N_iter+1
        θ_bar_arr[:,i] = phys_params.θ_func(θ_bar_arr[:,i])
        θθ_cov = phys_params.dθ_func(θ_bar_arr[:,i]) * kiobj.θθ_cov[i] * phys_params.dθ_func(θ_bar_arr[:,i])' 
        for j = 1:N_θ
            θθ_std[j, i] = sqrt(θθ_cov[j,j])
        end
    end

    

    errorbar(ites, θ_bar_arr[1,ites], yerr=3.0*θθ_std[1,ites], fmt="--o",fillstyle="none", label=L"E")
    errorbar(ites, θ_bar_arr[2,ites], yerr=3.0*θθ_std[2,ites], fmt="--o",fillstyle="none", label=L"ν") 
    errorbar(ites, θ_bar_arr[3,ites], yerr=3.0*θθ_std[3,ites], fmt="--o",fillstyle="none", label=L"σ_Y")
    errorbar(ites, θ_bar_arr[4,ites], yerr=3.0*θθ_std[4,ites], fmt="--o",fillstyle="none", label=L"K")
    semilogy()
    xlabel("Iterations")
    legend()
    grid("on")
    tight_layout()
    savefig("Plate_theta.png")
    close("all")
    
    return kiobj
end

# function driver_test()
#     tids = [100; 102]
#     porder = 2
#     fiber_size = 5
#     force_scale = 5.0
#     T = 200.0
#     NT = 200
#     n_obs_time = NT
#     n_obs_point = 2 # top left and right corners
#     phys_params = Params(tids, n_obs_point, n_obs_time, T, NT)
    
#     θ_func, matlaw, ρ, force_scale, n_tids, n_obs = phys_params.θ_func, phys_params.matlaw, phys_params.ρ, phys_params.force_scale, phys_params.n_tids, phys_params.n_obs
    
#     E, ν = 1e+6, 0.2
#     tid = tids[1]
#     obs = Dict()
#     @info "Run PlaneStressPlasticity"
#     matlaw = "PlaneStressPlasticity"
#     θ_scale = [1.0e+5 ,  0.02 ,  1.0e+3 , 1.0e+4]
#     θ = [E ;  ν ; 0.97e+4 ; 1e+5] ./ θ_scale
#     obs[matlaw] = Run_Homogenized(θ, θ_scale, matlaw, ρ, tid, force_scale)[2][:]
    
#     @info "Run PlanePlasticity"
#     matlaw = "PlanePlasticity"
#     θ_scale = [1.0e+5,  1.0e+5, 1.0e+5, 1.0e+5,  1.0e+3, 1.0e+4]
#     H11 = E/(1. -ν*ν)
#     H12 = ν*E/(1. -ν*ν)
#     H22 = H11
#     H33 = E/(2.0*(1.0+ν))

#     θ = [H11; H12; H22; H33; 0.97e+4; 1e+5]  ./ θ_scale
#     obs[matlaw] = Run_Homogenized(θ, θ_scale, matlaw, ρ, tid, force_scale)[2][:]
    
#     fig_disp, ax_disp = PyPlot.subplots(ncols = 2, nrows=1, sharex=false, sharey=false, figsize=(12,6))
#     ts = LinRange(0, T, NT+1)
    
#     for matlaw in ("PlaneStressPlasticity", "PlanePlasticity")
        
#         obs_mean = reshape(obs[matlaw], n_obs_time , 2n_obs_point)
        
#         ax_disp[1].plot(ts[2:end], obs_mean[end-NT+1:end,1], label = matlaw)
        
#         ax_disp[2].plot(ts[2:end], obs_mean[end-NT+1:end,2], label = matlaw)
        
#         ax_disp[1].set_xlabel("Time")
#         ax_disp[1].set_ylabel("x-Disp")
#         ax_disp[1].grid("on")
#         ax_disp[1].legend()
        
#         ax_disp[2].set_xlabel("Time")
#         ax_disp[2].set_ylabel("y-Disp")
#         ax_disp[2].grid("on")
#         ax_disp[2].legend()
        
#         fig_disp.tight_layout()
#         fig_disp.savefig("Plate_disp_test.png")
#         close(fig_disp)
#     end
# end

function prediction(phys_params, kiobj, θ_mean, θθ_cov, obs_noise_level, porder::Int64=2, tid::Int64=300, force_scale::Float64=0.5, fiber_size::Int64=5)
    # test on 300
    
    @load "Data/order$porder/obs$(tid)_$(force_scale)_$(fiber_size).jld2" obs
    obs_ref = copy(obs)
    
    
    θ_scale, matlaw, ρ, force_scale, n_tids, n_obs = phys_params.θ_func, phys_params.matlaw, phys_params.ρ, phys_params.force_scale, phys_params.n_tids, phys_params.n_obs
    
    # only visulize the first point
    NT, T = phys_params.NT, phys_params.T
    ts = LinRange(0, T, NT+1)
    
    # optimization related plots
    fig_disp, ax_disp = PyPlot.subplots(ncols = 2, nrows=1, sharex=false, sharey=false, figsize=(12,6))
    
    ax_disp[1].plot(ts[2:end], obs_ref[end-NT+1:end,1], "--o", color="grey", fillstyle="none", label="Reference")
    ax_disp[2].plot(ts[2:end], obs_ref[end-NT+1:end,2], "--o", color="grey", fillstyle="none", label="Reference")
    

    θθ_cov = (θθ_cov+θθ_cov')/2 
    θ_p = construct_sigma_ensemble(kiobj, θ_mean, θθ_cov)
    N_ens = kiobj.N_ens

    n_tids, n_obs_point, n_data = phys_params.n_tids, phys_params.n_obs_point, phys_params.n_data
    
    n_obs_time = div(n_data, 2n_obs_point * n_tids)

    obs = zeros(Float64, N_ens, n_obs_time * 2n_obs_point)

    Threads.@threads for i = 1:N_ens
        θ = θ_p[i, :]

        @info "θ is ", θ
        
        obs[i, :] = Run_Homogenized(θ, phys_params.θ_func, matlaw, ρ, tid, force_scale)[2][:]
    end

    obs_mean = obs[1, :]

    obs_cov  = construct_cov(kiobj,  obs, obs_mean)  + Array(Diagonal( obs_noise_level^2 * obs_mean.^2)) 
    obs_std = sqrt.(diag(obs_cov))

    obs_mean = reshape(obs_mean, n_obs_time , 2n_obs_point)
    obs_std = reshape(obs_std, n_obs_time , 2n_obs_point)

    ax_disp[1].plot(ts[2:end], obs_mean[end-NT+1:end,1])
    ax_disp[1].plot(ts[2:end], obs_mean[end-NT+1:end,1] + 3obs_std[end-NT+1:end,1])
    ax_disp[1].plot(ts[2:end], obs_mean[end-NT+1:end,1] - 3obs_std[end-NT+1:end,1])

    ax_disp[2].plot(ts[2:end], obs_mean[end-NT+1:end,2])
    ax_disp[2].plot(ts[2:end], obs_mean[end-NT+1:end,2]+ 3obs_std[end-NT+1:end,2])
    ax_disp[2].plot(ts[2:end], obs_mean[end-NT+1:end,2]- 3obs_std[end-NT+1:end,2])
    

    
    ax_disp[1].set_xlabel("Time")
    ax_disp[1].set_ylabel("x-Disp")
    ax_disp[1].grid("on")
    ax_disp[1].legend()
    
    
    
    ax_disp[2].set_xlabel("Time")
    ax_disp[2].set_ylabel("y-Disp")
    ax_disp[2].grid("on")
    ax_disp[2].legend()
    
    fig_disp.tight_layout()
    fig_disp.savefig("Plate_disp-$(tid).png")
    close(fig_disp)
end

# driver_test()

# error("stop")

tids = [200; 202]
porder = 2
fiber_size = 5
force_scale = 5.0
T = 200.0
NT = 200
n_obs_time = NT
n_obs_point = 2 # top left and right corners
obs_noise_level = 0.05
t_mean_noiseless = []
for i = 1:length(tids)
    @load "Data/order$porder/obs$(tids[i])_$(force_scale)_$(fiber_size).jld2" obs 
    obs = obs[end-NT+1:end, :][:]
    push!(t_mean_noiseless, obs)
end

t_mean_noiseless = vcat(t_mean_noiseless...)



# first choice of the observation covariance, only observation error 
t_cov = Array(Diagonal(obs_noise_level^2 * t_mean_noiseless.^2))

# add 5 percents observation noise
Random.seed!(123); 
t_mean = copy(t_mean_noiseless)
for i = 1:length(t_mean)
    noise = obs_noise_level*t_mean[i] * (rand(Uniform(0, 2))-1) # rand(Normal(0, 1))
    t_mean[i] += noise
end



α_reg = 1.0
N_iter = 20
matlaw = "PlaneStressPlasticity"

if matlaw == "PlaneStressPlasticity"

    phys_params = Params("PlaneStressPlasticity", tids, n_obs_point, n_obs_time, T, NT)
    N_θ = 4
    E, ν = 1e+6, 0.2
    θ0_bar =  phys_params.θ_func_inv([E; ν; 0.97e+4; 1e+5])
    θθ0_cov = Array(Diagonal(fill(1.0, N_θ))) 
    kiobj = Multiscale_Test(phys_params, t_mean, t_cov, θ0_bar, θθ0_cov, α_reg, N_iter)

else
    error("unrecognized matlaw: ", matlaw)
end


# @load "exkiobj.dat" kiobj
# update t_cov
data_misfit = (kiobj.g_bar[end] - t_mean)
n_dm = length(kiobj.g_bar[end] - t_mean)
@info "Mean error : ", sum(data_misfit)/n_dm, " Cov error : ", sum(data_misfit.^2)/n_dm
@info "Mean max : ", maximum(abs.(data_misfit)), " Cov max : ", maximum(data_misfit.^2)

 
# estimation of the constant model error, and re-train
t_cov = Array(Diagonal( obs_noise_level^2 * t_mean_noiseless.^2  .+  sum(data_misfit.^2)/n_dm))
kiobj = Multiscale_Test(phys_params, t_mean, t_cov, θ0_bar, θθ0_cov, α_reg, N_iter)


tid = 203
prediction(phys_params, kiobj, kiobj.θ_bar[end], kiobj.θθ_cov[end], obs_noise_level, porder, tid, force_scale, fiber_size)

tid = 300
prediction(phys_params, kiobj, kiobj.θ_bar[end], kiobj.θθ_cov[end], obs_noise_level, porder, tid, force_scale, fiber_size)
