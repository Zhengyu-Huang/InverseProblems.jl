include("Piston.jl")
include("Euler1D.jl")
# L is the length of the tube, assuming that the tube is at [0, L]
# N is the number of fluid ceils
# structure_inf = [xs_0, vs_0, ks, ms], which are parameters for structure
# fluid_inf =[gamma,ρ_l, v_l, p_l], which are parameters for the fluid
# mesh  = [L, N ,CFL]

function Solve(L::FT,  N::IT, fluid_info::Array{FT,1}, structure_info::Array{Any,1}, time_info::Array{FT,1}; output_freq::IT = Int64(time_info[2]/time_info[1])) where {FT<:AbstractFloat, IT<:Int}
    # constant flow velocity
    γ, ρ_0, v_0, p_0 = fluid_info
    Δt, T = time_info
    N_T = Int64(T/Δt)

    # create output array
    fluid_history, structure_history = zeros(Float64, 3, N, div(N_T, output_freq) + 1), zeros(Float64, 3, div(N_T, output_freq) + 1)
    
    # assume the structure spring is attached at L/2
    

    t = 0.0
    #####################################################
    # Initialize structure
    #####################################################

    ms, cs, ks, ds_0, vs_0, x0, motion, forced_motion_func = structure_info
    
    as_0 = (p_0 - ks*ds_0 - cs*vs_0)/ms
    Q = [ds_0; vs_0; as_0]

    structure = Structure(ds_0, vs_0, as_0, x0, ms, cs, ks, motion, forced_motion_func, t)
    

    #####################################################
    # Initialize fluid
    #####################################################
    emb = [x0 + ds_0, vs_0]
    fluid = Euler1D(L, N, emb; FIVER_order = 2, γ = γ)
    init_func = (x) -> (x <= x0 ? (true, ρ_0, v_0, p_0) : (false, 0, 0, 0))
    Initial!(fluid, init_func)


    # Save the initial condition
    fluid_history[:, :, 1]  .= fluid.V
    structure_history[:, 1] .= ds_0, vs_0, as_0


    ################################################################
    if structure.motion == "AEROELASTIC"
        fext = p_0
        A6_First_Half_Step!(structure, fext, t, Δt)
    end

    for i_t = 1:N_T

        if i_t % output_freq == 0
            # save the structure info at i_t Δt
            structure_history[:, div(i_t , output_freq) + 1] = structure.Q
        end

        # the predicted structure position and velocity at t + Δt
        emb = Send_Embeded_Disp(structure, t, Δt)
        
        # update fluid to t + Δt
        Fluid_Time_Advance!(fluid, emb, t, Δt)


        if structure.motion == "AEROELASTIC"
            # the corrected fluid load at t + Δt/2
            fext = Send_Force(fluid, t, Δt)

            # update structure to t + Δt
            Structure_Time_Advance!(structure, fext, t, Δt)
        end

        if i_t % output_freq == 0
            # save the fluid data at i_t Δt
            fluid_history[:, :, div(i_t,output_freq)+1] .= fluid.V
        end


        t = t + Δt

    end

    return fluid, structure, fluid_history, structure_history
end


###################################################################
# Inverse problem
###################################################################

mutable struct Setup_Param{FT<:AbstractFloat, IT<:Int}
    # mesh information
    
    L::FT              # computational domain [0, L]
    N::IT            # number of grid points (including both ends)
    Δx::FT
    xx::Array{FT, 1}   # uniform grid [0, Δx, 2Δx ... L]
    
    # time information
    time_info::Array{FT,1}

    # fluid information
    fluid_info::Array{FT,1}

        
    # truth structure information ms, cs, ks
    θ_ref::Array{FT,1}
 
    
    # inverse parameters
    θ_names::Array{String, 1}
    N_θ::IT

    
    # observation locations and number of observations
    obs_freq::IT
    N_y::IT
    
end


function Setup_Param(L::FT, N::IT, 
    time_info::Array{FT,1}, fluid_info::Array{FT,1}, 
    θ_ref::Array{FT,1}, θ_names::Array{String, 1},
    obs_freq::IT) where {FT<:AbstractFloat, IT<:Int}
    
    Δx = L/N
    xx = Array(LinRange(Δx/2, L - Δx/2, N))
    N_θ = length(θ_names)
    
    N_T = Int64(T/Δt)
    N_y = div(N_T, obs_freq) + 1
    
    return Setup_Param(L,  N, Δx, xx, time_info, fluid_info, θ_ref, θ_names, N_θ, obs_freq, N_y)
end




function forward(s_param::Setup_Param, θ::Array{FT, 1}) where {FT<:AbstractFloat}
    L, N = s_param.L, s_param.N

    fluid_info, time_info = s_param.fluid_info, s_param.time_info
    
    ms = s_param.θ_ref[1]
    

    cs, ks = θ

    structure_info = [ms, cs, ks, 0.0, 0.0, L/2, "AEROELASTIC", nothing] 
    
    obs_freq = s_param.obs_freq

    _, _, _, structure_history = Solve(L, N, fluid_info, structure_info, time_info, output_freq = obs_freq)


    # structure displacement
    y = structure_history[1, :]

    return y
end

# using PyPlot
# L, N = 2.0, 200
# mesh_info  = [L, N]
# fluid_info = [1.4, 1.225, 0. , 1.0] 
# ms, cs, ks = 1.0, 0.0, 1.0
# structure_info = [ms, cs, ks, 0.0, 0.0, L/2, "AEROELASTIC", nothing] 

# # function structure(t)
    
# #     ds = 1.0/8.0 * t^4
# #     vs = 1.0/2.0 * t^3
# #     as = 3.0/2.0 * t^2

# #     return [ds; vs; as]
# # end

# # structure_info = [ms, cs, ks, 0.0, 0.0, L/2, "FORCED", structure] 
# Δt, T = 0.001, 0.5
# time_info = [Δt, T]

# fluid, piston = Solve(mesh_info, fluid_info, structure_info, time_info)
# V = fluid.V
# xx = fluid.xx

# fig, ax = PyPlot.subplots(ncols=3, nrows=1, figsize=(18, 5))

# ax[1].plot(xx, V[1, :], "-o", fillstyle = "none")
# ax[2].plot(xx, V[2, :], "-o", fillstyle = "none")
# ax[3].plot(xx, V[3, :], "-o", fillstyle = "none")
# ax[1].set_xlabel("x")
# ax[1].set_ylabel("ρ")
# ax[2].set_xlabel("x")
# ax[2].set_ylabel("v")
# ax[3].set_xlabel("x")
# ax[3].set_ylabel("p")
# fig.tight_layout()


include("../Inversion/Plot.jl")
include("../Inversion/KalmanInversion.jl")
N, L = 400, 2.0

# flow gas constant, intial density, velocity and pressure
γ, ρ0, v0, p0 = 1.4, 1.225, 0. , 1.0
fluid_info = [γ, ρ0, v0, p0] 

# piston mass, damping coefficient and spring stiffness
ms, cs, ks = 1.0, 0.50, 2.0
θ_ref = [ms; cs; ks]
# initial displacement, velocity, and initial position
u0, v0, x0 = 0.0, 0.0, L/2
structure_info = [ms, cs, ks, u0, v0, x0, "AEROELASTIC", nothing] 

# time step and end time
Δt, T = 0.001, 1.0
N_T = Int64(T/Δt)
obs_freq = 10
time_info = [Δt, T]

s_param = Setup_Param(L, N,  time_info, fluid_info,  θ_ref, ["cs", "ks"], obs_freq)

# y_noiseless  = forward(s_param, s_param.θ_ref) 

fluid, piston, _, piston_history = Solve(L, N, fluid_info, structure_info, time_info; output_freq = 1)
y_noiseless = piston_history[1, 1:obs_freq:end]

y = copy(y_noiseless)
noise_level = 0.05
N_y = length(y)
Random.seed!(123);
for i = 1:N_y
    # noise = rand(Normal(0, noise_level*y[i]))
    noise = rand(Normal(0, 0.01))
    y[i] += noise
end


# visualize the observation
PyPlot.figure()
tt = Array(LinRange(0, T, N_T+1))
PyPlot.plot(tt, piston_history[1, :])
PyPlot.plot(tt[1:obs_freq:end], y, "o", color="black")
PyPlot.xlabel("Time")
PyPlot.ylabel("Displacement")


N_y, N_θ = s_param.N_y, s_param.N_θ
# observation
Σ_η = Array(Diagonal(fill(0.01^2, N_y)))


# UKI 
θ0_mean =  ones(Float64, N_θ) # [0.5; 2.0] # 
θθ0_cov = Array(Diagonal(fill(1.0^2.0, N_θ)))
N_iter = 10
α_reg = 1.0
update_freq = 1
ukiobj = UKI_Run(s_param, forward, θ0_mean, θθ0_cov, y, Σ_η, α_reg, update_freq, N_iter)



####
θ_mean_arr = hcat(ukiobj.θ_mean...)
θθ_std_arr = zeros(Float64, (N_θ, N_iter+1))
for i = 1:N_iter+1
    for j = 1:N_θ
        θθ_std_arr[j, i] = sqrt(ukiobj.θθ_cov[i][j,j])
    end
end

ites = Array(LinRange(1, N_iter+1, N_iter+1))
PyPlot.errorbar(ites, θ_mean_arr[1,:], fmt="--o",fillstyle="none", yerr=θθ_std_arr[1,:],  label=L"c_s")
PyPlot.plot(ites, fill(cs, N_iter+1), "--", color="grey")

PyPlot.errorbar(ites, θ_mean_arr[2,:], fmt="--o", fillstyle="none", yerr=θθ_std_arr[2,:], label=L"k_s")
PyPlot.plot(ites, fill(ks, N_iter+1), "--", color="grey")
PyPlot.legend()




# fig, (ax1, ax2, ax3) = PyPlot.subplots(ncols=3, figsize=(18,6))
# ites = Array(LinRange(1, N_iter, N_iter))
# errors = zeros(Float64, (3, N_iter))
# for i = 1:N_iter
#     errors[1, i] = norm(s_param.θ_ref - ukiobj.θ_mean[i])/norm(s_param.θ_ref)
#     errors[2, i] = 0.5*(ukiobj.y_pred[i] - ukiobj.y)'*(ukiobj.Σ_η\(ukiobj.y_pred[i] - ukiobj.y))
#     errors[3, i] = norm(ukiobj.θθ_cov[i])
# end

# ax1.semilogy(ites, errors[1, :], "-o", fillstyle="none", markevery=2, label= "UKI")
# ax1.set_xlabel("Iterations")
# ax1.set_ylabel("Relative L₂ norm error of logκ")
# ax1.legend()
# ax1.grid(true)

# ax2.semilogy(ites, errors[2, :], "-o", fillstyle="none", markevery=2, label= "UKI")
# ax2.set_xlabel("Iterations")
# ax2.set_ylabel("Optimization error")
# ax2.grid(true)
# ax2.legend()

# ax3.semilogy(ites, errors[3, :], "-o", fillstyle="none", markevery=2, label= "UKI")
# ax3.set_xlabel("Iterations")
# ax3.set_ylabel("Frobenius norm of covariance")
# ax3.grid(true)
# ax3.legend()
# fig.tight_layout()


