using NNGCM
using LinearAlgebra
using Random


vor0_max, vor0_vmin = 4.208362320588924e-5, -1.7565000874341874e-5
"""
@info norm(spe_vor_c), norm(spe_vor_c[1:8, 1:8])
"""

function Barotropic_Main(nframes::Int64, init_type::String, init_data = nothing, spe_vor_b = nothing, obs_coord = nothing, plot_data::Bool=false)
  # the decay of a sinusoidal disturbance to a zonally symmetric flow 
  # that resembles that found in the upper troposphere in Northern winter.
  name = "Barotropic"
  num_fourier, nθ, nd = 170, 256, 1 #85, 128, 1  # 42, 64, 1
  num_spherical = num_fourier + 1
  nλ = 2nθ
  
  radius = 6371.2e3
  omega = 7.292e-5
  
  # Initialize mesh
  mesh = Spectral_Spherical_Mesh(num_fourier, num_spherical, nθ, nλ, nd, radius)
  θc, λc = mesh.θc,  mesh.λc
  cosθ, sinθ = mesh.cosθ, mesh.sinθ
  
  
  # Initialize atmo_data
  atmo_data = Atmo_Data(name, nλ, nθ, nd, false, false, false, false, sinθ, radius, omega)
  
  # Initialize integrator
  damping_order = 4
  damping_coef = 1.e-04
  robert_coef  = 0.04 
  implicit_coef = 0.0
  
  start_time = 0 
  
  day_to_second = 86400
  
  obs_time = Int64(day_to_second/nframes)
  end_time = day_to_second  #2 day
  Δt = 900
  init_step = true
  
  integrator = Filtered_Leapfrog(robert_coef, 
  damping_order, damping_coef, mesh.laplacian_eig,
  implicit_coef,
  Δt, init_step, start_time, end_time)
  
  # Initialize data
  dyn_data = Dyn_Data(name, num_fourier, num_spherical, nλ, nθ, nd, 0, 0)
  
  
  ######################## Initialization 
  # Initialize: 
  # dyn_data.spe_vor_c, dyn_data.spe_div_c = 0.0
  # dyn_data.grid_vor, dyn_data.grid_div = 0.0
  # dyn_data.grid_u_c, dyn_data.grid_v_c
  ########################
  
  grid_u, grid_v = dyn_data.grid_u_c, dyn_data.grid_v_c
  spe_vor_c, spe_div_c = dyn_data.spe_vor_c, dyn_data.spe_div_c
  grid_vor, grid_div = dyn_data.grid_vor, dyn_data.grid_div
  
  grid_vor_b = nothing
  
  if init_type == "truth"
    
    
    
    for i = 1:nλ
      grid_u[i, :, 1] .= 25 * cosθ - 30 * cosθ.^3 + 300 * sinθ.^2 .* cosθ.^6
    end
    grid_v .= 0.0 
    
    Lat_Lon_Pcolormesh(mesh, grid_u,  1; save_file_name = "Figs/Barotropic_u_backgroud.png", cmap = "jet")
    
    Vor_Div_From_Grid_UV!(mesh, grid_u, grid_v, spe_vor_c, spe_div_c) 
    Trans_Spherical_To_Grid!(mesh, spe_vor_c,  grid_vor)
    Trans_Spherical_To_Grid!(mesh, spe_div_c,  grid_div)
    
    Lat_Lon_Pcolormesh(mesh, grid_vor,  1; save_file_name = "Figs/Barotropic_vor_backgroud.png", cmap = "jet")
    
    grid_vor_b = copy(grid_vor); spe_vor_b = copy(spe_vor_c)
    
    grid_vor_pert = similar(grid_vor)
    # ! adding a perturbation to the vorticity
    m, θ0, θw, A = 4.0, 45.0 * pi / 180, 15.0 * pi / 180.0, 8.0e-5
    for i = 1:nλ
      for j = 1:nθ
        grid_vor_pert[i,j, 1] = A / 2.0 * cosθ[j] * exp(-((θc[j] - θ0) / θw)^2) * cos(m * λc[i])
      end
    end
    
    
    ################# Observations are grid points
    # Random sample
    
    # Random sample in the band
    # band = Int64[]
    # for j = 1:nθ
    #   if θc[j] < θ0 + θw && θc[j]  > θ0 - 2θw;    push!(band, j);    end
    
    # end
    
    
    
    nobs = 60
    obs_coord = zeros(Int64, nobs, 2)
    Random.seed!(42)
    obs_coord[:,1], obs_coord[:, 2] = rand(1:nλ-1, nobs), rand(Int64(nθ/2)+1:nθ-1, nobs)
    # Lat_Lon_Pcolormesh(mesh, grid_u, 1, obs_coord,  "obs.png")
    # error("stop")
    # # X,Y = repeat(θc_deg, 1, nd), repeat(σc, 1, nθ)'
    # obs_x = trunc.(Int64, LinRange(1, nλ, nobs_x+1))[1:nobs_x]
    # # obs_y = trunc.(Int64, LinRange(band[1], band[end], nobs_y))
    # obs_y = trunc.(Int64, LinRange(Int64(nθ/2), nθ, nobs_y))
    # for i = 1:nobs_x
    #   for j = 1:nobs_y
    #       obs_coord[i + (j-1)*nobs_x, 1], obs_coord[i + (j-1)*nobs_x, 2] = obs_x[i], obs_y[j]
    #   end
    # end
    ##################################
    
    
    Lat_Lon_Pcolormesh(mesh, grid_vor_pert, 1; save_file_name = "Figs/Barotropic_vor_pert0.png", cmap = "jet")
    
    grid_vor .+= grid_vor_pert
    Lat_Lon_Pcolormesh(mesh, grid_vor, 1; save_file_name = "Figs/Barotropic_vor0.png", cmap = "jet")
    
    
    Trans_Grid_To_Spherical!(mesh, grid_vor, spe_vor_c)
    UV_Grid_From_Vor_Div!(mesh, spe_vor_c,  spe_div_c, grid_u, grid_v)
    
    Lat_Lon_Pcolormesh(mesh, grid_u, 1; save_file_name = "Figs/Barotropic_vel_u0.png", cmap = "jet")
    
    @info norm(spe_vor_c), norm(spe_vor_c[1:8, 1:8])
    
    
  else
    
    Barotropic_ω0!(mesh, init_type, init_data, spe_vor_c, grid_vor, spe_vor_b = spe_vor_b)
    
    spe_div_c .= 0.0
    grid_div  .= 0.0
    UV_Grid_From_Vor_Div!(mesh, spe_vor_c,  spe_div_c, grid_u, grid_v)
  end
  
  spe_vor0 = copy(spe_vor_c); grid_vor0 = copy(grid_vor)
  
  ###########################################################################################
  
  obs_data = Array{Float64, 3}[]
  
  
  NT = Int64(end_time / Δt)
  time = start_time
  Barotropic_Dynamics!(mesh, atmo_data, dyn_data, integrator)
  Update_Init_Step!(integrator)
  time += Δt 
  for i = 2:NT
    Barotropic_Dynamics!(mesh, atmo_data, dyn_data, integrator)
    time += Δt
    if time % obs_time == 0
      # @info "day = ", div(time , day_to_second)
      # TODO can change to other quantities
      if init_type == "truth" || plot_data 
        @info "plot :",  "Figs/Barotropic_u-"*string(div(time , obs_time))*".png", " time = ", time
        @info "norm(grid_u): ", norm(grid_u)
        Lat_Lon_Pcolormesh(mesh, grid_u, 1, obs_coord; save_file_name =   "Figs/Barotropic_u-"*string(div(time , obs_time))*".png", cmap = "jet")
      end
      push!(obs_data, grid_u)
    end
    
  end
  if init_type == "truth" || plot_data 
    Lat_Lon_Pcolormesh(mesh, grid_u,  1; save_file_name =  "Figs/Barotropic_vel_u.png", cmap = "jet")
    Lat_Lon_Pcolormesh(mesh, grid_vor, 1; save_file_name =  "Figs/Barotropic_vor.png", cmap = "jet")
  end
  
  return mesh,  grid_vor_b, spe_vor_b, grid_vor0, spe_vor0, obs_coord, obs_data
end


function Barotropic_ω0!(mesh, init_type::String, init_data, spe_vor0, grid_vor0; spe_vor_b = nothing)
  radius = 6371.2e3
  
  if init_type == "grid_vor"
    grid_vor0[:] = copy(init_data)
    
    # grid_vor0 ./= radius
    
    Trans_Grid_To_Spherical!(mesh, grid_vor0, spe_vor0)
    
  elseif init_type == "spec_vor"
    # 
    spe_vor0 .= spe_vor_b
    # m = 0,   1, ... N
    # n = m, m+1, ... N
    # F_m,n = {F_0,1 F_0,2, F_0,3 ... F_0,N,    F_1,1 F_1,2, ... F_1,N, ..., F_N,N}
    # F_0,1 F_1,1  
    # F_0,2 F_1,2 F_2,2
    #   ...
    # F_0,N F_1,N F_2,N, ..., F_N,N
    # N + 2(N+1)N/2 = N^2 + 2N
    
    n_init_data = length(init_data)
    N = Int64(sqrt(1 + n_init_data) - 1)
    
    m = 0
    for n = 1:N
      spe_vor0[m+1,n+1,1] += init_data[n]/radius
    end
    i_init_data = N
    for m = 1:N
      for n = m:N
        # m = 1, 2, 3 .. m 
        #     0  N  N-1...
        #     N + N-1 ... N-m+2 
        spe_vor0[m+1,n+1,1] += (init_data[i_init_data+1] - init_data[i_init_data + 2]*im)/radius
        i_init_data += 2
      end 
    end
    @assert(i_init_data == n_init_data)
    
    Trans_Spherical_To_Grid!(mesh, spe_vor0, grid_vor0)
  else 
    error("Initial condition type :", init_type,  " is not recognized.")
  end
  
end

###########################################################################################
