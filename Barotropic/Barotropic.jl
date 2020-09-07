using NNGCM
using LinearAlgebra

"""
@info norm(spe_vor_c), norm(spe_vor_c[1:8, 1:8])
"""

function Barotropic_Main(init_type::String, init_data = nothing)
  # the decay of a sinusoidal disturbance to a zonally symmetric flow 
  # that resembles that found in the upper troposphere in Northern winter.
  name = "Barotropic"
  num_fourier, nθ, nd = 85, 128, 1
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
  end_time = 2*day_to_second  #2 day
  Δt = 1800
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
  
  if init_type == "truth"
    
    
    for i = 1:nλ
      grid_u[i, :, 1] .= 25 * cosθ - 30 * cosθ.^3 + 300 * sinθ.^2 .* cosθ.^6
    end
    grid_v .= 0.0 
    
    Lat_Lon_Pcolormesh(mesh, grid_u,  1, "Barotropic_vel_u_pert0.png")
    
    Vor_Div_From_Grid_UV!(mesh, grid_u, grid_v, spe_vor_c, spe_div_c) 
    Trans_Spherical_To_Grid!(mesh, spe_vor_c,  grid_vor)
    Trans_Spherical_To_Grid!(mesh, spe_div_c,  grid_div)
    
    grid_vor_pert = similar(grid_vor)
    # ! adding a perturbation to the vorticity
    m, θ0, θw, A = 4.0, 45.0 * pi / 180, 15.0 * pi / 180.0, 8.0e-5
    for i = 1:nλ
      for j = 1:nθ
        grid_vor_pert[i,j, 1] = A / 2.0 * cosθ[j] * exp(-((θc[j] - θ0) / θw)^2) * cos(m * λc[i])
      end
    end
    
    Lat_Lon_Pcolormesh(mesh, grid_vor_pert, 1, "Barotropic_vor_pert0.png")
    
    grid_vor .+= grid_vor_pert
    Lat_Lon_Pcolormesh(mesh, grid_vor, 1, "Barotropic_vor0.png")
    
    
    Trans_Grid_To_Spherical!(mesh, grid_vor, spe_vor_c)
    UV_Grid_From_Vor_Div!(mesh, spe_vor_c,  spe_div_c, grid_u, grid_v)

    Lat_Lon_Pcolormesh(mesh, grid_u, 1, "Barotropic_vel_u0.png")
    
  else
    
    Barotropic_ω0!(mesh, init_type, init_data, spe_vor_c, grid_vor)
      
    spe_div_c .= 0.0
    grid_div  .= 0.0
    UV_Grid_From_Vor_Div!(mesh, spe_vor_c,  spe_div_c, grid_u, grid_v)
  end
  
  ###########################################################################################
  
  obs = Array{Float64, 3}[]
  
  
  NT = Int64(end_time / Δt)
  time = start_time
  Barotropic_Dynamics!(mesh, atmo_data, dyn_data, integrator)
  Update_Init_Step!(integrator)
  time += Δt 
  for i = 2:NT
    Barotropic_Dynamics!(mesh, atmo_data, dyn_data, integrator)
    time += Δt
    if time % day_to_second == 0
      @info "day = ", div(time , day_to_second)
      push!(obs, grid_vor)
    end
    
  end
  
  Lat_Lon_Pcolormesh(mesh, grid_u,  1, "Barotropic_vel_u.png")
  Lat_Lon_Pcolormesh(mesh, grid_vor, 1, "Barotropic_vor.png")
  
  return mesh, dyn_data, obs
end


function Barotropic_ω0!(mesh, init_type::String, init_data, spe_vor0, grid_vor0)
  if init_type == "grid_vor"
    grid_vor0[:] = copy(init_data)
    
    Trans_Grid_To_Spherical!(mesh, grid_vor0, spe_vor0)
    
  elseif init_type == "spec_vor"
    # 
    spe_vor0 .= 0.0
    # m = 0,   1, ... N
    # n = m, m+1, ... N
    # F_m,n = {F_0,1 F_1,1   F_0,2 F_1,2 F_2,2, ..., F_N,N}
    # F_0,1 F_1,1  
    # F_0,2 F_1,2 F_2,2
    #   ...
    # F_0,N F_1,N F_2,N, ..., F_N,N
    # 
    n_init_data = length(init_data)
    N = Int64((sqrt(9 + 4*n_init_data) - 3)/2)
    
    for n = 1:N
      for m = 0:n
        i_init_data = Int64((n+2)*(n-1)/2) + m + 1
        # @info m+1,n+1, i_init_data
        spe_vor0[m+1,n+1] = (init_data[2*i_init_data-1] + init_data[2*i_init_data] * im)/radius
      end
    end
    
    Trans_Spherical_To_Grid!(mesh, spe_vor0, grid_vor0)
  else 
    error("Initial condition type :", init_type,  " is not recognized.")
  end

end

###########################################################################################
