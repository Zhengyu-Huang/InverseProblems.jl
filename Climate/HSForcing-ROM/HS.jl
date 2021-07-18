using NNGCM


function Atmos_Spectral_Dynamics_Main(physcis_params::Dict{String, Float64}, end_day::Int64 = 1200, spinup_day::Int64 = 200, mesh_size::String = "T42")
  # the decay of a sinusoidal disturbance to a zonally symmetric flow 
  # that resembles that found in the upper troposphere in Northern winter.
  name = "Spectral_Dynamics"

  if mesh_size == "T42"
    num_fourier, nθ, nd = 42, 64, 20
  elseif mesh_size == "T21"
    num_fourier, nθ, nd = 21, 32, 10
  else
    error("mesh size : " ,  mesh_size, " is not recognized")
  end

  num_spherical = num_fourier + 1
  nλ = 2nθ
  
  radius = 6371000.0
  omega = 7.292e-5
  sea_level_ps_ref = 1.0e5
  init_t = 264.0

  
  # Initialize mesh
  mesh = Spectral_Spherical_Mesh(num_fourier, num_spherical, nθ, nλ, nd, radius)
  θc, λc = mesh.θc,  mesh.λc
  cosθ, sinθ = mesh.cosθ, mesh.sinθ
  
  vert_coord = Vert_Coordinate(nλ, nθ, nd, "even_sigma", "simmons_and_burridge", "second_centered_wts", sea_level_ps_ref)
  # Initialize atmo_data
  do_mass_correction = true
  do_energy_correction = true
  do_water_correction = false
  
  use_virtual_temperature = false
  atmo_data = Atmo_Data(name, nλ, nθ, nd, do_mass_correction, do_energy_correction, do_water_correction, use_virtual_temperature, sinθ, radius,  omega)
  
  # Initialize integrator
  damping_order = 4
  damping_coef = 1.15741e-4
  robert_coef  = 0.04 
  
  implicit_coef = 0.5
  day_to_sec = 86400
  start_time = 0
  end_time = end_day*day_to_sec  #
  Δt = 1200
  init_step = true
  
  integrator = Filtered_Leapfrog(robert_coef, 
  damping_order, damping_coef, mesh.laplacian_eig,
  implicit_coef, Δt, init_step, start_time, end_time)
  
  ps_ref = sea_level_ps_ref
  t_ref = fill(300.0, nd)
  wave_numbers = mesh.wave_numbers
  semi_implicit = Semi_Implicit_Solver(vert_coord, atmo_data,
  integrator, ps_ref, t_ref, wave_numbers)


  # Data Visualization
  op_man= Output_Manager(mesh, vert_coord, start_time, end_time, spinup_day)
    
  
  # Initialize data
  dyn_data = Dyn_Data(name, num_fourier, num_spherical, nλ, nθ, nd)
  
  
  NT = Int64(end_time / Δt)
  
  Get_Topography!(dyn_data.grid_geopots)
  
  Spectral_Initialize_Fields!(mesh, atmo_data, vert_coord, sea_level_ps_ref, init_t,
  dyn_data.grid_geopots, dyn_data)
  
  
  Atmosphere_Update!(mesh, atmo_data, vert_coord, semi_implicit, dyn_data, physcis_params)
  Update_Init_Step!(semi_implicit)
  integrator.time += Δt 
  Update_Output!(op_man, dyn_data, integrator.time)

  for i = 2:NT

    Atmosphere_Update!(mesh, atmo_data, vert_coord, semi_implicit, dyn_data, physcis_params)
  
    integrator.time += Δt
    #@info integrator.time

    Update_Output!(op_man, dyn_data, integrator.time)

    if (integrator.time%(100*day_to_sec) == 0)
      @info "Day: ", div(integrator.time,day_to_sec), " Max |U|,|V|,|P|,|T| : ", maximum(abs.(dyn_data.grid_u_c)), maximum(abs.(dyn_data.grid_v_c)), maximum(dyn_data.grid_p_full), maximum(dyn_data.grid_t_c)
    end


  end

  return op_man
  
end

