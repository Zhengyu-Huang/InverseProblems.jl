using Distributions
using ArgParse



include(joinpath(@__DIR__, "helper_funcs.jl"))




obs_mean, obs_noise_cov = read_observation(augment)

ukiobj = UKIObj(para_names, init_mean, init_cov, prior_mean, prior_cov, obs_mean, obs_noise_cov, Δt, α_reg, update_freq)


save_params(ukiobj, 0)


u_p_ens_new, _ = prediction_ensemble(ukiobj)


# transform parameters  

constraint_u_p_ens_new = constraint(u_p_ens_new)

# update input files

write_solver_input_files(constraint_u_p_ens_new)
