using NNGCM
using JLD2
using Statistics
using PyPlot

@load "HS_OpM.dat" output_manager


n_day = output_manager.n_day
spinup_day = output_manager.spinup_day
nθ, nd = output_manager.nθ, output_manager.nd


t_daily_zonal_mean, u_daily_zonal_mean, v_daily_zonal_mean = output_manager.t_daily_zonal_mean, output_manager.u_daily_zonal_mean, output_manager.v_daily_zonal_mean
   

# Currently, I have 1200 day data, spinup first 200 days
n_data = nθ*nd
obs = reshape(t_daily_zonal_mean, (n_data, n_day))
n_obs = 200
n_obs_box = Int64((n_day - spinup_day)/n_obs)
obs_box = zeros(Float64, n_data, n_obs_box)

for i = 1:n_obs_box
    n_obs_start = spinup_day + (i - 1)*n_obs + 1
    obs_box[:, i] = mean(obs[:, n_obs_start : n_obs_start + n_obs - 1], dims = 2)
end


obs_mean = dropdims(mean(obs_box, dims=2), dims = 2)
obs_cov = zeros(Float64, n_data, n_data)
for i = 1:n_obs_box
    global obs_cov
    obs_cov += (obs_box[:,i] - obs_mean) *(obs_box[:,i] - obs_mean)'
end
obs_cov ./= (n_obs_box - 1)
