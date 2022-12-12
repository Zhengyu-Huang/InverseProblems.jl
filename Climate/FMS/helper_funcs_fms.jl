using NCDatasets
using Statistics
using Interpolations
using LinearAlgebra
using JLD
using PyPlot


"""
Transfer function 
"""



params_prefix =  "output/output_params_"
input_prefix  =  "/groups/esm/dzhuang/fms_GCMForcing/exp/held_suarez/run_"
output_prefix =  "/central/scratch/dzhuang/fms_"
input_file_template = "input_fms_template"




prior_mean = [0.0; 0.0;]
prior_cov  = Array(Diagonal([10.0^2; 10.0^2]))
param_bounds = [[1.0, nothing], [1.0, nothing]]

init_mean = [0.0; 0.0;]
init_cov  = Array(Diagonal([1.0^2; 1.0^2]))

#######################################################################################

N_params = length(prior_mean)
para_names = ["unknowns"]
get_zonal(ds, varname) = dropdims(mean(ds[varname][:], dims=1), dims=1)


function read_observation(augment::Bool)
	 
    data = NCDataset("/central/scratch/dzhuang/fms/fms_output/held_suarez/control_calibration/history/day1000h00.nc", "r");
    
    obs_mean = get_zonal(data, "temp")'[:]
    N_y = length(obs_mean)
    obs_noise_cov = Array(Diagonal(fill(5.0^2, N_y))) 
    

    if augment
       return [obs_mean; prior_mean], [obs_noise_cov zeros(Float64, N_y, N_params); zeros(Float64, N_params, N_y) prior_cov]
    else
       return obs_mean, obs_noise_cov
    end

end



obs_mean, _ = read_observation(augment)
N_y = length(obs_mean)






function write_solver_input_files(constraint_u_p_ens)

    N_ens, N_params = size(constraint_u_p_ens)

    input_lines = readlines(input_file_template)
    
    modified_lines = [11; 15; 35; 286; 287]
    for i = 1:N_ens
    	constraint_u_p = constraint_u_p_ens[i, :]
        input_folder = input_prefix*string(i)
        output_folder = output_prefix*string(i)

	# TODO set run_name and input file name
        run_name = "control_"*string(i)
 	input_name = input_folder*"/run_"*run_name
        
 
    	input_io = open(input_name, "w")
	 	
    	for (n, line) in enumerate(input_lines)
       	   
	   if n ∉ modified_lines

              write(input_io, line*"\n")

           else
              if n == 11
                  write(input_io, "cd  $(input_folder)\n")
	      elseif n == 15
	      	  write(input_io, "set run_name = $(run_name)\n")
	      elseif n == 35
                  write(input_io, "set data_dir_base = $(output_folder)\n")

	      elseif n == 286
	      	  write(input_io, "ka      =-$(constraint_u_p[1]),\n")    
	      elseif n == 287
                  write(input_io, "ks      =-$(constraint_u_p[2])/\n")
	      else
		  @error("STOP: line $(n) should not be modified!")
	      end              
	   
	   end

       end

       close(input_io)

    end
end




function read_solver_output(iteration_::Int64,  ens_index::Int64, augment::Bool)
    data = NCDataset("/central/scratch/dzhuang/fms_$(string(ens_index))/fms_output/held_suarez/control_$(ens_index)/history/day0400h00.nc", "r");	
       
    _, _, _, u_p_ens = read_params( iteration_)
    
    y = get_zonal(data, "temp")'[:]

    
    if augment
       return [y; u_p_ens[ens_index, :]]
    else 
       return y
    end

end



function plot(N_iterations::Int64)
	
    u_name = ["ka", "ks"]
    
    obs_mean, obs_noise_cov = read_observation(false)

    u_refs = [40.0; 4.0]
 
    constraint_u_mean_all, constraint_uu_cov_all = read_all_params(N_iterations)
    constraint_u_mean_all_arr, constraint_uu_cov_all_arr = zeros(N_params, N_iterations+1), zeros(N_params, N_iterations+1)

    for iteration_ = 0: N_iterations

    	constraint_u_mean_all_arr[:, iteration_+1] = constraint_u_mean_all[ iteration_+1, :]
	constraint_uu_cov_all_arr[:, iteration_+1] = diag(constraint_uu_cov_all[ iteration_+1, :,:])
    end



    ites = Array(0:N_iterations)

    figure(figsize = (7.5, 4.8))
    for i = 1: N_params
        errorbar(ites, constraint_u_mean_all_arr[i,:], yerr=3.0*constraint_uu_cov_all_arr[i,:], fmt="--o",fillstyle="none", label=L"$(u_names[i])")
        semilogy(ites, fill(u_refs[i], N_iterations+1), "--", color="gray")

    end
    xlabel("Iterations")
    legend(bbox_to_anchor=(0.95, 0.8))
    grid("on")
    tight_layout()
    savefig("uki_parameters.pdf")
    close("all")

end