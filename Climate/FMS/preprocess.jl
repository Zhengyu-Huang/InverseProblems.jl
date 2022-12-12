using Distributions
using ArgParse
include(joinpath(@__DIR__, "helper_funcs.jl"))



output_folders = output_prefix*"*"
run(`rm -rf $(output_folders)`)

run(`mkdir -p output`)
run(`mkdir -p output/slurm`)


for i = 1:N_ens
    run(`mkdir -p $(input_prefix)$(i)`)
    run(`mkdir -p $(output_prefix)$(i)`)
end

