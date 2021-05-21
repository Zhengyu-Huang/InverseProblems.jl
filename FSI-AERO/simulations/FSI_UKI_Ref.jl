@info "start FSI UKI Ref"
using JLD2
include("../../Inversion/Plot.jl")
include("../calibration/IO.jl")



# N_θ = 5 case with 5% Gaussian error
N_θ = 10

# generate the inpute file
damage_ω, damage_θ = generate_mat_file("./agard.fem.composite", nothing, N_θ)


# Plot damage field test
xx = Array(LinRange(1.5, 28.5, 10))
PyPlot.figure(figsize=(6,4))
PyPlot.plot(xx, damage_ω)
PyPlot.title("Damage_field")
PyPlot.savefig("Damage.png")

@info "start AEROF AEROS"

# run aero-f aero-s
run(`./run.unsteady`)


@info "finish AEROF AEROS"
# Plot obs_data
obs_data, obs_ts = read_obs("results.1/AGARD.disp.")
PyPlot.figure(figsize=(6,4))
for i = 1:size(obs_data)[2]
    PyPlot.plot(obs_ts, obs_data[:, i], "-*")
end
PyPlot.title("Displacements")
PyPlot.savefig("Displacement.png")