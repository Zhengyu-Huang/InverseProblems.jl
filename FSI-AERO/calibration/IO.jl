using Random
using Distributions
using PyPlot

function read_obs(prefix::String = "results.1/AGARD.disp."; NT = 2000, Δobs = 20)
    ΔT = 5e-5

    node_ids = [111, 116, 121, 56, 61, 66, 352, 362, 377, 197, 207, 222]
    N_node = length(node_ids)

    N_obs_per_node = div(NT, Δobs)
    obs_ts         = Array(Δobs:Δobs:NT)
    obs_data       = zeros(Float64, N_obs_per_node, length(node_ids))

    
    for (i_n, node) in enumerate(node_ids)
        input_lines = readlines(prefix*"$(node)")
        # Δt = 5e-5, T = 1e-1, 
        @assert(length(input_lines) == NT+1+1) # header
        for (i_obs, i_t) in enumerate(obs_ts)
            # skip the t=0 and header
            disps = parse.(Float64, split(input_lines[i_t + 2]))
            @assert(disps[1] ≈ i_t*ΔT)
            obs_data[i_obs, i_n] = sqrt(disps[2]^2 +  disps[3]^2 + disps[4]^2)
        end
    end

    return obs_data, obs_ts*ΔT
end


function generate_damage_ω(xx::Array{FT,1}, N_KL::IT, d::FT=1.0, τ::FT=2.0; θ::Union{Nothing, Array{FT, 1}} = nothing, seed::IT=123) where {FT<:AbstractFloat, IT<:Int}
    # xx = 1.5 + 3n, n= 0, 1 ,..., 9, the domain is [0,30]
    Lx = 30.0 
    N_x = length(xx) 
    φ = zeros(FT, N_KL, N_x)
    λ = zeros(FT, N_KL)
    
    # start from sqrt(2/Lx)cos(x1π/Lx), sqrt(2/Lx)cos(x2π/Lx), ... , sqrt(2/Lx)cos(xNπ/Lx)
    for l = 1:N_KL
        φ[l, :] = sqrt(2/Lx)*cos.(pi * l * xx/Lx)
    end
    
    if θ === nothing
        Random.seed!(seed);
        θ = rand(Normal(0, 1), N_KL)
    else
        @assert(length(θ) == N_KL)
    end

    logκ = zeros(FT, N_x)
    for l = 1:N_KL
        λ[l] = (pi^2*l^2/Lx^2  + τ^2/Lx^2)^(-d)

        logκ .+= θ[l]*sqrt(λ[l])*φ[l, :]
    end


    ω_max =  1.0
    ω_min = -0.1
    c = -ω_max/ω_min #2.0
    damage_ω = (ω_max - ω_min)./(1  .+ c * exp.(logκ)) .+ ω_min
    
    return damage_ω, logκ, φ, λ, θ
end

function generate_mat_file(output_file="../simulations/agard.fem.composite", damage_θ::Union{Nothing, Array{Float64, 1}} = nothing, N_θ::Int64 = 5)

    yy = Array(LinRange(1.5, 28.5, 10))
    
    @assert(damage_θ === nothing || N_θ == length(damage_θ))
    # hard code the first 5 modes
    damage_ω, _, _, _, θ = generate_damage_ω(yy, N_θ; θ = damage_θ)

    output_io = open(output_file, "w") 
    write(output_io, "********************************************************************\n")
    write(output_io, "COMPOSITE\n")
    write(output_io, "********************************************************************\n")

    input_lines = readlines("../calibration/agard.composite")
    for iy = 1:10
       for iLAYC = 10*(iy-1)+1:10*(iy-1) +10
           write(output_io, "LAYC      $(iLAYC)\n")
           
           ω = damage_ω[iy]
   
           properties = parse.(Float64, split(input_lines[2iLAYC]))
           write(output_io, "1    $(properties[2]*(1.0-ω))   $(properties[3]*(1.0-ω))  $(properties[4])  $(properties[5]*(1.0-ω))   $(properties[6])   $(properties[7])  $(properties[8])  $(properties[9])   $(properties[10]) \n")

       end
    end
    close(output_io);

    return damage_ω, θ
end

# # Plot damage field test
# xx = Array(LinRange(1.5, 28.5, 10))
# N_KL = 5
# PyPlot.figure()
# damage_ω, logκ, φ, λ, θ = generate_damage_ω(xx, N_KL, 1.0, 2.0; seed=123)
# @info θ
# PyPlot.plot(xx, damage_ω)

# damage_ω, logκ, φ, λ, θ = generate_damage_ω(xx, N_KL, 1.0, 2.0; θ=[0.0; 0.0; 0.0; 0.0; 0.0])
# PyPlot.plot(xx, damage_ω)


function generate_damage_ω_uq(xx::Array{FT,1}, N_KL::IT, d::FT=1.0, τ::FT=2.0; θ::Array{FT, 1}, θθ::Array{FT, 2}, n_std::FT=3.0) where {FT<:AbstractFloat, IT<:Int}
    # xx = 1.5 + 3n, n= 0, 1 ,..., 9, the domain is [0,30]
    Lx = 30.0 
    N_x = length(xx) 
    φ = zeros(FT, N_KL, N_x)
    λ = zeros(FT, N_KL)
    
    # start from sqrt(2/Lx)cos(x1π/Lx), sqrt(2/Lx)cos(x2π/Lx), ... , sqrt(2/Lx)cos(xNπ/Lx)
    for l = 1:N_KL
        φ[l, :] = sqrt(2/Lx)*cos.(pi * l * xx/Lx)
    end
    
    @assert(length(θ) == N_KL)

    logκ = zeros(FT, N_x)
    for l = 1:N_KL
        λ[l] = (pi^2*l^2/Lx^2  + τ^2/Lx^2)^(-d)

        logκ .+= θ[l]*sqrt(λ[l])*φ[l, :]
    end


    ω_max =  1.0
    ω_min = -0.1
    c = -ω_max/ω_min #2.0
    damage_ω = (ω_max - ω_min)./(1  .+ c * exp.(logκ)) .+ ω_min
    
    dlogκ_dθ = zeros(FT, N_x, N_KL)
    for l = 1:N_KL
        dlogκ_dθ[:, l] .= sqrt(λ[l])*φ[l, :]
    end
    dω_dθ = - (ω_max - ω_min)./(1  .+ c * exp.(logκ)).^2 .* (c * exp.(logκ)) .*  dlogκ_dθ
    damage_ωω = dω_dθ * θθ * dω_dθ'

    damage_ω_uq = n_std*[sqrt.(damage_ωω[i,i]) for i = 1: N_x]
    return damage_ω, damage_ω_uq
end


function Damage_Plot(ukiobj, N_iter = -1)
    if N_iter < 0
        N_iter = length(ukiobj.θ_mean) - 1
    end

    # optimization error
    ites = Array(1:N_iter)

    errors = zeros(Float64,  N_iter)
    for i = 1:N_iter
        errors[i] = 0.5*(ukiobj.y_pred[i] - ukiobj.y)'*(ukiobj.Σ_η\(ukiobj.y_pred[i] - ukiobj.y)) 
    end
    figure()
    plot(ites.-1, errors, linestyle="--", marker="o", fillstyle="none", markevery=1, label="UKI")
    ylabel("Optimization error")
    xlabel("Iterations")
    grid(true)
    tight_layout()
    legend()
    savefig("AGARD_Error.pdf")

    # UQ plot
    yy = Array(LinRange(1.5, 28.5, 10))
    damage_ω, _, _, _, θ = generate_damage_ω(yy, 10)
    n_std = 2.0
    damage_ω_uki, damage_ω_uq_uki= generate_damage_ω_uq(yy, 5;θ=ukiobj.θ_mean[N_iter], θθ=ukiobj.θθ_cov[N_iter], n_std=n_std)
    figure()
    plot(yy, damage_ω, "-*", label="Reference")
    plot(yy, zeros(length(yy)), "--", color = "green", label="UKI (initial)")
    plot(yy, damage_ω_uki, "-", color = "red", label="UKI")
    plot(yy, damage_ω_uki - damage_ω_uq_uki, "--", color = "red")
    plot(yy, damage_ω_uki + damage_ω_uq_uki, "--", color = "red")
    ylabel("ω")
    xlabel("Y (inch)")
    tight_layout()
    legend()
    savefig("AGARD_UQ.pdf")

    # UQ plot
    yy = Array(LinRange(1.5, 28.5, 10))
    yy2 = zeros(Float64, 20)
    damage_ω2 = zeros(Float64, 20)
    damage_ω_uki2, damage_ω_uq_uki2 = zeros(Float64, 20), zeros(Float64, 20)
    for i = 1:10
        yy2[2i-1] , yy2[2i] = 3.0*(i-1), 3.0*i
        damage_ω2[2i-1] = damage_ω2[2i] = damage_ω[i]
        damage_ω_uki2[2i-1] = damage_ω_uki2[2i] = damage_ω_uki[i]
        damage_ω_uq_uki2[2i-1] =  damage_ω_uq_uki2[2i] = damage_ω_uq_uki[i]
    end

    figure()
    plot(yy2, damage_ω2, "-*", label="Reference")
    plot(yy2, damage_ω_uki2, "-", color = "red", label="UKI")
    plot(yy2, damage_ω_uki2 - damage_ω_uq_uki2, "--", color = "red")
    plot(yy2, damage_ω_uki2 + damage_ω_uq_uki2, "--", color = "red")
    ylabel("ω")
    xlabel("Y (inch)")
    tight_layout()
    legend()
    savefig("AGARD_UQ2.pdf")


    # Displacement field
    NT = 50
    Δt = 0.1/NT
    tt = Array(LinRange(Δt, 0.1, NT))
    y_ref = reshape(ukiobj.y, NT, 12)
    y_init = reshape(ukiobj.y_pred[1], NT, 12)
    y_final = reshape(ukiobj.y_pred[N_iter], NT, 12)
    
    # [111, 116, 121, 56, 61, 66, 352, 362, 377, 197, 207, 222]
    # [111, 352, 56, 197]  => 1   7   4   10
    # [116, 362, 61, 207]  => 2   8   5   11
    # [121, 377, 66, 222]  => 3   9   6   12
    fig, ax = PyPlot.subplots(nrows = 3, sharex=true, sharey=true, figsize=(12,15))
    
    ax[1].plot(tt,  y_ref[:,10], "-o", fillstyle = "none", color = "C0", label="Observation")
    ax[1].plot(tt,  y_ref[:,4],  "-o", fillstyle = "none", color = "C0")
    ax[1].plot(tt,  y_ref[:,7],  "-o", fillstyle = "none", color = "C0")
    ax[1].plot(tt,  y_ref[:,1],  "-o", fillstyle = "none", color = "C0")

    ax[1].plot(tt,  y_init[:,10], "--", color = "green", label="UKI (initial)")
    ax[1].plot(tt,  y_init[:,4],  "--", color = "green")
    ax[1].plot(tt,  y_init[:,7],  "--", color = "green")
    ax[1].plot(tt,  y_init[:,1],  "--", color = "green")

    ax[1].plot(tt,  y_final[:,10], "-", color = "red", label="UKI")
    ax[1].plot(tt,  y_final[:,4],  "-", color = "red")
    ax[1].plot(tt,  y_final[:,7],  "-", color = "red")
    ax[1].plot(tt,  y_final[:,1],  "-", color = "red")


    ax[2].plot(tt,  y_ref[:,11], "-o", fillstyle = "none", color = "C0", label="Reference")
    ax[2].plot(tt,  y_ref[:,5],  "-o", fillstyle = "none", color = "C0")
    ax[2].plot(tt,  y_ref[:,8],  "-o", fillstyle = "none", color = "C0")
    ax[2].plot(tt,  y_ref[:,2],  "-o", fillstyle = "none", color = "C0")

    ax[2].plot(tt,  y_init[:,11], "--", color = "green", label="UKI (initial)")
    ax[2].plot(tt,  y_init[:,5],  "--", color = "green")
    ax[2].plot(tt,  y_init[:,8],  "--", color = "green")
    ax[2].plot(tt,  y_init[:,2],  "--", color = "green")

    ax[2].plot(tt,  y_final[:,11], "-", color = "red", label="UKI")
    ax[2].plot(tt,  y_final[:,5],  "-", color = "red")
    ax[2].plot(tt,  y_final[:,8],  "-", color = "red")
    ax[2].plot(tt,  y_final[:,2],  "-", color = "red")


    ax[3].plot(tt,  y_ref[:,12], "-o", fillstyle = "none", color = "C0", label="Reference")
    ax[3].plot(tt,  y_ref[:,6],  "-o", fillstyle = "none", color = "C0")
    ax[3].plot(tt,  y_ref[:,9],  "-o", fillstyle = "none", color = "C0")
    ax[3].plot(tt,  y_ref[:,3],  "-o", fillstyle = "none", color = "C0")

    ax[3].plot(tt,  y_init[:,12], "--", color = "green", label="UKI (initial)")
    ax[3].plot(tt,  y_init[:,6],  "--", color = "green")
    ax[3].plot(tt,  y_init[:,9],  "--", color = "green")
    ax[3].plot(tt,  y_init[:,3],  "--", color = "green")

    ax[3].plot(tt,  y_final[:,12], "-", color = "red", label="UKI")
    ax[3].plot(tt,  y_final[:,6],  "-", color = "red")
    ax[3].plot(tt,  y_final[:,9],  "-", color = "red")
    ax[3].plot(tt,  y_final[:,3],  "-", color = "red")
    
    ax[1].set_ylabel("Displacement (inch)")
    ax[2].set_ylabel("Displacement (inch)")
    ax[3].set_ylabel("Displacement (inch)")
    ax[3].set_xlabel("Time (s)")
    tight_layout()
    ax[1].legend()
    savefig("AGARD_Disp.pdf")

end


function Wing_Plot()
    X = zeros(Float64, 2, 11)
    Y = zeros(Float64, 2, 11)

    # UQ plot
    yy = Array(LinRange(1.5, 28.5, 10))
    damage_ω, _, _, _, θ = generate_damage_ω(yy, 10)

    for i = 1:11
        X[1, i], X[2, i] = 31.86/10*(i - 1), 21.96 + (46.36 - 21.96)/10*(i - 1)
        Y[1, i], Y[2, i] = 30.0/10*(i - 1), 30.0/10*(i - 1)
    end




    pcolormesh(X, Y, reshape(damage_ω, 1, 10), color="grey", cmap= "RdBu")
    

    colorbar()
    scatter([31.86, 31.86*3/4, 31.86*1/2, 31.86*1/4, 
             39.11, 39.11*3/4 + 10.98*1/4, 39.11*2/4 + 10.98*2/4, 39.11*1/4 + 10.98*3/4,
             46.36, 46.36*3/4 + 21.96*1/4, 46.36*2/4 + 21.96*2/4, 46.36*1/4 + 21.96*3/4],
            [30, 30*3/4, 30*1/2, 30*1/4, 
             30, 30*3/4, 30*1/2, 30*1/4,
             30, 30*3/4, 30*1/2, 30*1/4], color = "black")
    xlim([-1, 48])
    ylim([-1, 31])
    xlabel("X (inch)")
    ylabel("Y (inch)")
    tight_layout()
    savefig("Damage.pdf")
end