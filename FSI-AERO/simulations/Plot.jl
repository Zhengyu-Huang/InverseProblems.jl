using PyPlot
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    mysize = 15
    font0 = Dict(
    "font.size" => mysize,
    "axes.labelsize" => mysize,
    "xtick.labelsize" => mysize,
    "ytick.labelsize" => mysize,
    "legend.fontsize" => mysize,
    )
merge!(rcParams, font0)



function Gaussian_1d(θ_mean::FT, θθ_cov::FT, Nx::IT) where {FT<:AbstractFloat, IT<:Int}
    # 1d Gaussian plot
    
    θ_range = 5*sqrt(θθ_cov)
    θ = Array(LinRange(θ_mean - θ_range, θ_mean + θ_range, Nx))
    ρθ = similar(θ)
    
    for ix = 1:Nx  
        Δθ = θ[ix] - θ_mean
        ρθ[ix] = exp(-0.5*(Δθ/θθ_cov*Δθ)) / (sqrt(2 * pi * θθ_cov))
    end
    return θ, ρθ 
end


function Gaussian_2d(θ_mean::Array{FT,1}, θθ_cov, Nx::IT, Ny::IT) where {FT<:AbstractFloat, IT<:Int}
    # 2d Gaussian plot
    
    θ_range = [5*sqrt(θθ_cov[1,1]); 5*sqrt(θθ_cov[2,2])]
    xx = Array(LinRange(θ_mean[1] - θ_range[1], θ_mean[1] + θ_range[1], Nx))
    yy = Array(LinRange(θ_mean[2] - θ_range[2], θ_mean[2] + θ_range[2], Ny))
    
    X,Y = repeat(xx, 1, Ny), repeat(yy, 1, Nx)'
    Z = zeros(FT, Nx, Ny)
    
    det_θθ_cov = det(θθ_cov)

    for ix = 1:Nx
        for iy = 1:Ny
            Δxy = [xx[ix] - θ_mean[1]; yy[iy] - θ_mean[2]]
            Z[ix, iy] = exp(-0.5*(Δxy'/θθ_cov*Δxy)) / (2 * pi * sqrt(det_θθ_cov))
        end
    end
    
    return X, Y, Z
    
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