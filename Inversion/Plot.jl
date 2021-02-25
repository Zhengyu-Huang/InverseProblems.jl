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
    
    θ_range = min(5*sqrt(θθ_cov), 5)
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
    
    θ_range = [min(5*sqrt(θθ_cov[1,1]), 5); min(5*sqrt(θθ_cov[2,2]), 5)]

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