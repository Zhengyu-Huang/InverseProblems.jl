using LinearAlgebra
using Distributions
using Random
using SparseArrays
include("../Inversion/Plot.jl")
include("../Inversion/KalmanInversion.jl")
include("Spectral-Navier-Stokes.jl");



ν = 1.0/40                                      # viscosity
N, L = 128, 2*pi                                 # resolution and domain size 
ub, vb = 0.0, 0.0                               # background velocity 
method="Crank-Nicolson"                         # RK4 or Crank-Nicolson
N_t = 5000;                                     # time step
T = 10.0;                                        # final time
obs_ΔNx, obs_ΔNy, obs_ΔNt = 16, 16, 5000        #observation
σ = sqrt(2)*pi
N_KL = 0
N_θ = 100

Random.seed!(1);
N_data = 2000
θθ = rand(Normal(0,1),8, N_data)
mesh = Spectral_Mesh(N, N, L, L)
ωω = zeros(N,N, N_data)

for i_d = 1:N_data
    θ = θθ[:, i_d]
    # this is used for generating random initial condition
    s_param = Setup_Param(ν, ub, vb,  
        N, L,  
        method, N_t,
        obs_ΔNx, obs_ΔNy, obs_ΔNt, 
        N_KL,
        N_θ;
        f = (x, y) -> ((θ[3]*cos(y)-θ[4]*sin(y))/(sqrt(2)*pi), 
                       (-θ[1]*cos(x)+θ[2]*sin(x)
                        -θ[5]*cos(x+y)/2 - θ[6]*cos(x-y)/2 
                        +θ[7]*sin(x+y)/2 + θ[8]*sin(x-y)/2)/(sqrt(2)*pi)),
        σ = σ)


    ω0_ref = s_param.ω0_ref

    solver = Spectral_NS_Solver(mesh, ν; fx = s_param.fx, fy = s_param.fy, ω0 = ω0_ref, ub = ub, vb = vb)  
    
    if i_d == 1
        PyPlot.figure(figsize = (4,3))
        Visual(mesh, solver.ω, "ω")
    end

    Δt = T/N_t 
    for i = 1:N_t
        Solve!(solver, Δt, method)
        if i%N_t == 0
            Update_Grid_Vars!(solver)
            PyPlot.figure(figsize = (4,3))
            Visual(mesh, solver.ω, "ω")
        end
    end
    ωω[:, : , i_d] .= solver.ω
end

npzwrite("random_8_direct_theta.npy", θθ)
npzwrite("random_8_direct_omega.npy", ωω)

