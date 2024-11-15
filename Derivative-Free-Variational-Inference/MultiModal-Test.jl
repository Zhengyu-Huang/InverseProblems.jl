using Random
using Distributions
using LinearAlgebra
using ForwardDiff
include("../Inversion/Plot.jl")
include("../Inversion/GMVI.jl")

include("./MultiModal.jl")

# Test the examples in MultiModal-DFGMVI.ipynb with natural gradient flow

TEST_2 = true
if TEST_2
        N_modes_array = [10; 20; 40]    
        fig, ax = PyPlot.subplots(nrows=5, ncols=length(N_modes_array)+2, sharex=false, sharey=false, figsize=(20,16))

        Random.seed!(111);
        N_modes = N_modes_array[end]
        x0_w  = ones(N_modes)/N_modes
        μ0, Σ0 = [0.0; 0.0], [1.0 0.0; 0.0 1.0]
        N_x = length(μ0)
        x0_mean, xx0_cov = zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)
        for im = 1:N_modes
        x0_mean[im, :]    .= rand(MvNormal(zeros(N_x), Σ0)) + μ0
        xx0_cov[im, :, :] .= Σ0
        end
        Hessian_correct_GM = false

        N_iter = 20000
        Nx, Ny = 100,100

        ση = 1.0
        Gtype = "Gaussian"
        dt = 5e-1  
        A = [1.0 1.0; 1.0 2.0]
        y = [0.0; 1.0; zeros(N_x-2)]
        func_args = (y, ση, A , Gtype)
        func_F(x) = F(x, func_args)
        func_dPhi(x) = dPhi(x, func_args)
        objs = [Gaussian_mixture_VI(func_dPhi, nothing, x0_w[1:N_modes], x0_mean[1:N_modes,:], xx0_cov[1:N_modes,:,:]; N_iter = N_iter, dt = dt, Hessian_correct_GM=Hessian_correct_GM)[1] 
                for N_modes in N_modes_array]
        visualization_2d(ax[1,:]; Nx = Nx, Ny = Ny, x_lim=[-7.0, 5.0], y_lim=[-4.0, 5.0], func_F=func_F, objs=objs, label="GMVI")



        ση = 1.0
        dt = 3e-3 #2e-2 fails
        Gtype = "Four_modes"
        y = [4.2297; 4.2297; 0.5; 0.0; zeros(N_x-2)]
        func_args = (y, ση, 0, Gtype)
        func_F(x) = F(x, func_args)
        func_dPhi(x) = dPhi(x, func_args)
        objs = [Gaussian_mixture_VI(func_dPhi, nothing, x0_w[1:N_modes], x0_mean[1:N_modes,:], xx0_cov[1:N_modes,:,:]; N_iter = N_iter, dt = dt, Hessian_correct_GM=Hessian_correct_GM)[1] 
                for N_modes in N_modes_array]
        visualization_2d(ax[2,:]; Nx = Nx, Ny = Ny, x_lim=[-4.0, 4.0], y_lim=[-4, 4], func_F=func_F, objs=objs)



        ση = [0.5; ones(N_x-2)]
        Gtype = "Circle"
        dt = 1e-2     #2e-2 fails
        A = [1.0 1.0; 1.0 2.0]
        y = [1.0; zeros(N_x-2)]
        func_args = (y, ση, A , Gtype)
        func_F(x) = F(x, func_args)
        func_dPhi(x) = dPhi(x, func_args)
        μ0, Σ0 = [0.0; 0.0], [1.0 0.0; 0.0 1.0]
        objs = [Gaussian_mixture_VI(func_dPhi, nothing, x0_w[1:N_modes], x0_mean[1:N_modes,:], xx0_cov[1:N_modes,:,:]; N_iter = N_iter, dt = dt, Hessian_correct_GM=Hessian_correct_GM)[1] 
                for N_modes in N_modes_array]
        visualization_2d(ax[3,:]; Nx = Nx, Ny = Ny, x_lim=[-3.0, 3.0], y_lim=[-3.0, 3.0], func_F=func_F, objs=objs)



        ση = [sqrt(10.0); ones(N_x-2)]
        Gtype = "Banana"
        dt = 7e-3  #8e-3 fails
        λ = 10.0
        y = [0.0; 1.0; zeros(N_x-2)]
        func_args = (y, ση, λ , Gtype)
        func_F(x) = F(x, func_args)
        func_dPhi(x) = dPhi(x, func_args)
        μ0, Σ0 = [0.0; 0.0], [1.0 0.0; 0.0 1.0]
        objs = [Gaussian_mixture_VI(func_dPhi, nothing, x0_w[1:N_modes], x0_mean[1:N_modes,:], xx0_cov[1:N_modes,:,:]; N_iter = N_iter, dt = dt, Hessian_correct_GM=Hessian_correct_GM)[1] 
                for N_modes in N_modes_array]
        visualization_2d(ax[4,:]; Nx = Nx, Ny = Ny, x_lim=[-4.0, 4.0], y_lim=[-2.0, 10.0], func_F=func_F, objs=objs)



        ση = [0.3; 1.0; 1.0; ones(N_x-2)]
        Gtype = "Double_banana"
        dt = 1e-4    #2e-4 fails
        λ = 100.0
        y = [log(λ+1); 0.0; 0.0; zeros(N_x-2)]
        func_args = (y, ση, λ , Gtype)
        func_F(x) = F(x, func_args)
        func_dPhi(x) = dPhi(x, func_args)
        objs = [Gaussian_mixture_VI(func_dPhi, nothing, x0_w[1:N_modes], x0_mean[1:N_modes,:], xx0_cov[1:N_modes,:,:]; N_iter = N_iter, dt = dt, Hessian_correct_GM=Hessian_correct_GM)[1] 
                for N_modes in N_modes_array]
        visualization_2d(ax[5,:]; Nx = Nx, Ny = Ny, x_lim=[-3.0, 3.0], y_lim=[-3.0, 3.0], func_F=func_F, objs=objs)

        fig.tight_layout()
        fig.savefig("GMVI-2D-Multimodal.pdf")
end






TEST_100 = false
if TEST_100 

        N_modes_array = [10; 20; 40]    
        fig, ax = PyPlot.subplots(nrows=5, ncols=length(N_modes_array)+2, sharex=false, sharey=false, figsize=(20,16))

        Random.seed!(111);
        N_modes = N_modes_array[end]
        x0_w  = ones(N_modes)/N_modes
        N_x = 100
        μ0, Σ0 = zeros(N_x), Diagonal(ones(N_x))
        x0_mean, xx0_cov = zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)
        for im = 1:N_modes
        x0_mean[im, :]    .= rand(MvNormal(zeros(N_x), Σ0)) + μ0
        xx0_cov[im, :, :] .= Σ0
        end
        Hessian_correct_GM = false

        N_iter = 10000
        Nx, Ny = 100,100


        ση = 1.0
        Gtype = "Gaussian"
        dt = 1e-1
        A = [1.0 1.0; 1.0 2.0]
        y = [0.0; 1.0; zeros(N_x-2)]
        func_args = (y, ση, A , Gtype)
        func_F(x) = F(x, func_args)
        func_dPhi(x) = dPhi(x, func_args)
        objs = [Gaussian_mixture_VI(func_dPhi, nothing, x0_w[1:N_modes], x0_mean[1:N_modes,:], xx0_cov[1:N_modes,:,:]; N_iter = N_iter, dt = dt, Hessian_correct_GM=Hessian_correct_GM)[1] 
                for N_modes in N_modes_array]
        # compute marginal distribution
        y_2d = y[1:2]
        func_args = (y_2d, ση, A , Gtype)
        func_F(x) = F(x, func_args)
        visualization_2d(ax[1,:]; Nx = Nx, Ny = Ny, x_lim=[-7.0, 5.0], y_lim=[-4.0, 5.0], func_F=func_F, objs=objs, label="DF-GMVI")



        ση = 1.0
        dt = 2e-3
        Gtype = "Four_modes"
        y = [4.2297; 4.2297; 0.5; 0.0; zeros(N_x-2)]
        func_args = (y, ση, 0, Gtype)
        func_F(x) = F(x, func_args)
        func_dPhi(x) = dPhi(x, func_args)
        objs = [Gaussian_mixture_VI(func_dPhi, nothing, x0_w[1:N_modes], x0_mean[1:N_modes,:], xx0_cov[1:N_modes,:,:]; N_iter = N_iter, dt = dt, Hessian_correct_GM=Hessian_correct_GM)[1] 
                for N_modes in N_modes_array]
        # compute marginal distribution
        y_2d = y[1:4]
        func_args = (y_2d, ση, A , Gtype)
        func_F(x) = F(x, func_args)
        visualization_2d(ax[2,:]; Nx = Nx, Ny = Ny, x_lim=[-4.0, 4.0], y_lim=[-4, 4], func_F=func_F, objs=objs)



        ση = [0.5; ones(N_x-2)]
        Gtype = "Circle"
        dt = 5e-3
        A = [1.0 1.0; 1.0 2.0]
        y = [1.0; zeros(N_x-2)]
        func_args = (y, ση, A , Gtype)
        func_F(x) = F(x, func_args)
        func_dPhi(x) = dPhi(x, func_args)
        μ0, Σ0 = [0.0; 0.0], [1.0 0.0; 0.0 1.0]
        objs = [Gaussian_mixture_VI(func_dPhi, nothing, x0_w[1:N_modes], x0_mean[1:N_modes,:], xx0_cov[1:N_modes,:,:]; N_iter = N_iter, dt = dt, Hessian_correct_GM=Hessian_correct_GM)[1] 
                for N_modes in N_modes_array]
        # compute marginal distribution
        y_2d, ση_2d = y[1:1], ση[1:1]
        func_args = (y_2d, ση_2d, A , Gtype)
        func_F(x) = F(x, func_args)
        visualization_2d(ax[3,:]; Nx = Nx, Ny = Ny, x_lim=[-3.0, 3.0], y_lim=[-3.0, 3.0], func_F=func_F, objs=objs)



        ση = [sqrt(10.0); sqrt(10.0); ones(N_x-2)]
        Gtype = "Banana"
        dt = 2e-3
        λ = 10.0
        y = [0.0; 1.0; zeros(N_x-2)]
        func_args = (y, ση, λ , Gtype)
        func_F(x) = F(x, func_args)
        func_dPhi(x) = dPhi(x, func_args)
        μ0, Σ0 = [0.0; 0.0], [1.0 0.0; 0.0 1.0]
        objs = [Gaussian_mixture_VI(func_dPhi, nothing, x0_w[1:N_modes], x0_mean[1:N_modes,:], xx0_cov[1:N_modes,:,:]; N_iter = N_iter, dt = dt, Hessian_correct_GM=Hessian_correct_GM)[1] 
                for N_modes in N_modes_array]
        # compute marginal distribution
        y_2d, ση_2d = y[1:2], ση[1:2]
        func_args = (y_2d, ση_2d, λ , Gtype)
        func_F(x) = F(x, func_args)
        visualization_2d(ax[4,:]; Nx = Nx, Ny = Ny, x_lim=[-4.0, 4.0], y_lim=[-2.0, 10.0], func_F=func_F, objs=objs)



        ση = [0.3; 1.0; 1.0; ones(N_x-2)]
        Gtype = "Double_banana"
        dt = 1e-5
        λ = 100.0
        y = [log(λ+1); 0.0; 0.0; zeros(N_x-2)]
        func_args = (y, ση, λ , Gtype)
        func_F(x) = F(x, func_args)
        func_dPhi(x) = dPhi(x, func_args)
        objs = [Gaussian_mixture_VI(func_dPhi, nothing, x0_w[1:N_modes], x0_mean[1:N_modes,:], xx0_cov[1:N_modes,:,:]; N_iter = N_iter, dt = dt, Hessian_correct_GM=Hessian_correct_GM)[1] 
                for N_modes in N_modes_array]
        # compute marginal distribution
        y_2d, ση_2d = y[1:3], ση[1:3]
        func_args = (y_2d, ση_2d, λ , Gtype)
        func_F(x) = F(x, func_args)
        visualization_2d(ax[5,:]; Nx = Nx, Ny = Ny, x_lim=[-3.0, 3.0], y_lim=[-3.0, 3.0], func_F=func_F, objs=objs)

        fig.tight_layout()
        fig.savefig("GMVI-100D-Multimodal.pdf")


end
