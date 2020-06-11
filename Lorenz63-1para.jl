using PyPlot
using JLD2
using Statistics
using LinearAlgebra

include("EKI.jl")
include("UKI.jl")

# J = x_j[ix]^m + x_{j+1}[ix]^m + ... + x_{k}[ix]^m
# x_1 = μ0
# x_{i+1} = x_i + Δt f(x_i, μ1), i = 1, 2, ... nt
# here μ0 is the initial condition, and μ1 is the parameter [σ=10.0; r=28.0; β=8.0/3.0]
# 

function f(x::Array{Float64,1}, σ::Float64, r::Float64, β::Float64)
    return [σ*(x[2]-x[1]); x[1]*(r-x[3])-x[2]; x[1]*x[2]-β*x[3]] 
end

function df_du(x::Array{Float64,1}, σ::Float64, r::Float64, β::Float64)
    return  [-σ        σ       0.0; 
    (r-x[3])  -1.0    -x[1]; 
    x[2]      x[1]    -β]
end

function df_dμ(x::Array{Float64,1}, σ::Float64, r::Float64, β::Float64)
    return [(x[2]-x[1]) 0.0    0.0; 
    0.0        x[1]   0.0; 
    0.0        0.0   -x[3]] 
    
end

function compute_foward(μ::Array{Float64,1}, Δt::Float64, nt::Int64)
    @assert(length(μ) == 6)
    nx = 3
    x0 = μ[1:nx]
    σ,r, β = μ[nx+1:6]
    
    xs = zeros(nx, nt+1)
    xs[:,1] = x0
    for i=2:nt+1
        xs[:,i] = xs[:,i-1] + Δt*f(xs[:,i-1], σ, r, β)
    end
    
    return xs
end

function compute_foward_RK4(μ::Array{Float64,1}, Δt::Float64, nt::Int64)
    @assert(length(μ) == 6)
    nx = 3
    x0 = μ[1:nx]
    σ,r, β = μ[nx+1:6]
    
    xs = zeros(nx, nt+1)
    xs[:,1] = x0
    for i=2:nt+1
        k1 = Δt*f(xs[:,i-1], σ, r, β)
        k2 = Δt*f(xs[:,i-1] + k1/2, σ, r, β)
        k3 = Δt*f(xs[:,i-1] + k2/2, σ, r, β)
        k4 = Δt*f(xs[:,i-1] + k3, σ, r, β)
        
        xs[:,i] = xs[:,i-1] + k1/6 + k2/3 + k3/3 + k4/6
    end
    
    return xs
end

function compute_J_helper(x::Array{Float64,1})
    # todo hard coded here ix, m
    ix, m = 3, 1
    nx = length(x)
    J = x[ix]^m
    pJ_px = zeros(nx)
    pJ_px[ix] = m*x[ix]^(m-1)
    return J, pJ_px
end

function compute_J(xs::Array{Float64,2}, μ::Array{Float64,1}, j::Int64, k::Int64)
    # J is defined as the sum 
    # J = x_j[ix]^m + x_j+1[ix]^m ... + x_k[ix]^m /(k-j+1)
    # pJ/pμ = 0
    # return J and dJ/dxs
    nμ = 6
    @assert(length(μ) == nμ)
    
    nx, nt = size(xs, 1), size(xs, 2) - 1
    σ,r, β = μ[nx+1:nμ]
    
    J = 0.0
    pJ_pxs = zeros(nx, nt+1)
    
    for i = j:k
        J_i, pJ_pxs_i = compute_J_helper(xs[:,i])
        J +=  J_i
        pJ_pxs[:,i] = pJ_pxs_i
    end
    
    
    pJ_pxs ./= (k-j+1)
    J /= (k-j+1)
    
    return J, pJ_pxs
end

function compute_adjoint(μ::Array{Float64,1}, xs::Array{Float64,2}, pJ_pxs::Array{Float64,2}, Δt::Float64, nt::Int64)
    nμ = 6
    @assert(length(μ) == nμ)
    nx = 3
    x0 = μ[1:nx]
    σ, r, β = μ[nx+1:nμ]
    
    
    lambdas = zeros(nx, nt+1)
    lambdas[:,nt+1] .= pJ_pxs[:,nt+1]
    
    for i=nt:-1:1
        lambdas[:,i] = pJ_pxs[:,i] + lambdas[:,i+1] + Δt*df_du(xs[:,i], σ, r, β)'*lambdas[:,i+1]   
    end
    
    dJ_dμ = zeros(nμ)
    for i=2:nt+1
        dJ_dμ[nx+1:nμ] .+= df_dμ(xs[:,i-1], σ, r, β)' * lambdas[:,i]*Δt
    end
    
    dJ_dμ[1:nx] = lambdas[:,1]
    
    #J = ∑_j^k f(xi)/(k-j+1)
    return dJ_dμ
end



function visualize_stat(xs::Array{Float64,2}, Δt::Float64, nt::Int64, nspinup::Int64)
    @assert(nt+1 == size(xs,2))
    T = Δt * nt
    ts =  LinRange(0,T,nt+1)
    
    ns = 9
    # x1, x2, x3, x1^2, x2^2, x3^2, x2x3, x3x1, x1x2
    stats = zeros(ns, nt+1)
    
    for i = nspinup+1:nt+1
        x1, x2, x3 = xs[:,i]
        moment = [x1, x2, x3, x1^2, x2^2, x3^2, x2*x3, x3*x1, x1*x2]
        stats[:,i] .= stats[:,i-1] + moment
    end
    
    for i = nspinup+1:nt+1
        stats[:,i] /= (i - nspinup)
    end
    
    
    fig = figure(figsize=(6,6))
    for pid = 1:9
        ax = fig.add_subplot(9,1,pid)
        ax.plot(ts, stats[pid,:])
    end
    
end

function visualize_states(xs::Array{Float64,2}, Δt::Float64, nt::Int64)
    @assert(nt+1 == size(xs,2))
    T = Δt * nt
    ts =  LinRange(0,T,nt+1)
    
    fig = figure(figsize=(6,6))
    for pid = 1:3
        ax = fig.add_subplot(9,1,pid)
        ax.plot(ts, xs[pid,:])
    end
    
end

function fd_test()
    T = 20
    Δt = 0.01
    nt = Int64(T/Δt)
    #σ, r, β = 10.0, 28.0, 8.0/3.0
    # non-chaotic case
    σ, r, β = 1.0, 2.0, 3.0
    
    μ = [-8.67139571762; 4.98065219709; 25; σ; r; β]
    nμ = length(μ)
    δμ = rand(nμ)
    #δμ = [1.0;1.0;1.0;1.0;1.0;1.0]
    #δμ = [1.0;2.0;3.0;4.0;5.0;6.0]
    ε = 1.0e-4
    
    xs = compute_foward(μ, Δt, nt)
    J, pJ_pxs = compute_J(xs, μ, 1, nt+1)
    dJ_dμ = compute_adjoint(μ, xs, pJ_pxs, Δt, nt)
    
    
    
    μ_p = μ + δμ*ε
    xs_p = compute_foward(μ_p, Δt, nt)
    J_p, _ = compute_J(xs_p, μ_p, 1, nt+1)
    
    
    μ_m = μ - δμ*ε
    xs_m = compute_foward(μ_m, Δt, nt)
    J_m, _ = compute_J(xs_m, μ_m, 1, nt+1)
    
    @info "fd error is ", (J_p - J_m)/(2*ε) .- dJ_dμ'*δμ
    
end


function adjoint_plot()
    @load "r-x3.dat" x3_arr
    @load "dr-dx3.dat" dx3_dr_arr

    r_max = 50.0
    N_r = length(x3_arr)
    r_arr = Array(LinRange(0, r_max, N_r))

    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    mysize = 18
    font0 = Dict(
    "font.size" => mysize,
    "axes.labelsize" => mysize,
    "xtick.labelsize" => mysize,
    "ytick.labelsize" => mysize,
    "legend.fontsize" => mysize,
    )
    merge!(rcParams, font0)
    
    plot(r_arr, x3_arr, "--", fillstyle="none")
    xlabel(L"r")
    ylabel(L"\overline{x_3}")
    grid("on")
    tight_layout()
    savefig("Lorenz_J.pdf")
    close("all")
    
    semilogy(r_arr, abs.(dx3_dr_arr), "or", markersize=1)
    xlabel("\$r\$")
    ylabel(L"|\frac{d\overline{x_3}}{dr}|")
    grid("on")
    tight_layout()
    savefig("Lorenz_dJ.pdf")
    close("all")

    #filter the result
    dr = r_arr[2] - r_arr[1]
    filtered_x3_arr = copy(x3_arr)
    filtered_dx3_dr_arr = copy(dx3_dr_arr)

    σr = 0.1
    filter_Δ = Int64(ceil(σr/dr))
    for i = 1:N_r
        if (i > filter_Δ && i < N_r - filter_Δ)
            filtered_x3 = 0.0
            filtered_dx3_dr = 0.0
            filtered_dx3_dr_numerator = 0.0
            filtered_dx3_dr_denominator = 0.0

            for j = i-filter_Δ:i+filter_Δ
                filtered_x3 += x3_arr[j]
                filtered_dx3_dr_numerator += (r_arr[j] - r_arr[i])*(x3_arr[j] - x3_arr[i])
                filtered_dx3_dr_denominator += (r_arr[j] - r_arr[i])*(r_arr[j] - r_arr[i])
            end

            filtered_x3_arr[i] =  filtered_x3/(2*filter_Δ+1)
            filtered_dx3_dr_arr[i] =  filtered_dx3_dr_numerator/filtered_dx3_dr_denominator
        end
    end

    plot(r_arr[filter_Δ+1:N_r - filter_Δ-1], filtered_x3_arr[filter_Δ+1:N_r - filter_Δ-1], "--", fillstyle="none")
    xlabel(L"r")
    ylabel(L"\mathcal{F}\overline{x_3}")
    grid("on")
    tight_layout()
    savefig("Filtered_Lorenz_J.pdf")
    close("all")
    
    plot(r_arr[filter_Δ+1:N_r - filter_Δ-1], filtered_dx3_dr_arr[filter_Δ+1:N_r - filter_Δ-1], "or", markersize=1)
    xlabel("\$r\$")
    ylabel(L"\mathcal{F}\frac{d\overline{x_3}}{dr}")
    grid("on")
    tight_layout()
    savefig("Filtered_Lorenz_dJ.pdf")
    close("all")

end


function adjoint_demo()
    
    Tobs = 20.0
    Tspinup = 30.0
    Δt = 0.001
    
    T = Tobs + Tspinup
    nt = Int64(T/Δt)
    nspinup = Int64(Tspinup/Δt)
    
    r_max = 50.0
    N_r = Int64(r_max/0.005) + 1
    r_arr = Array(LinRange(0, r_max, N_r))
    
    
    x3_arr = zeros(Float64, N_r)
    dx3_dr_arr = zeros(Float64, N_r)
    for i = 1:N_r
        σ, r, β = 10.0, r_arr[i], 8.0/3.0
        
        μ = [-8.67139571762; 4.98065219709; 25; σ; r; β]
        xs = compute_foward(μ, Δt, nt)
        
        
        J, pJ_pxs = compute_J(xs, μ, nspinup+1, nt+1)
        dJ_dμ = compute_adjoint(μ, xs, pJ_pxs, Δt, nt)
        x3_arr[i], dx3_dr_arr[i] = J, dJ_dμ[5]
        
        @info i
        
    end

    @save "r-x3.dat" x3_arr
    @save "dr-dx3.dat" dx3_dr_arr

    # @load "r-x3.dat" x3_arr
    # @load "dr-dx3.dat" dx3_dr_arr

    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    mysize = 18
    font0 = Dict(
    "font.size" => mysize,
    "axes.labelsize" => mysize,
    "xtick.labelsize" => mysize,
    "ytick.labelsize" => mysize,
    "legend.fontsize" => mysize,
    )
    merge!(rcParams, font0)
    
    plot(r_arr, x3_arr, "--", fillstyle="none")
    xlabel(L"r")
    ylabel(L"\overline{x_3}")
    grid("on")
    tight_layout()
    savefig("Lorenz_J.pdf")
    close("all")
    
    semilogy(r_arr, abs.(dx3_dr_arr), "or", markersize=1)
    xlabel("\$r\$")
    ylabel(L"|\frac{d\overline{x_3}}{dr}|")
    grid("on")
    tight_layout()
    savefig("Lorenz_dJ.pdf")
    close("all")
    
    
    return r_arr, x3_arr, dx3_dr_arr
    
end


function compute_obs(xs::Array{Float64,2})
    # N_x × N_t
    x1, x2, x3 = xs[1,:]', xs[2,:]', xs[3,:]'
    obs = [x3;]
end

# fix σ, learn r and β
# f = [σ*(x[2]-x[1]); x[1]*(r-x[3])-x[2]; x[1]*x[2]-β*x[3]]
# σ, r, β = 10.0, 28.0, 8.0/3.0
function run_Lorenz_ensemble(params_i, Tobs, Tspinup, Δt)
    T = Tobs + Tspinup
    nt = Int64(T/Δt)
    nspinup = Int64(Tspinup/Δt)
    
    N_ens,  N_θ = size(params_i)
    
    
    g_ens = Vector{Float64}[]
    
    for i = 1:N_ens
        σ,  β = 10.0, 8.0/3.0
        r = params_i[i, :]
        
        
        μ = [-8.67139571762; 4.98065219709; 25; σ; r; β]
        xs = compute_foward(μ, Δt, nt)
        obs = compute_obs(xs)
        # g: N_ens x N_data
        push!(g_ens, dropdims(mean(obs[:, nspinup : end], dims = 2), dims=2)) 
    end
    return hcat(g_ens...)'
end


function Data_Gen(T::Float64 = 360.0, Tobs::Float64 = 10.0, Tspinup::Float64 = 30.0, Δt::Float64 = 0.01)
    
    nt = Int64(T/Δt)
    nspinup = Int64(Tspinup/Δt)
    σ, r, β = 10.0, 28.0, 8.0/3.0
    #σ, r, β = 10.0, 8.0/3.0, 28.0
    
    μ = [-8.67139571762; 4.98065219709; 25; σ; r; β]
    xs = compute_foward(μ, Δt, nt)
    
    obs = compute_obs(xs)
    
    ##############################################################################
    
    n_obs_box = Int64((T - Tspinup)/Tobs)
    obs_box = zeros(size(obs, 1), n_obs_box)
    
    n_obs = Int64(Tobs/Δt)
    
    for i = 1:n_obs_box
        n_obs_start = nspinup + (i - 1)*n_obs + 1
        obs_box[:, i] = mean(obs[:, n_obs_start : n_obs_start + n_obs - 1], dims = 2)
    end
    
    t_mean = vec(mean(obs_box, dims=2))
    #t_mean = vec(obs_box[:, 1])
    
    t_cov = zeros(Float64, size(obs, 1), size(obs, 1))
    for i = 1:n_obs_box
        t_cov += (obs_box[:,i] - t_mean) *(obs_box[:,i] - t_mean)'
    end
    
    @info obs_box
    t_cov ./= (n_obs_box - 1)
    
    @save "t_cov.jld2" t_cov
    @save "t_mean.jld2" t_mean
    return t_mean, t_cov
end



function UKI_Run(t_mean, t_cov, θ_bar, θθ_cov, Tobs::Float64 = 10.0, Tspinup::Float64 = 30.0, Δt::Float64 = 0.01, 
    N_iter::Int64 = 100, update_cov::Int64 = 0)
    
    parameter_names = ["r"]
    
    
    ens_func(θ_ens) = run_Lorenz_ensemble(θ_ens, Tobs, Tspinup, Δt)
    
    ukiobj = UKIObj(parameter_names,
    θ_bar, 
    θθ_cov,
    t_mean, # observation
    t_cov)
    
    
    for i in 1:N_iter

        params_i = deepcopy(ukiobj.θ_bar[end])
        
        @info "At iter ", i, " params_i : ", params_i
        
        update_ensemble!(ukiobj, ens_func) 
        
        if (update_cov) > 0 && (i%update_cov == 1) 
            reset_θθ0_cov!(ukiobj)
        end
    end
    
    @info "θ is ", ukiobj.θ_bar[end], " θ_ref is ",  28.0

    return ukiobj
    
end


#adjoint_demo()
#adjoint_plot()

@info "start"
Tobs = 20.0
Tspinup = 30.0
T = Tspinup + 5*Tobs
Δt = 0.001

t_mean, t_cov = Data_Gen(T, Tobs, Tspinup, Δt)


t_cov = Array(Diagonal(0.01*t_mean))

# initial distribution is 
θ0_bar = [5.0;]          # mean 
θθ0_cov = rashape([0.5^2], (1,1))        # standard deviation
          

N_ite = 20 
update_cov = 0

ukiobj = UKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov,  Tobs, Tspinup, Δt, N_ite, update_cov)
ites = Array(LinRange(1, N_ite+1, N_ite+1))
θ_bar = ukiobj.θ_bar
θθ_cov = ukiobj.θθ_cov

θ_bar_arr = hcat(θ_bar...)



rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    mysize = 18
    font0 = Dict(
    "font.size" => mysize,
    "axes.labelsize" => mysize,
    "xtick.labelsize" => mysize,
    "ytick.labelsize" => mysize,
    "legend.fontsize" => mysize,
    )
    merge!(rcParams, font0)

    
plot(ites, θ_bar_arr[1,:], "--o", fillstyle="none", label=L"r")
plot(ites, fill(28.0, N_ite+1), "--", color="gray")

xlabel("Iterations")
legend()
grid("on")
tight_layout()
savefig("Lorenz_inverse-1para.pdf")
close("all")