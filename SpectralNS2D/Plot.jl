using JLD2
include("../Plot.jl")
include("Random_Init.jl")
include("../RUKI.jl")
include("../REKI.jl")


@load "ekiobj.dat" ekiobj_θ ekiobj_g_bar
@load "ukiobj.dat" ukiobj_θ_bar ukiobj_g_bar


phys_params = Params()
obs_cov = Array(Diagonal(fill(1.0, phys_params.n_data))) 
mesh = Spectral_Mesh(phys_params.nx, phys_params.ny, phys_params.Lx, phys_params.Ly)
ω_ref, obs_ref =  Generate_Data(phys_params)
na = 100
seq_pairs = Compute_Seq_Pairs(na)


N_ite = length(ukiobj_θ_bar)-1
uki_errors = zeros(N_ite, 2)
eki_errors = zeros(N_ite, 2)
for i = 1:N_ite
    ω_uki = Initial_ω0_KL(mesh, ukiobj_θ_bar[i], seq_pairs)
    uki_errors[i,1] = norm(ω_ref - ω_uki) / norm(ω_ref)
    uki_errors[i,2] = (ukiobj_g_bar[i] - obs_ref)'*(obs_cov\(ukiobj_g_bar[i] - obs_ref))
    
    ω_eki = Initial_ω0_KL(mesh, dropdims(mean(ekiobj_θ[i], dims=1), dims=1) , seq_pairs)
    eki_errors[i,1] = norm(ω_ref - ω_eki) / norm(ω_ref)
    eki_errors[i,2] = (ekiobj_g_bar[i] - obs_ref)'*(obs_cov\(ekiobj_g_bar[i] - obs_ref))
end
ites = Array(LinRange(1, N_ite, N_ite))
plot(ites, uki_errors[:,1], "--o", markevery=2, fillstyle="none", label= "UKI")
plot(ites, eki_errors[:,1], "--o", markevery=2, fillstyle="none", label= "EnKI")
xlabel("Iterations")
ylabel("Relative L₂ norm error")
#ylim((0.1,15))
grid("on")
legend()
tight_layout()
savefig("SpectralNS_Param.pdf")
close("all")




plot(ites, uki_errors[:,2], "--o", markevery=2, fillstyle="none", label= "UKI")
plot(ites, eki_errors[:,2], "--o", markevery=2, fillstyle="none", label= "EnKI")
xlabel("Iterations")
ylabel("Optimization error")
#ylim((0.1,15))
grid("on")
legend()
tight_layout()
savefig("SpectralNS_Data_Misfit.pdf")
close("all")