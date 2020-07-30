function Rk4_Update!(t::Float64, Δt::Float64, f::Function, Q::Array{Float64,1})
    k1 = Δt*f(t, Q)
    k2 = Δt*f(t+0.5*Δt, Q+0.5*k1)
    k3 = Δt*f(t+0.5*Δt, Q+0.5*k2)
    k4 = Δt*f(t+Δt, Q+k3)

    Q .+= (k1 + 2*k2 + 2*k3 + k4)/6
end

