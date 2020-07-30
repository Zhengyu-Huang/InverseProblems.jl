struct Params
    K::Int64
    J::Int64

    F::Float64
    h::Float64
    c::Float64
    d::Float64
end


function perd(j::Int64, J::Int64)
    if j > 0 && j <= J
        return j
    elseif j <= 0 
        return j + J
    else # k > K
        return j - J
    end
end

"""
Q = X Y(k=1), Y(k=2) ... Y(k=K)
"""
function rhs(phys_params::Params, Q::Array{Float64, 1})
    X = Q[1:K]
    Y = 
    for k = 1:K
        Q[k] = - Q[perd(k-1, K)]*(Q)
end