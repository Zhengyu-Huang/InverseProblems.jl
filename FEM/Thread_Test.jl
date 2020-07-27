#export JULIA_NUM_THREADS=12
using SparseArrays

neq, n = 100, 1000
sol = zeros(Float64, neq, n)
Threads.@threads for i = 1:n
    @info Threads.threadid()
    ii = Array(1:neq)
    jj = Array(1:neq)
    vv = ones(Float64, neq)
    M = sparse(ii, jj, vv, neq, neq)
    sol[:,i] = M\vv
end