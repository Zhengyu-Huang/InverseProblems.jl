using LinearAlgebra
using Optim

function Misfit2Diagcov(type::Int64, data_misfit, y)
    Ny = length(data_misfit)

    if type == 1
        # no fit
        return data_misfit.^2
    elseif type == 2
        # constant fit cov = σe^2
        new_cov =  sum(data_misfit.^2)/Ny   
        # estimation of the constant model error, and re-train
        return t_cov = fill(new_cov, Ny)
    elseif type == 3 
        
        # linear fit data_misfit^2 = σe^2 + γ^2 y^2

        # Direct fit
        A = ones(Ny, 2)
        A[:, 2] = y.^2
        x = A\data_misfit.^2
        @info "Optimization result is x.^2 = ", x
        if x[1] >= 0.0 && x[2] >= 0.0
            return x[1] .+ x[2]*y.^2
        end



        function f(x) 
            f = dot(x[1]^2 .+ x[2]^2*y.^2 - data_misfit.^2 , x[1]^2 .+ x[2]^2*y.^2 - data_misfit.^2)
            @info "f = ", f
            return f
        end
        function g!(storage, x)
            storage[1] = sum(2(x[1]^2 .+ x[2]^2*y.^2 - data_misfit.^2)*2*x[1])
            storage[2] = 2*dot(x[1]^2 .+ x[2]^2*y.^2 - data_misfit.^2, 2*x[2]*y.^2)
            @info "storage = ", storage
        end

        results = optimize(f, g!, [sqrt(sum(data_misfit.^2)/Ny), 0.05], LBFGS())

        x = Optim.minimizer(results)

        @info "Optimization result is x = ", x
        
        


        return x[1]^2 .+ x[2]^2*y.^2
        
        
    else
        error("Misfit2Diagcov type : ", type, " is not recogonized")
    end
    

end