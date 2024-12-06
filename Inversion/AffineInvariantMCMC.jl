using Random
using LinearAlgebra

function Run_StretchMove(X0::Matrix,prob::Function;
    a::Real=2.0, output="Last", savedim::Int=size(X0,1), N_iter=100)

    """
    a: parameter in Stretch Move
    savedim: save the first (savedim) dimension.
    output: "Last" to save the final result, "History" to save all the results in iterations
    """
    
    @assert a>1

    #Inverse function of distribution funtion
    F(y)=(1/sqrt(a)+y*(sqrt(a)-1/sqrt(a)))^2
    
    N_x,N_ens=size(X0)
 
    X=copy(X0)
    Xhist=X0[1:savedim,:]


    for iter = 1:N_iter

        if iter%max(1,div(N_iter, 10)) == 0  @info "iter = ", iter, " / ", N_iter  end
        
        dX=zeros(N_x,N_ens)
        for k = 1:N_ens
            index=rand(1:N_ens)
            while index==k
                index=rand(1:N_ens)
            end
            Z=F(rand())
            Y=X[:,index]+Z*(X[:,k]-X[:,index])
            if rand()< prob(Y)/prob(X[:,k])*Z^(N_x-1)
                dX[:,k]=Y-X[:,k]
            end
        end
        X=X+dX
        if output=="History" 
            Xhist=cat(Xhist, X[1:savedim,:], dims=3)
        end
        
    end
    
    if output=="Last"
        return X
    else
        return Xhist 
    end       
end


"""
RandomSubset: return a subset A contained in {1,2,...,J} such that #A=S, k is not in A
"""
function RandomSubset(J::Int,k::Int,S::Int)
    @assert S<=J-1 && k<=J
    numbers = collect(1:J)  # 创建一个从1到J的数字列表
    shuffle!(numbers) 
    SubsetIndex=numbers[1:S]
    replace!(SubsetIndex,k=>numbers[end])
    return SubsetIndex
end

function Run_WalkMove(X0::Matrix,
    prob::Function;
    S::Int=-1,
    output::String="Last",
    N_iter::Int=100)

    N_x,N_ens=size(X0)

    @assert N_ens>3

    X=copy(X0)
    Xhist=copy(X0)
    
    if S==-1
        S=Int64(min(N_ens-1,floor(sqrt(N_ens))+1))
    end
    @assert S>=2 && S<=J-1

    
    for iter = 1:N_iter
        dX=zeros(size(X))
        
        for k = 1:J
            SubsetIndex=RandomSubset(N_ens,k,S)
            W=zeros(size(X,1))
            SubsetMean=sum(X[:,j] for j in SubsetIndex)/S
            
            for j in SubsetIndex
                W=W+randn()*(X[:,j]-SubsetMean)
            end
            
            if rand()<prob(X[:,k]+W)/prob(X[:,k])
                dX[:,k]=W
            end
        end
        X=X+dX
        if output=="History" 
            Xhist=cat(Xhist, X, dims=3)
        end
    end
        
    if output=="Last"
        return X
    else
        return Xhist 
    end
end
