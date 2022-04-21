using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions



include("Utility.jl")

# Kalman Inversion methods 
# which sample posterior in an infinite time horizon 
include("KI.jl")

# Iterative Kalman Filtering, continous time Kalman inversion, 
# which trasport the prior to the posterior in one time unit
include("IKF.jl")

# Ensemble Kalman sampler
include("EKS.jl")



# Consensus-based sampler
include("CBS.jl")







