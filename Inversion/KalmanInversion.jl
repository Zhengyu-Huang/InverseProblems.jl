using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions



include("Utility.jl")

# Kalman Inversion methods 
# which sample the posterior in an infinity time horizontal
include("UKI.jl")
include("EKI.jl")
# include("TUKI.jl")

# Ensemble Kalman sampler
include("EKS.jl")

# Iterative Kalman Filtering, continous time Kalman inversion, 
# which trasport the prior to the posterior in one time unit
include("CTKI.jl")

# Consensus-based sampler
include("CBS.jl")







