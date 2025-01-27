module STForest

using NearestNeighbors
using LinearAlgebra
using Distances
using SparseArrays
using Random
using Distributions
using DataFrames, CSV
using PolyaGammaSamplers
using ProgressBars
using Optim

########################
# Structures
#########################

struct InputData
    y::Vector{Float64}
    X::Matrix{Float64}
    loc::Matrix{Float64}
    time::Matrix{Float64}
end

struct SpatialParams
    m::Int64
    timeKnots::Matrix{Float64}
end

struct McmcParams
    thetayVar::Matrix{Float64}
    thetazVar::Matrix{Float64}
end

struct SpatialPriors
    thetay10::Vector{Float64}
    alphay10::Vector{Float64}
    thetay20::Vector{Float64}
    alphay20::Vector{Float64}
    thetaz0::Vector{Float64}
    alphaz0::Vector{Float64}
end

######################

include("misc.jl")
export quiltplot

#######################

include("covariances.jl")

########################

include("samplers.jl")

#########################

include("likelihoods.jl")

#########################

include("inference.jl")

export dataSimulation
export NNGP_ZIST
export NNGP_ZIST_yMAP
export getPropVars
export getLastSamp

#########################

include("prediction.jl")
export NNGP_ZIST_PRED




end
