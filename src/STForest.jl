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
using Plots

########################
# Structures
#########################

struct InputData{T}
    y::Vector{T}
    X::Matrix{Float64}
    loc::Matrix{Float64}
    time::Matrix{Float64}
end

export InputData

#= struct SpatialParams
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
end =#

######################

include("misc.jl")
export quiltplot
export datasplit
export getPropVars
export getLastSamp

#######################

include("covariances.jl")

#######################

include("nngp.jl")

########################

include("samplers.jl")

#########################

include("likelihoods.jl")

#########################

#= include("inference.jl")

export dataSimulation
export NNGP_ZIST
export NNGP_ZIST_yMAP
export getPropVars
export getLastSamp
 =#
#########################

include("bernoulli_mcmc.jl")
export NNGP_Bernoulli

###############

include("bernoulli2_mcmc.jl")
export NNGP_Bernoulli2

###############

include("bernoulli_simulate.jl")
export simulate_Bernoulli

#########################

include("bernoulli_simulate2.jl")
export simulate_Bernoulli2

#########################

include("independent_timeseries.jl")

#########################

include("continuous_map.jl")
export NNGP_Continuous_MAP

#########################

include("continuous_mcmc.jl")
export NNGP_Continuous_MCMC

#########################

include("continuous_simulate.jl")
export simulate_Continuous

#######################

#include("prediction.jl")
#export NNGP_ZIST_PRED

########################

include("bernoulli_predict.jl")
export bernoulli_predict




end
