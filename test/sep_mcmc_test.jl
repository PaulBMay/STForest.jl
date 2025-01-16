using STForest

using Random
using LinearAlgebra
using Plots
using CSV, DataFrames
using Distributions

#

n = 1000
nUnq = Integer(floor(n / 4))


Random.seed!(96)

locUnq = rand(nUnq, 2)
loc = locUnq[sample(1:nUnq, n), :]
time = rand(n,1)

#

posparams = (beta = 5, sw1 = 5, rangeS1 = 0.2, rangeT1 = 1, sw2 = 0, rangeT2 = 0.5, tSq = 0.01)

m = 25

#

nKnots = 10


y = simulate_Continuous(loc, time, nKnots, posparams, m)


X = ones(n, 1)

data = InputData(y, X, loc, time)


#

outDir = "./test/dump/"

if !isdir(outDir)
    mkdir(outDir)
end

#

priors = (theta0 = [1, 0.1, 1], alpha0 = [0.5, 0.5, 0.5])

initparams = (sw = posparams.sw1, rangeS = posparams.rangeS1, rangeT = posparams.rangeT1, tSq = posparams.tSq)

map, maplog = separable_map(data, m, initparams, priors; f_tol = 1e-6)

thetaVar = 1e-5*Matrix(I, 4, 4)

nSampsBurn = 500

separable_mcmc(data, m, map, priors, thetaVar, outDir, nSampsBurn)


thetaVar = getPropVars("./test/dump/yparams.csv", ["sw", "rangeS", "rangeT", "tSq"], nSampsBurn)

nSamps = 10000

separable_mcmc(data, m, map, priors, thetaVar, outDir, nSamps)


pardf_pos = CSV.read("./test/dump/yparams.csv", DataFrame)

plot(pardf_pos.sw)
plot(pardf_pos.rangeS)
plot(pardf_pos.rangeT)
plot(pardf_pos.tSq)
plot(pardf_pos.beta_0)


#

rm(outDir, recursive = true)
