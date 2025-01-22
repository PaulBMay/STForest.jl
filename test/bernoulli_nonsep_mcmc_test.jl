using STForest

using Random
using LinearAlgebra
using Plots
using CSV, DataFrames
using Distributions

#

n = 5000
nUnq = Integer(floor(n / 4))


Random.seed!(96)

locUnq = rand(nUnq, 2)
loc = locUnq[sample(1:nUnq, n), :]
time = rand(n,1)


zparams = (beta = 2, sw1 = 3, rangeS1 = 0.3, rangeT1 = 2, sw2 = 4, rangeT2 = 0.5, tSq = 0)

m = 25

nKnots = 10
#


gz = simulate_Continuous(loc, time, nKnots, zparams, m)

probz = STForest.softmax.(gz)

z = 1*(rand(n) .< probz)

X = ones(n,1)

zdata = InputData(z, X, loc, time)

#

outDir = "./test/dump/"

if !isdir(outDir)
    mkdir(outDir)
end

#

zpriors = (theta10 = [1, 0.1, 1], alpha10 = [0.5, 0.5, 0.5], theta20 = [1, 1], alpha20 = [0.5, 0.5], beta = [2 20])

timeKnots = reshape( collect(0:0.1:1), :, 1)


initparams = (sw1 = 3, rangeS1 = 0.3, rangeT1 = 2, sw2 = 4, rangeT2 = 0.5)
#map, logbook = bernoulli_nonsep_map(initparams, zdata, timeKnots, m, zpriors)

#map = (sw1 = 1.8284148281355268, rangeS1 = 0.17361834506337445, rangeT1 = 2.376555215857682, sw2 = 2.2554230294433, rangeT2 = 1.3293337201606898)

bernoulli_nonsep_mcmc(zdata, m, timeKnots, initparams, zpriors, outDir, 10000; pgwarmup = 100)




rm(outDir, recursive = true)



nKnots = size(timeKnots, 1)

pos = zdata.y .> 0

# Unique plot locations
locUnq = unique(zdata.loc, dims = 1)
nUnq = size(locUnq, 1)

# For each location in the full stack, what's it's position among the set of unique locations?
map2unq = indexin(STForest.loc2str(zdata.loc), STForest.loc2str(locUnq))

kt = nUnq*nKnots

Q2 = STForest.expCor(timeKnots, zparams.rangeT2)

Prec = kron(STForest.speye(nUnq), (zparams.sw2^2)*Q2)

w2 = randn(kt)

w2'*Prec*w2

w2mat = reshape(w2, nKnots, nUnq)

zparams.sw2^2*tr((w2mat*w2mat')*Q2)