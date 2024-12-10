using STForest

using Random
using LinearAlgebra
using Plots
using CSV, DataFrames
using Distributions

#

n = 1000
nUnq = Integer(floor(n / 4))

nPred = 1000
nPredUnq = Integer(floor(nPred / 4))

Random.seed!(96)

locUnq = rand(nUnq, 2)
loc = locUnq[sample(1:nUnq, n), :]
time = rand(n,1)

locPredUnq = rand(nPredUnq, 2)
locPred = locPredUnq[sample(1:nPredUnq, nPred), :]
timePred = rand(nPred, 1)

locFull = vcat(loc, locPred)
timeFull = vcat(time, timePred)
#

posparams = (beta = 5, sw1 = 2, rangeS1 = 0.2, rangeT1 = 1, sw2 = 1, rangeT2 = 0.5, tSq = 0.01)
zparams = (beta = 2, sw = 4, rangeS = 0.3, rangeT = 2)

m = 25

#

zFull, zFullmu = simulate_Bernoulli(locFull, timeFull, zparams, m)

nKnots = 10


yposFull = simulate_Continuous(locFull, timeFull, nKnots, posparams, m)

yFull = yposFull .* zFull

y = yFull[1:n]
ypos = yposFull[1:n]
z = zFull[1:n]
zmu = zFullmu[1:n]

X = ones(n, 1)

data = InputData(y, X, loc, time)

zdata, posdata = datasplit(data)

#

outDir = "./test/dump/"

if !isdir(outDir)
    mkdir(outDir)
end

#

priors_pos = (theta10 = [1, 0.1, 1], alpha10 = [0.5, 0.5, 0.5], theta20 = [1, 1], alpha20 = [0.5, 0.5])
priors_z = (theta0 = [1, 0.1, 1], alpha0 = [0.5, 0.5, 0.5], beta = [2 20])

timeKnots = reshape( collect(0:0.1:1), :, 1)


posmap, posmaplog = NNGP_Continuous_MAP(posdata, m, timeKnots, posparams, priors_pos; f_tol = 1e-6)

thetaVar_pos = 1e-5*Matrix(I, 6, 6)

nSampsBurn = 500

NNGP_Continuous_MCMC(posdata, m, timeKnots, posmap, priors_pos, thetaVar_pos, outDir, nSampsBurn)


thetaVar_pos = getPropVars("./test/dump/yparams.csv", ["sw1", "rangeS1", "rangeT1", "sw2", "rangeT2", "tSq"], nSampsBurn)

nSamps = 1000

NNGP_Continuous_MCMC(posdata, m, timeKnots, posmap, priors_pos, thetaVar_pos, outDir, nSamps)


pardf_pos = CSV.read("./test/dump/yparams.csv", DataFrame)

plot(pardf_pos.sw1)
plot(pardf_pos.rangeS1)
plot(pardf_pos.rangeT1)
plot(pardf_pos.sw2)
plot(pardf_pos.rangeT2)
plot(pardf_pos.tSq)
plot(pardf_pos.beta_0)


#

thetaVar_z = 1e-3*Matrix(I,3,3)

NNGP_Bernoulli(zdata, m, zparams, priors_z, outDir, nSampsBurn, adaptStart = 50, thetalog = true)



thetaVar_z = getPropVars("./test/dump/zparams.csv", ["sw", "rangeS", "rangeT"], nSampsBurn)

NNGP_Bernoulli(zdata, m, zparams, priors_z, thetaVar_z, outDir, nSamps)

pardf_z= CSV.read("./test/dump/zparams.csv", DataFrame)

plot(pardf_z.sw)
plot(pardf_z.rangeS)
plot(pardf_z.rangeT)
plot(pardf_z.beta_0)


Xpred = ones(nPred, 1)

zsamps = bernoulli_predict(outDir, Xpred, locPred, timePred, m)

zPredmu = zFullmu[(n+1):(n+nPred)]

scatter(mean(zsamps, dims = 1)[1,:], zPredmu)

ysamps = continuous_predict(outDir, Xpred, locPred, timePred, m)


yposPred = yposFull[(n+1):(n+nPred)]


scatter(mean(ysamps, dims = 1)[1,:], yposPred)


function getCI(samps::AbstractArray, alpha::Number)

    q1 = (1 - alpha)/2
    q2 = 1 - q1

    n = size(samps, 2)

    CI = zeros(n, 2)

    for i = 1:n
        CI[i,:] = quantile(samps[:,i], [q1, q2])
    end

    return CI

end

yCI = getCI(ysamps, 0.95)

mean((yCI[:,1] .< yposPred) .&& (yCI[:,2] .> yposPred))


rm(outDir, recursive = true)
