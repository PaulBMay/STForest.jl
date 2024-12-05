using STForest

using Random
using LinearAlgebra
using Plots
using CSV, DataFrames
using Distributions
using SparseArrays

#

n = 1000
nUnq = Integer(floor(n / 4))

Random.seed!(96)

locUnq = rand(nUnq, 2)
loc = locUnq[sample(1:nUnq, n), :]
time = rand(n,1)




#

zparams = (beta = 2, sw1 = 5, rangeS1 = 0.5, rangeT1 = 0.5, sw2 = 5, rangeS2 = 0.2)

m = 25

#

z = simulate_Bernoulli2(loc, time, zparams, m)

#quiltplot(loc, z)
#quiltplot(loc, zmu)

X = ones(n,1)

data = InputData(z, X, loc, time)

##############

const nb1 = STForest.getNeighbors(data.loc, m)
const nb2 = STForest.getNeighbors(locUnq, m)

theta = log.(collect(zparams)[2:6])
priors = (theta10 = [1, 0.1, 1], alpha10 = [0.5, 0.5, 0.5], theta20 = [2, 0.1], alpha20 = [0.5, 0.5], beta = [0 0.1])



#########
map2unq = indexin(STForest.loc2str(data.loc), STForest.loc2str(locUnq))
P = sparse(1:n, map2unq, 1)

Dsgn = sparse_hcat(data.X, STForest.speye(n), P)

STForest.thetaz_nlp2(theta, data, locUnq, Dsgn, nb1, nb2, priors, 1e-6, 10)

##########



map, logbook = bernoullimap2(randn(5), data, m, priors; f_tol = 1e-6, nr_tol = 1e-10, nr_maxiter = 100)

# How does the simple model do>

priors_simple = (theta0 = [3, 0.1, 1], alpha0 = [0.5, 0.5, 0.5], beta = priors.beta)

map_simple, logbook_simple = bernoullimap(randn(3), data, m, priors_simple; f_tol = 1e-6, nr_tol = 1e-10, nr_maxiter = 100)

Optim.minimum(logbook)
Optim.minimum(logbook_simple)

5*log(n) + 2*Optim.minimum(logbook)
3*log(n) + 2*Optim.minimum(logbook_simple)


#######################################

outDir = "./test/dump/"

if !isdir(outDir)
    mkdir(outDir)
end


thetaVar = 1e-3*Matrix(I,3,3)

NNGP_Bernoulli(data, m, map, priors, thetaVar, outDir, 1000, thetalog = true)


thetaVar = getPropVars("./test/dump/zparams.csv", ["sw", "rangeS", "rangeT"], 1000)
lastsamp = getLastSamp("./test/dump/zparams.csv", ["sw", "rangeS", "rangeT"])

NNGP_Bernoulli(data, m, map, priors, thetaVar, outDir, 100000, thetalog = true)

pardf_z= CSV.read("./test/dump/zparams.csv", DataFrame)

plot(pardf_z.sw)
plot(pardf_z.rangeS)
plot(pardf_z.rangeT)
plot(pardf_z.beta_0)

thetalog = CSV.read("./test/dump/zthetalog.csv", DataFrame)

rm(outDir, recursive = true)