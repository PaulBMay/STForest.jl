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

zparams = (beta = 2, sw = 20, rangeS = 0.5, rangeT = 0.5)

m = 25

#

z, zmu = simulate_Bernoulli(loc, time, zparams, m)

#quiltplot(loc, z)
#quiltplot(loc, zmu)

X = ones(n,1)

data = InputData(z, X, loc, time)

##############

const nb = STForest.getNeighbors(data.loc, m)

theta = log.([zparams.sw, zparams.rangeS, zparams.rangeT])
priors = (theta0 = [1, 0.1, 1], alpha0 = [0.5, 0.5, 0.5], beta = [2 0.1])

#########

    STForest.thetaz_nlp(theta, data, priors, nb, 1e-6, 10)
    STForest.thetaz_nlp_mcmc(theta, data, priors, nb, 10, 100, returnsd = true)

##########

swseq = 1:0.25:4
rSseq = 0.05:0.05:0.25
rTseq = 0.1:0.5:5

vals = zeros(length(swseq), length(rSseq), length(rTseq))
vals2 = copy(vals)

for i = 1:length(swseq)
    for j = 1:length(rSseq)
        for k = 1:length(rTseq)
            t = log.([swseq[i], rSseq[j], rTseq[k]])
            vals[i,j,k] = STForest.thetaz_nlp(t, data, priors, nb, 1e-6, 10)
            vals2[i,j,k] = STForest.thetaz_nlp_mcmc(t, data, priors, nb, 10, 100, returnsd = false)
        end
        println(j)
    end
    println(i)
    println("#######################")
end

minind = argmin(vals)
swseq[minind[1]], rSseq[minind[2]], rTseq[minind[3]]

minind2 = argmin(vals2)
swseq[minind2[1]], rSseq[minind2[2]], rTseq[minind2[3]]


heatmap(swseq, rSseq, vals[:,:,minind[3]])
heatmap(swseq, rTseq, vals[:,minind[2],:])
heatmap(rSseq, rTseq, vals[minind[1],:,:])

heatmap(swseq, rSseq, sum(vals, dims = 3)[:,:,1])
heatmap(swseq, rTseq, sum(vals, dims = 2)[:,1,:])
heatmap(rSseq, rTseq, vals[minind[1],:,:])

heatmap(swseq, rSseq, sum(vals2, dims = 3)[:,:,1])
heatmap(swseq, rTseq, sum(vals2, dims = 2)[:,1,:])
heatmap(rSseq, rTseq, sum(vals2, dims = 1)[1,:,:])




map, logbook = bernoullimap(randn(3), data, m, priors; f_tol = 1e-6, nr_tol = 1e-10, nr_maxiter = 100)

mode1, mu1 = STForest.thetaz_postmode(theta, data, priors, nb, 1e-10, 100)
mode2, mu2 = STForest.thetaz_postmode(Optim.minimizer(logbook), data, priors, nb, 1e-10, 100)

scatter(mu1, zmu)
scatter(mu2, zmu)

norm(mu1 - zmu)
norm(mu2 - zmu)

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