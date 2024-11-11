using STForest

using Random
using LinearAlgebra


n = 1000

Random.seed!(96)

loc = rand(n,2)
time = rand(n,1)

yparams = (beta = 5, sw1 = 1, rangeS1 = 0.1, rangeT1 = 1, sw2 = 2, rangeT2 = 0.5, tSq = 0.01)
zparams = (beta = 2, sw = 4, rangeS = 0.3, rangeT = 2)

m = 25

z = simulate_Bernoulli(loc, time, zparams, m)

nKnots = 10
ypos = simulate_Continuous(loc, time, nKnots, yparams, m)

