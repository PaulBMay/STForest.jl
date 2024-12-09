using STForest

using Random
using LinearAlgebra
using Plots
using CSV, DataFrames
using Distributions

using LinearSolve
using SparseArrays

using Preconditioners
using BenchmarkTools

#

n = 20000
nUnq = Integer(floor(n / 4))

Random.seed!(96)

locUnq = rand(nUnq, 2)
loc = locUnq[sample(1:nUnq, n), :]
time = rand(n,1)

m = 25

nb = STForest.getNeighbors(loc, m)
B,F,dump = STForest.getNNGPmatsST(nb, loc, time, 0.1, 1)

X = ones(n,1)

Dsgn = sparse_hcat(X, STForest.speye(n))

Qpost = blockdiag(spdiagm([1]), B'*spdiagm(1 ./ F)*B) + Dsgn'*Dsgn

b = randn(n+1)

@btime Pl = Preconditioners.CholeskyPreconditioner(Qpost, 2)

@time UpdatePreconditioner!(Pl, Qpost)

prob = LinearProblem(Qpost, b)

@btime sol = solve(prob, KrylovJL_CG(), Pl = Pl, reltol = 1e-6)

@btime sol2 = cholesky(Hermitian(Qpost)) \ b
