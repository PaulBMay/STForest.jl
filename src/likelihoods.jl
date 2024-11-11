
# PC prior for exponential covariance (par = [sigma, range])
function pcprior(par::Vector{Float64}, par0::Vector, alpha::Vector{Float64})

    sig, rho = par
    sig0, rho0 = par0
    alpha1, alpha2 = alpha

    lambda1 = -log(alpha1)/sig0
    lambda2 = -log(alpha2)*rho0

    lprior = log(lambda1) + log(lambda2) - 2*log(rho) - (lambda2/rho + lambda1*sig)

    return lprior


end

# PC prior for separable ST exponential covariance (par = [sigma, range_space, range_time])
function pcpriorST(par::Vector{Float64}, par0::Vector, alpha::Vector{Float64})

    sig, rho, rhot = par
    sig0, rho0, rhot0 = par0
    alpha1, alpha2, alpha3 = alpha

    lambda1 = -log(alpha1)/sig0
    lambda2 = -log(alpha2)*rho0
    lambda3 = -log(alpha3)*rhot0

    lprior = log(lambda1) + log(lambda2) + log(lambda3) - 2*(log(rho) + log(rhot)) - (lambda2/rho + lambda1*sig + lambda3/rhot)

    return lprior


end

# Objective function for the MAP. Using the log parametrization.
function thetayNLP(thetay::Vector, spriors::NamedTuple, data::InputData, timeKnots::Matrix, betayprec::Vector, nby::Matrix, By::SparseMatrixCSC, Fy::Vector, ByOrder::Vector, Bt::SparseMatrixCSC, BtCompact::Matrix, BtOrder::Vector, Qt::Matrix, QyPostChol::SparseArrays.CHOLMOD.Factor)

    local swy1, rangeSy1, rangeTy1, swy2, rangeTy2, t2y = exp.(thetay)

    local np = length(data.y)
    local nKnots = size(timeKnots,1)
    local kt = size(Bt, 2)
    local npUnq = Integer(kt/nKnots)

    getNNGPmatsST!(By, Fy, ByOrder, nby, data.loc, data.time, rangeSy1, rangeTy1)
    expCor!(Qt, timeKnots, rangeTy2)
    expCor!(BtCompact, data.time, timeKnots, rangeTy2)
    Bt.nzval .= view(vec(BtCompact'), BtOrder)

    local Dsgny = sparse_hcat(data.X, (swy2^2)*Bt, speye(np))

    local QyPost = blockdiag(
     spdiagm(betayprec),
     kron(speye(npUnq), swy2^2*Qt),
     (1/swy1^2)*By'*spdiagm(1 ./ Fy)*By
     ) + (1/t2y)*Dsgny'*Dsgny

    cholesky!(QyPostChol, Hermitian(QyPost))

    local ypSolve = (data.y ./ t2y) - (Dsgny*(QyPostChol \ (Dsgny'*data.y) ) ./ t2y^2)

    local sse = dot(data.y, ypSolve)

    local priorldet = sum(log.(betayprec)) + 2*kt*log(swy2) + npUnq*logdet(Qt) - 2*np*log(swy1) - sum(log.(Fy))

    local ldet = np*log(t2y) + logdet(QyPostChol) - priorldet

    local nll = 0.5*(sse + ldet)

    local prior1 = pcpriorST([swy1, rangeSy1, rangeTy1], spriors.theta10, spriors.alpha10)
    local prior2 = pcprior([swy2, rangeTy2], spriors.theta20, spriors.alpha20)

    local nlp = nll - prior1 - prior2

    return nlp


end

# Evaluate the NNGP prior log density, up to an additive constant.
function wll(B::SparseMatrixCSC, F::Vector{Float64}, s2::Real, w::AbstractArray{Float64,1})
    local n = length(w)
    local ll = -0.5*(
        norm(Diagonal(1 ./ sqrt.(F))*B*w)^2/s2 +
        sum(log.(F)) +
        n*log(s2)
    )
    return ll
end

# Evaluate the log posterior for the y spatial parameters. 
# Pretty much the negative return of thetayNLP(), but assumes that the cholesky of the posterior precision 'QyPostChol' has already been computed in the outer scope, which it has in my MCMC code, NNGP_ZIST(). 
function thetayLP(thetay::Vector, spriors::NamedTuple, yp::Vector,  betayprec::Vector, Fy::Vector, Qt::Matrix, Dsgny::SparseMatrixCSC, QyPostChol::SparseArrays.CHOLMOD.Factor)

    local swy1, rangeSy1, rangeTy1, swy2, rangeTy2, t2y = exp.(thetay)

    local np = length(yp)
    local nKnots = size(Qt,1)
    local nEffects = size(Dsgny, 2)
    local p = length(betayprec)
    local kt = nEffects - np - p
    local npUnq = Integer(kt/nKnots)
    
    local ypSolve = (yp ./ t2y) - (Dsgny*(QyPostChol \ (Dsgny'*yp) ) ./ t2y^2)

    local sse = dot(yp, ypSolve)

    local priorldet = sum(log.(betayprec)) + 2*kt*log(swy2) + npUnq*logdet(Qt) - 2*np*log(swy1) - sum(log.(Fy))

    local ldet = np*log(t2y) + logdet(QyPostChol) - priorldet

    local ll = -0.5*(sse + ldet)

    local prior1 = pcpriorST([swy1, rangeSy1, rangeTy1], spriors.theta10, spriors.alpha10)
    local prior2 = pcprior([swy2, rangeTy2], spriors.theta20, spriors.alpha20)

    local lp = ll + prior1 + prior2

    return lp


end
