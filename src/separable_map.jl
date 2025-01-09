# Objective function for the MAP. Using the log parametrization.
function thetayNLP_sep(thetay::Vector, spriors::NamedTuple, data::InputData, betayprec::Vector, nby::Matrix, By::SparseMatrixCSC, Fy::Vector, ByOrder::Vector, QyPostChol::SparseArrays.CHOLMOD.Factor)

    local sw, rangeS, rangeT, t2y = exp.(thetay)

    local np = length(data.y)

    getNNGPmatsST!(By, Fy, ByOrder, nby, data.loc, data.time, rangeS, rangeT)

    local Dsgny = sparse_hcat(data.X, speye(np))

    local QyPost = blockdiag(
     spdiagm(betayprec),
     (1/sw^2)*By'*spdiagm(1 ./ Fy)*By
     ) + (1/t2y)*Dsgny'*Dsgny

    cholesky!(QyPostChol, Hermitian(QyPost))

    local ypSolve = (data.y ./ t2y) - (Dsgny*(QyPostChol \ (Dsgny'*data.y) ) ./ t2y^2)

    local sse = dot(data.y, ypSolve)

    local priorldet = sum(log.(betayprec)) - 2*np*log(sw) - sum(log.(Fy))

    local ldet = np*log(t2y) + logdet(QyPostChol) - priorldet

    local nll = 0.5*(sse + ldet)

    local prior1 = pcpriorST([sw, rangeS, rangeT], spriors.theta0, spriors.alpha0)

    local nlp = nll - prior1 - prior2

    return nlp


end



# Compute the MAP estimates. Unconstrained BFGS using the log parameterization.
function separable_map(data::InputData, m::Integer, initparams::NamedTuple, spriors::NamedTuple; sub = 1.0, f_tol = 1e-3, g_tol = 1e-3, alpha = 1e-6, show_trace = true, store_trace = false)
    
    ############
    # Tuple Check
    #############

    # params
    gtg = haskey(initparams, :sw) & haskey(initparams, :rangeS) & haskey(initparams, :rangeT) & haskey(initparams, :tSq)
    if !gtg
        error("bad 'initparams': The expected fields are 'sw, rangeS, rangeT, tSq'.") 
    end

    inittheta = log.([initparams.sw, initparams.rangeS, initparams.rangeT, initparams.tSq])

    # spriors
    gtg = haskey(spriors, :theta0) & haskey(spriors, :alpha0) 
    if !gtg
        error("bad 'spriors': The expected fields are 'theta0, alpha0'.") 
    end

    ############################
    # Data dimensions and prep
    ##########################

    n = size(data.y, 1) # sample size
    p = size(data.X, 2) # num predictors

    ##########################################################
    # I am making the forest/non-forest discrims time static.
    # I also have differing random errors across space (expected to be larger) than time (expected to be smaller)
    ##########################################################

    # Unique plot locations
    locUnq = unique(data.loc, dims = 1)

    nUnq = size(locUnq,1)

    if nUnq == n
        error("No repeat locations... this model will not be identifiable (in particular, sw2, rangeT2 and tSq)")
    end

    # For each location in the full stack, what's it's position among the set of unique locations?
    map2unq = indexin(loc2str(data.loc), loc2str(locUnq))

    # Organize this is a sparse matrix. Entry (i,j) is one if row i is associated with unique location j
    P = sparse(1:n, map2unq, 1)

    # Are we subsetting the data?
    if sub < 1
        nUnqOG = size(P,2)
        nUnq = Integer(floor(sub*size(P,2)))
        cols2keep = sort(shuffle(1:nUnqOG)[1:nUnq])
        P = P[:,cols2keep]
        rows2keep = sum(P, dims = 2)[:,1] .> 0
        P = P[rows2keep,:]
        data = InputData(data.y[rows2keep], data.X[rows2keep,:], data.loc[rows2keep,:], data.time[rows2keep,:])
        n = size(data.y, 1) # sample size
        locUnq = unique(data.loc, dims = 1)
        map2unq = indexin(loc2str(data.loc), loc2str(locUnq))
        println("Working with subsetted data, "*string(n), " points")
    end

    nb = getNeighbors(data.loc, m)

    sw, rangeS, rangeT, tSq = exp.(inittheta)

    B,F,BOrder = getNNGPmatsST(nb, data.loc, data.time, rangeS, rangeT)

    betaprec = fill(0.01, p)

    Dsgn = sparse_hcat(data.X, speye(n))

    QPost = blockdiag(
        spdiagm(betaprec),
        (1/sw^2)*B'*spdiagm(1 ./ F)*B
    ) + (1/tSq)*Dsgn'*Dsgn

    QPostChol = cholesky(Hermitian(QPost))
    
    thetaMin = optimize(ty -> thetayNLP_sep(ty, spriors, data, betaprec, nb, B, F, BOrder, QPostChol), 
                            inittheta, BFGS(alphaguess = Optim.LineSearches.InitialStatic(alpha=alpha)), 
                            Optim.Options(f_tol = f_tol, g_tol = g_tol, store_trace = store_trace, show_trace = show_trace, extended_trace = (show_trace || store_trace)))

    params = NamedTuple(zip( [:sw, :rangeS, :rangeT, :tSq], exp.(Optim.minimizer(thetaMin))))


    return params, thetaMin


end
