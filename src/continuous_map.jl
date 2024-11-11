# Compute the MAP estimates. Unconstrained BFGS using the log parameterization.
function NNGP_Continuous_MAP(data::InputData, m::Integer, timeKnots::Matrix, initparams::NamedTuple, spriors::NamedTuple; sub = 1.0, f_tol = 1e-3, g_tol = 1e-3, alpha = 1e-6, show_trace = true, store_trace = false)
    
    ############
    # Tuple Check
    #############

    # params
    gtg = haskey(initparams, :sw1) & haskey(initparams, :rangeS1) & haskey(initparams, :rangeT1) & haskey(initparams, :sw2) & haskey(initparams, :rangeT2) & haskey(initparams, :tSq)
    if !gtg
        error("bad 'initparams': The expected fields are 'sw1, rangeS1, rangeT1, sw1, rangeT2, tSq'.") 
    end

    inittheta = log.([initparams.sw1, initparams.rangeS1, initparams.rangeT1, initparams.sw2, initparams.rangeT2, initparams.tSq])

    # spriors
    gtg = haskey(spriors, :theta10) & haskey(spriors, :alpha10) & haskey(spriors, :theta20) & haskey(spriors, :alpha20)
    if !gtg
        error("bad 'spriors': The expected fields are 'theta0, alpha0'.") 
    end

    ############################
    # Data dimensions and prep
    ##########################

    n = size(data.y, 1) # sample size
    p = size(data.X, 2) # num predictors
    nKnots = size(timeKnots, 1)

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

    sw1, rangeS1, rangeT1, sw2, rangeT2, tSq = exp.(inittheta)

    B1,F1,B1Order = getNNGPmatsST(nb, data.loc, data.time, rangeS1, rangeT1)

    B2Rows, B2Cols, B2Order = getBtNZ(n, map2unq, nKnots)
    B2Compact = expCor(data.time, timeKnots, rangeT2)
    B2 = sparse(B2Rows, B2Cols, view(vec(B2Compact'), B2Order))
    Q2 =expCor(timeKnots, rangeT2)

    betaprec = fill(0.01, p)

    Dsgn = sparse_hcat(data.X, (sw2^2)*B2, speye(n))

    QPost = blockdiag(
        spdiagm(betaprec),
        kron(speye(nUnq), sw2^2*Q2),
        (1/sw1^2)*B1'*spdiagm(1 ./ F1)*B1
    ) + (1/tSq)*Dsgn'*Dsgn

    QPostChol = cholesky(Hermitian(QPost))
    
    thetaMin = optimize(ty -> thetayNLP(ty, spriors, data, timeKnots, betaprec, nb, B1, F1, B1Order, B2, B2Compact, B2Order, Q2, QPostChol), 
                            inittheta, BFGS(alphaguess = Optim.LineSearches.InitialStatic(alpha=alpha)), 
                            Optim.Options(f_tol = f_tol, g_tol = g_tol, store_trace = store_trace, show_trace = show_trace, extended_trace = (show_trace || store_trace)))

    params = NamedTuple(zip( [:sw1, :rangeS1, :rangeT1, :sw2, :rangeT2, :tSq], exp.(Optim.minimizer(thetaMin))))


    return params, thetaMin


end
