function NNGP_Continuous_MCMC(data::InputData, m::Integer, timeKnots::Matrix, initparams::NamedTuple, spriors::NamedTuple, thetaVar::Matrix, outDir::String, nSamps::Int64; writew2 = false)

    # params
    gtg = haskey(initparams, :sw1) & haskey(initparams, :rangeS1) & haskey(initparams, :rangeT1) & haskey(initparams, :sw2) & haskey(initparams, :rangeT2) & haskey(initparams, :tSq)
    if !gtg
        error("bad 'initparams': The expected fields are 'sw1, rangeS1, rangeT1, sw1, rangeT2, tSq'.") 
    end
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
    nUnq = size(locUnq, 1)

    # For each location in the full stack, what's it's position among the set of unique locations?
    map2unq = indexin(loc2str(data.loc), loc2str(locUnq))

    kt = nUnq*nKnots

    ##########################
    # Initial values and CSV creation
    ##########################

    # Does the out_dir exist?

    if !isdir(outDir)
        error("Can't find your outDir")
    end

    # Prepare CSV's

    locOut = joinpath(outDir, "locPosUnq.csv")
    CSV.write(locOut, DataFrame(lon = locUnq[:,1], lat = locUnq[:,2]))

    loctimeOut = joinpath(outDir, "locTimePos.csv")
    CSV.write(loctimeOut, DataFrame(lon = data.loc[:,1], lat = data.loc[:,2], time = data.time[:,1]))

    knotsOut = joinpath(outDir, "timeKnots.csv")
    CSV.write(knotsOut, DataFrame(time = timeKnots[:,1]))

    paramsOut = joinpath(outDir, "yparams.csv")
    paramsDf = DataFrame(zeros(1, p + 6), 
        ["beta_".*string.(0:(p-1)); ["sw1", "rangeS1", "rangeT1", "sw2", "rangeT2", "tSq"]]
        )

    w1Out = joinpath(outDir, "wy1.csv")
    w1Df = DataFrame(zeros(1,n), "w1_".*string.(1:n))

    if writew2
        w2Out = joinpath(outDir, "w2.csv")
        w2Df = DataFrame(zeros(1, kt), "w2_l" .* string.(repeat(1:nUnq, inner = nKnots)) .* "_k" .* string.(repeat(1:nKnots, nUnq)) )
    end
    

    # Parameter/effect values

    effects = zeros(p + kt + n)
    beta = view(effects, 1:p)
    w2 = view(effects, (p+1):(p+kt))
    w1 = view(effects, (p+kt+1):(p+kt+n))

    sw1, rangeS1, rangeT1, sw2, rangeT2, tSq = initparams.sw1, initparams.rangeS1, initparams.rangeT1, initparams.sw2, initparams.rangeT2, initparams.tSq
    sw1p, rangeS1p, rangeT1p, sw2p, rangeT2p, tSqp = sw1, rangeS1, rangeT1, sw2, rangeT2, tSq


    paramsDf[1,:] = [beta; [sw1, rangeS1, rangeT1, sw2, rangeT2, tSq]]
    w1Df[1,:] = w1
    if writew2; w2Df[1,:] = w2 end

    CSV.write(paramsOut, paramsDf)
    CSV.write(w1Out, w1Df)
    if writew2; CSV.write(w2Out, w2Df) end


    ####################
    # Lord have mercy that was boring.
    # Now fun stuff. Get the neighbor sets and initial NNGP mats
    #####################


    print("Getting neighbor sets\n")

    nb = getNeighbors(data.loc, m)

    print("Initial NNGP mats\n")

    B1,F1,B1Order = getNNGPmatsST(nb, data.loc, data.time, rangeS1, rangeT1)
    #Byp,Fyp = copy(By), copy(Fy)

    B2Rows, B2Cols, B2Order = getBtNZ(n, map2unq, nKnots)
    B2Compact = expCor(data.time, timeKnots, rangeT2)
    B2 = sparse(B2Rows, B2Cols, view(vec(B2Compact'), B2Order))
    Q2 = expCor(timeKnots, rangeT2)
    #BtCompactp, Btp, Qtp = copy(BtCompact), copy(Bt), copy(Qt)

    Dsgn = sparse_hcat(data.X, sw2^2*B2, speye(n)) 

    yProj = Dsgn'*(data.y ./ tSq)

    betaPrec = fill(0.01, p)

    ##############

    Q = blockdiag(
        spdiagm(betaPrec),
        kron(speye(nUnq), sw2^2*Q2),
        (1/sw1^2)*B1'*spdiagm(1 ./ F1)*B1
    ) + (1/tSq)*Dsgn'*Dsgn

    Qc = cholesky(Hermitian(Q))
    Qpc = copy(Qc)

    currentTheta = log.([sw1, rangeS1, rangeT1, sw2, rangeT2, tSq])
    propTheta = copy(currentTheta)
    lp = thetayLP(currentTheta, spriors, data.y, betaPrec, F1, Q2, Dsgn, Qc)
    lpProp = lp
    acceptTheta = false

    prop_chol = cholesky(thetaVar).L

    lpDf = DataFrame(lp = lp, lpProp = lpProp, accept = acceptTheta*1)
    lpOut = joinpath(outDir, "lp.csv")
    CSV.write(lpOut, lpDf)


    #########################
    # Begin Gibbs sampler
    #########################

    for i = ProgressBar(1:nSamps)

       ########################
       # Sample beta, w1, w2
       ########################

       effects .= getGaussSamp(Qc, yProj)

       ###########################
       # Sample all spatial parameters associated with y
       ###########################

       lpDf.lp[1] = lp

       propTheta = currentTheta + prop_chol*randn(6)

       sw1p, rangeS1p, rangeT1p, sw2p, rangeT2p, tSqp = exp.(propTheta)

       # Get NNGP and fixed-rank matrices associated with the proposal values
       getNNGPmatsST!(B1, F1, B1Order, nb, data.loc, data.time, rangeS1p, rangeT1p)
       expCor!(Q2, timeKnots, rangeT2p)
       expCor!(B2Compact, data.time, timeKnots, rangeT2p)
       B2.nzval .= view(vec(B2Compact'), B2Order)
       # Posterior precision for the proposal values
       Dsgn .= sparse_hcat(data.X, sw2p^2*B2, speye(n))
       Q .= blockdiag(
            spdiagm(betaPrec),
            kron(speye(nUnq), sw2p^2*Q2),
            (1/sw1p^2)*B1'*spdiagm(1 ./ F1)*B1
       ) + (1/tSqp)*Dsgn'*Dsgn

       cholesky!(Qpc, Hermitian(Q))

       lpProp = thetayLP(propTheta, spriors, data.y, betaPrec, F1, Q2, Dsgn, Qpc)

       acceptProb = exp.(lpProp + sum(propTheta)  - lp - sum(currentTheta))

       acceptTheta = rand(1)[1] < acceptProb

       if acceptTheta
        sw1, rangeS1, rangeT1, sw2, rangeT2, tSq = exp.(propTheta)
        Qc = copy(Qpc)
        lp = lpProp
        yProj .= Dsgn'*(data.y ./ tSq)
        currentTheta .= copy(propTheta)
       end

       lpDf.lpProp[1] = lpProp

       lpDf.accept[1] = acceptTheta*1

       #########################
       # Good work, team. Let's write out these results
       #########################

       paramsDf[1,:] = [beta; [sw1, rangeS1, rangeT1, sw2, rangeT2, tSq]]
       w1Df[1,:] = w1

       CSV.write(paramsOut, paramsDf; append = true, header = false)
       CSV.write(w1Out, w1Df; append = true, header = false)
       CSV.write(lpOut, lpDf; append = true, header = false)

       if writew2
        w2Df[1,:] = w2
        CSV.write(w2Out, w2Df; append = true, header = false)
       end



    end



    return nothing


end